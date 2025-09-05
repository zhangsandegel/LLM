from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
from reward_func import * # 奖励函数来源，自己定义的奖励规则
import os
from accelerate import Accelerator
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class GSM8KDataset(Dataset):
    """
    支持三种 data_path:
      1) 本地HF数据集目录: 例如 /home/work/zhangyu/data/gsm8k_chinese
         (目录内是 snapshot_download 的完整数据集结构)
      2) 本地 JSONL: 目录下存在 train.jsonl
         每行包含:
            {"question_zh-cn": "...", "answer_only": "..."}
         或者:
            {"question_zh": "...", "answer_zh": "..."} / {"question": "...", "answer": "..."} 等
      3) 线上数据集ID: 例如 "meta-math/GSM8K_zh" 或 "openai/gsm8k"
    初始化后，self.data 的每条样本都标准化为:
        {"question_zh-cn": <str>, "answer_only": <str>}
    """
    def __init__(self, data_path, tokenizer, split: str = "train"):
        self.tokenizer = tokenizer

        # 1) 判定本地目录 or 线上ID
        if os.path.isdir(data_path):
            jsonl = os.path.join(data_path, "train.jsonl")
            if os.path.exists(jsonl):
                data = load_dataset("json", data_files={split: jsonl})
            else:
                # 本地HF数据集目录（snapshot_download 拉下来的）
                data = load_dataset(data_path)
        else:
            # 线上数据集ID
            data = load_dataset(data_path)

        # 2) 选择 split（若不存在则回退 train/test 任一）
        if split not in data:
            split = "train" if "train" in data else list(data.keys())[0]
        ds = data[split]

        # 3) 统一字段，并尽量抽取“最终答案”
        def _extract_final_answer(a: str) -> str:
            if not isinstance(a, str):
                return str(a)
            s = a.strip()

            # 常见格式1: "... #### 42"
            m = re.search(r"####\s*(.+)$", s)
            if m:
                return m.group(1).strip()

            # 常见格式2: "答案是：42" / "答案: 42"
            m = re.search(r"(?:答案是|答案[:：])\s*(.+)$", s)
            if m:
                return m.group(1).strip()

            # 兜底：取最后一个数字（若存在），否则原文
            nums = re.findall(r"[-+−]?\d+(?:\.\d+)?", s)
            return nums[-1] if nums else s

        def _to_standard(ex):
            q = (ex.get("question_zh-cn")
                 or ex.get("question_zh")
                 or ex.get("question")
                 or ex.get("query"))
            a = (ex.get("answer_only")
                 or ex.get("answer_zh")
                 or ex.get("final_answer")
                 or ex.get("answer"))

            # 必要性检查（尽量不让 None 进训练）
            if q is None and "prompt" in ex:  # 有些数据叫 prompt
                q = ex["prompt"]
            if q is None:
                q = ""

            if a is None:
                a_std = ""
            else:
                a_std = _extract_final_answer(a)

            return {"question_zh-cn": q, "answer_only": a_std}

        std = ds.map(_to_standard, remove_columns=ds.column_names)
        self.data = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ex = self.data[index]
        # 直接返回“问题 + 答案”（不在这里套 chat_template，生成阶段再套）
        return {"prompt": ex["question_zh-cn"], "answer": ex["answer_only"]}
# 一题的一组（K条）样本
@dataclass
class Samples:
    # 拼接了 prompt+response 的 token id
    prompt_response_ids: torch.Tensor
    # 只含回答 token id
    response_ids: torch.Tensor
    prompt: Any
    answer: Any
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    # 回答token数
    num_actions: Union[int, torch.Tensor]
    response_length: int

# 训练参数
class GRPOArguments:
    output_dir = '/home/work/zhangyu/res/Qwen2-1.5B-GRPO'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.000001
    save_steps = 100
    epoch = 3
    num_generations = 4 # 组内样本数 GRPO中的G
    # 截断与生成长度
    max_prompt_length = 256 # 最大输入长度 
    max_generate_length = 256 # 最大输出长度
    reward_weights : List[float] = None # 奖励的权重（多个奖励函数）
    beta = 0.0 # KL散度的系数，为0则忽略KL散度，即不使用参考模型 （>0 时会启用参考模型约束，使新策略不至于飘太远）
    clip_eps = 0.2 #PPO的裁剪半径 （0.1～0.3）
    gradient_accumulation_steps = 2 # 梯度累加步数
    num_iterations = 1 # 采样一次样本训练模型轮数，一批经验重复训练几轮（不必每轮都重新生成）
    batch_size = 1

class GRPOTrainer:
    # 训练器初始化
    def __init__(self,
        # 可以是一个模型实例或者模型名称的字符串。如果是字符串，将通过AutoModelForCausalLM.from_pretrained加载预训练模型。
        model = None,
        # 奖励函数的列表，可以是函数或者是奖励模型名称的字符串。
        reward_funcs: Union[List[str], List[Callable]] = None,
        # 包含训练参数的对象，例如学习率（lr）和设备类型（device）。
        args = None,
        train_dataset: Optional[Union[Dataset]] = None,
        eval_dataset: Optional[Union[Dataset]] = None,
        # 用于文本编码的分词器，可以是分词器实例或者是分词器名称的字符串
        tokenizer = None,

        # 奖励函数专用的分词器列表。
        reward_tokenizers = None,
        accelerator: Accelerator=None,          # <<< 新增
        writer: SummaryWriter=None              # <<< 新增
        ):
        # 包含训练参数的对象，例如学习率（lr）和设备类型（device）。
        self.args = args
        self.accelerator = accelerator          # <<< 保存
        self.writer = writer                    # <<< 保存

        # 加载模型，如果传进来是字符串，就 from_pretrained[模型初始化]
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        # 将模型移动到args.device指定的设备上（例如GPU）。
        self.model = model.to(self.args.device)
        
        # 是否使用参考模型
        self.ref_model = None
        if self.args.beta != 0.0:
            self.ref_model = deepcopy(model).eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
    
        # 分瓷器初始化
        if isinstance(tokenizer, str):
            # 如果tokenizer是字符串，使用AutoTokenizer.from_pretrained方法从预训练模型加载对应的分词器。
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # 调用self.get_tokenizer(tokenizer)方法（这个方法在给出的代码段中没有定义）来获取最终的分词器。
        self.tokenizer = self.get_tokenizer(tokenizer)
        
        # 奖励函数处理:
        if isinstance(reward_funcs, str):
            # 如果reward_funcs是字符串，将其转换为包含一个元素的列表。
            reward_funcs = [reward_funcs]
        
        # 遍历reward_funcs，如果奖励函数是字符串，则使用AutoModelForSequenceClassification.from_pretrained方法加载预训练的奖励模型，并将其移动到args.device指定的设备上
        for i, reward_func in enumerate(reward_funcs):
            # 如果奖励函数为字符串，表示使用的是奖励模型，则加载模型
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1).to(self.args.device)
        
        self.reward_funcs = reward_funcs
        
        # 奖励分词器处理:
        # 如果没有提供reward_tokenizers，则创建一个与reward_funcs长度相同的None值列表。
        if reward_tokenizers is None:
            reward_tokenizers = [None] * len(reward_funcs)
        # 如果reward_tokenizers是字符串，将其转换为包含一个元素的列表。
        elif isinstance(reward_tokenizers, str):
            reward_tokenizers = [reward_tokenizers]
        #如果reward_tokenizers是列表，检查其长度是否与reward_funcs相同，否则抛出异常。
        else:
            if len(reward_tokenizers) != len(reward_funcs):
                raise ValueError("Length of reward_tokenizers must be equal to the number of reward_funcs.")
        # 对于每个奖励函数和对应的奖励分词器，如果奖励函数是预训练模型且奖励分词器为None，则从奖励模型配置中加载对应的分词器。
        for i, (reward_tokenizer, reward_func) in enumerate(zip(reward_tokenizers, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_tokenizer is None:
                    reward_tokenizer = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                # 如果奖励分词器的pad_token_id为None，则将其设置为eos_token。
                if reward_tokenizer.pad_token_id is None:
                    reward_tokenizer.pad_token = reward_tokenizer.eos_token
                # 更新奖励模型的pad_token_id配置
                reward_func.config.pad_token_id = reward_tokenizer.pad_token_id
                reward_tokenizers[i] = reward_tokenizer
        self.reward_tokenizers = reward_tokenizers
        # 优化器设置:
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=self.args.lr)
        except Exception:
            from transformers import Adafactor
            self.optimizer = Adafactor(
                self.model.parameters(), lr=self.args.lr,
                relative_step=False, scale_parameter=False, warmup_init=False
            )

        # 数据集和缓存:
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # 缓存已经生成的数据的一个批次的数据，可供模型多次训练迭代，无需重新生成
        self.input_buffer = [None] * self.args.gradient_accumulation_steps
        
        # 模型更新的次数
        self.update_steps = 0 

    # 对tokenizer对象进行配置。这个设置指定了在对齐文本时，填充（padding）应该添加在序列的左侧还是右侧。在这个例子中，填充被设置在左侧
    def get_tokenizer(self, tokenizer):
        tokenizer.padding_side = "left"
        return tokenizer
    
    # 生成样本，以组为单位
    def generate_samples(self, inputs):
        # 初始化一个空列表samples_list，用于存储生成的样本。然后，将模型设置为评估模式（self.model.eval()），这是因为在生成样本时，我
        samples_list = []
        self.model.eval()
        # 从输入的字典inputs中提取prompt列表，并初始化一个与prompts等长的answers列表，其元素初始值为None。
        # 如果inputs字典中包含answer键，则使用其对应的值填充answers列表
        prompts = [prompt for prompt in inputs['prompt']]
        answers = [None] * len(prompts)
        
        if 'answer' in inputs:
            answers = [answer for answer in inputs['answer']]
        # 根据类的参数args设置生成文本的最大长度，这个长度是生成文本的最大长度（max_generate_length）和提示的最大长度（max_prompt_length）之和
        max_length = self.args.max_generate_length + self.args.max_prompt_length
          # ★ 关键：拿到底层模型（加速器存在就用 unwrap_model，否则用原模型）
        base_model = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model
        device = next(base_model.parameters()).device

        for prompt, answer in zip(prompts, answers):
            # 应用聊天模板，加入系统提示词
            input_text = self.tokenizer.apply_chat_template([{"role": "system", 'content': SYSTEM_PROMPT}, {"role": "user", 'content': prompt}], add_generation_prompt=True, tokenize=False)
            
            # 生成一个group的输入数据
            inputs_tok = self.tokenizer([input_text] * self.args.num_generations, padding='max_length', max_length=self.args.max_prompt_length, truncation=True, return_tensors='pt')
            prompt_ids = inputs_tok['input_ids']
            with torch.no_grad():
                model_inputs = {k: v.to(device) for k, v in inputs_tok.items()}
                # prompt_response_ids = self.model.generate(
                #     **model_inputs,
                #     max_new_tokens=self.args.max_generate_length,
                #     temperature=0.9,
                #     top_p=1,
                #     top_k=50
                # )
                  # ★ 用底层 HF 模型来 generate（DDP 不支持 generate）
                prompt_response_ids = base_model.generate(
                    **model_inputs,
                    max_new_tokens=self.args.max_generate_length,
                    temperature=0.9,
                    top_p=1,
                    top_k=50
                )
                    
            if prompt_response_ids.size(1) >= max_length:
                prompt_response_ids = prompt_response_ids[:, :max_length]
            else:
                prompt_response_ids = torch.cat([prompt_response_ids, torch.full((prompt_response_ids.size(0), max_length - prompt_response_ids.size(1)), fill_value=self.tokenizer.pad_token_id, device=prompt_response_ids.device)], dim=1)
          
            attention_mask = (prompt_response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
            response_ids = prompt_response_ids[:, prompt_ids.size(1):]
            action_mask = (response_ids.ne(self.tokenizer.eos_token_id) & response_ids.ne(self.tokenizer.pad_token_id)).to(dtype=torch.long)
        

            # 存储的是一个group的数据
            samples = Samples(
                prompt_response_ids=prompt_response_ids,
                response_ids=response_ids,
                prompt = prompt,
                answer = answer,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                response_length=action_mask.float().sum(dim=-1)
            )
            samples_list.append(samples)

        return samples_list
    
    # 生成经验(优势、token的概率分布)
    def generate_experiences(self, inputs):
        
        self.model.eval()
        samples_list = self.generate_samples(inputs)
        
        batch_prompt_response_ids = []
        batch_attention_mask = []
        batch_action_mask = []
        batch_advantages = []
        batch_old_action_log_probs = []
        batch_ref_action_log_probs = []
        
        # 采样组内回答 
        for samples in samples_list:
            prompt_response_ids = samples.prompt_response_ids # shape: (num_generations, seq_len) （G，seq_len)
            response_ids = samples.response_ids # shape: (num_generations, seq_len) (G, |O_i|)
            answer = samples.answer
            attention_mask = samples.attention_mask # shape: (num_generations, seq_len) (G,|O_i|)
            action_mask = samples.action_mask # shape: (num_generations, seq_len)
            num_actions = samples.num_actions
            prompt = samples.prompt
            batch_prompt_response_ids.append(prompt_response_ids)
            batch_attention_mask.append(attention_mask)
            batch_action_mask.append(action_mask)
            
            with torch.no_grad():
                # 计算策略模型输出token的概率
                old_action_log_probs = self.get_action_log_probs(self.model, prompt_response_ids, attention_mask, num_actions)
                batch_old_action_log_probs.append(old_action_log_probs)
                
                # 是否使用参考模型
                if self.ref_model:
                    #计算参考模型输出token的概率
                    ref_action_log_probs = self.get_action_log_probs(self.ref_model, prompt_response_ids, attention_mask, num_actions)
                    batch_ref_action_log_probs.append(ref_action_log_probs)
                
                # 奖励与组内标准化（对A i，t）
                # 存储各个奖励函数在一个group内各个响应的奖励
                # ★ 取模型设备
                device = next(self.model.parameters()).device
                rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=device)
                # rewards_per_func = torch.zeros(len(self.reward_funcs), self.args.num_generations, device=self.args.device)
                
                # 将输出转换成文本
                response_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                prompt_texts = [prompt] * len(response_texts)
                prompt_response_texts = [prompt + response for prompt, response in zip(prompt_texts, response_texts)]
                
                for i, (reward_func, reward_tokenizer) in enumerate(
                    zip(self.reward_funcs, self.reward_tokenizers)
                ):
                    if isinstance(reward_func, PreTrainedModel):
                        with torch.inference_mode():
                            reward_model_inputs = reward_tokenizer(prompt_response_texts, return_tensors="pt", padding=True)
                            # rewards_per_func[i] = reward_func(**reward_model_inputs.to(self.args.device)).logits.squeeze(-1)
                            rewards_per_func[i] = reward_func(**{k: v.to(device) for k, v in reward_model_inputs.items()}).logits.squeeze(-1)
                    
                    else:
                        answers = [answer] * len(prompt_texts)
                        output_reward_func = reward_func(prompts=prompt_texts, responses=response_texts, answers=answers)
                        output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                        # rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=self.args.device)
                        rewards_per_func[i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
                
                # rewards_per_func: [num_funcs, num_generations]
                if not self.args.reward_weights:
                    self.args.reward_weights = [1.0] * len(self.reward_funcs)
                if len(self.args.reward_weights) != len(self.reward_funcs):
                    raise ValueError("The number of reward weights must be equal to the number of reward functions.")
                # 乘以各个奖励函数的权重
                rewards = rewards_per_func * torch.tensor(self.args.reward_weights, dtype=torch.float32, device=rewards_per_func.device).unsqueeze(1)
                
                # rewards: [num_funcs, num_generations]
                rewards = rewards.sum(dim=0) # shape: [num_generations]
                print(f'rewards: {rewards}')
                mean_group_rewards = rewards.mean()
                std_group_rewards = rewards.std()
                
                # GRPO的优势是句子粒度的，而非token粒度的
                advantages = (rewards - mean_group_rewards) / (std_group_rewards + 1e-8) # shape: [num_generations]
                batch_advantages.append(advantages)
        
               
        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if self.ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }
    
    def compute_loss(self, model, inputs):
        
        prompt_response_ids = inputs['prompt_response_ids']
        attention_mask = inputs['attention_mask']
        action_mask = inputs['action_mask']
        num_actions = action_mask.size(1)
        action_log_probs = self.get_action_log_probs(model, prompt_response_ids, attention_mask, num_actions)
        
        if self.args.beta != 0.0:
            
            ref_action_log_probs = inputs['ref_action_log_probs']
            log_ratio = ref_action_log_probs - action_log_probs 
            log_ratio = log_ratio * action_mask
            
            # k3: log_ratio.exp() - 1 - log_ratio
            k3 = log_ratio.exp() - 1 - log_ratio
        
        advantages = inputs['advantages']
        
        old_action_log_probs = inputs['old_action_log_probs'] if self.args.num_iterations > 1 else action_log_probs.detach()
        coef_1 = torch.exp(action_log_probs - old_action_log_probs) # 重要性采样 shape: [batch_size * num_generations, num_actions]
        coef_2 = torch.clamp(coef_1, 1 - self.args.clip_eps, 1 + self.args.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1) # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask
        if self.args.beta != 0.0:
            per_token_loss = per_token_loss + self.args.beta * k3
        
        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1) # shape: [batch_size * num_generations]
        loss = loss.mean()
        
        # loss = per_token_loss.sum() / action_mask.sum()
        
        return loss


    def get_action_log_probs(self, model, input_ids, attention_mask, num_actions):
        
        # 计算策略模型输出token的概率
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)[:, -num_actions:]
        return action_log_probs

    
    
    def train_step(self, model, inputs, optimizer, step):
        model.train()
        # scaler = torch.amp.GradScaler()
        # with torch.amp.autocast(device_type='cuda'):
        # loss = self.compute_loss(model, inputs)
        # ★ 混合精度：有 accelerator 就用它的 autocast；否则按原逻辑
        if self.accelerator is not None:
            with self.accelerator.autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)
        loss = loss / self.args.gradient_accumulation_steps

            # loss = scaler.scale(loss)
        # loss.backward()
        # ★ 反传：有 accelerator 就用它（会自动做 GradScaler + DDP 同步），否则用原生 backward
        if self.accelerator is not None:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        if (step + 1) % self.args.gradient_accumulation_steps == 0:
            
            optimizer.step()
            optimizer.zero_grad()
            # 只在主进程写日志/打印，避免多进程重复
            if self.accelerator is None or self.accelerator.is_main_process:
                writer.add_scalar("grpo_loss", loss.item(), self.update_steps)
                print(f"step: {self.update_steps}/{self.global_steps}  grpo_loss: {loss.item():.8f}")         
        torch.cuda.empty_cache()

        # 在 GRPOTrainer 里新增一个内部保存函数
    def _save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
            unwrapped = self.accelerator.unwrap_model(self.model)
            if self.accelerator.is_main_process:
                # 用 accelerator.save 更稳（兼容某些并行/零冗余场景）
                unwrapped.save_pretrained(output_dir, save_function=self.accelerator.save)
                self.tokenizer.save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

    def train(self):
        self.global_steps = self.args.num_iterations * self.args.epoch * len(self.train_dataset) // (self.args.batch_size * self.args.gradient_accumulation_steps)
        # 先创建 dataloader（放在 epoch 外）
        dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

        # 让 accelerate 接管模型/优化器/数据加载器（DDP）
        if self.accelerator is not None:
            self.model, self.optimizer,dataloader = self.accelerator.prepare(self.model, self.optimizer,dataloader)
            # self.model, self.optimizer, self.dataloader = self.accelerator.prepare(self.model, self.optimizer, self.dataloader)
            if self.ref_model is not None:
                # 参考模型只搬到同设备，不做 prepare（避免参与梯度与梯度同步）
                self.ref_model.to(next(self.model.parameters()).device)
        for _ in range(self.args.epoch):
            # dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            for idx, batch in enumerate(dataloader):
                
                inputs = self.generate_experiences(batch)
                self.input_buffer[idx % self.args.gradient_accumulation_steps] = inputs
                if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                   
                    for _ in range(self.args.num_iterations):
                        for step, inputs in enumerate(self.input_buffer):
                            self.train_step(self.model, inputs, self.optimizer, step)
                        
                        self.update_steps += 1
                        if self.update_steps % self.args.save_steps == 0:
                            self._save(os.path.join(self.args.output_dir, f"checkpoint_{self.update_steps}"))
                del inputs
    def save_model(self):
        self._save(self.args.output_dir)          

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="no")  # ★ 开启 fp16 + DDP
    
    SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""
    
    args = GRPOArguments()
    
    writer = SummaryWriter('./runs')
    # model.gradient_checkpointing_enable()
    # 策略模型
    tokenizer = AutoTokenizer.from_pretrained('/home/work/zhangyu/models/Qwen2-1.5B-Instruct')
    model = AutoModelForCausalLM.from_pretrained('/home/work/zhangyu/models/Qwen2-1.5B-Instruct',dtype=torch.float16,low_cpu_mem_usage=True)
    # 奖励函数
    # reward_model = '/home/user/Downloads/reward-model-deberta-v3-large-v2'
    # reward_tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/reward-model-deberta-v3-large-v2')

    # （可选强烈建议）省显存三件套：
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    prompts_dataset = GSM8KDataset('/home/work/zhangyu/data/gsm8k_chinese', tokenizer)
  
    trainer = GRPOTrainer(model=model,
                          reward_funcs = [correctness_reward, digit_reward, hard_format_reward, mark_reward],
                          args=args,
                          train_dataset=prompts_dataset,
                          tokenizer=tokenizer, 
                          accelerator=accelerator) #  # ★ 传入加这一项即可
    trainer.train()
    trainer.save_model()
    
