import torch
import torch.nn as nn

# 仅示例：长度奖励，用于演示如何产生 r_i
def reward_len(completions, **kw):
    # 让长度越接近 20 越好
    return [-abs(20 - len(c)) for c in completions]

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, reward_funcs=[reward_len], reward_weights=None, beta=0.02, num_generations=4):
        self.model = model                   # 训练中的策略 πθ
        self.ref_model = ref_model           # 参考策略 π_ref（冻结）
        self.tok = tokenizer
        self.reward_funcs = reward_funcs
        self.reward_weights = torch.tensor(reward_weights or [1.0]*len(reward_funcs), dtype=torch.float)
        self.beta = beta                     # KL 系数
        self.num_generations = num_generations

    def _prepare_inputs(self, batch):
        """
        batch: list of dicts, each含一个 prompt；我们会为每个 prompt 生成 num_generations 个 completion
        返回一个 dict，含：
          - prompt_ids / prompt_mask
          - completion_ids / completion_mask
          - ref_per_token_logps
          - advantages
        """
        prompts = [x["prompt"] for x in batch]

        # 1) 用当前模型生成多个回答（这里省略采样细节，你可以替换为 vLLM 或 HF generate）
        #    结果拼接成：B * G 个样本
        completions = []
        for p in prompts:
            gens = [p + " ...dummy completion..." for _ in range(self.num_generations)]
            completions.extend(gens)

        # 2) 计算每个 completion 的多路奖励，并加权求和  
        with torch.no_grad():
            rewards_per_func = []
            for f in self.reward_funcs:
                rewards_per_func.append(torch.tensor([f([c])[0] for c in completions], dtype=torch.float))
            rewards_per_func = torch.stack(rewards_per_func, dim=1)         # [B*G, R]
            rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).sum(dim=1)  # [B*G]

            # 3) 组内归一化得到 advantages  —— 截图第 27~32 行
            B = len(prompts)
            G = self.num_generations
            rewards = rewards.view(B, G)                                    # [B, G]
            mean_grouped = rewards.mean(dim=1, keepdim=True)                # [B, 1]
            std_grouped = rewards.std(dim=1, keepdim=True)                  # [B, 1]
            advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)    # [B, G]
            advantages = advantages.view(-1)                                 # [B*G]

        # 4) 计算参考模型的 per-token logp —— 与截图里 ref_per_token_logps 一致
        #    这里演示写法：真实代码应把 prompt+completion 拼成 input_ids，再喂 ref_model 得到 per-token logp
        ref_per_token_logps = torch.zeros((len(completions), 16))            # [B*G, T] 仅占位

        # 5) 同上，准备 prompt/completion 的 ids 和 mask（这里也用占位）
        prompt_ids = torch.zeros((len(completions), 8), dtype=torch.long)
        completion_ids = torch.zeros((len(completions), 16), dtype=torch.long)
        prompt_mask = (prompt_ids != 0).to(torch.float)
        completion_mask = torch.ones_like(completion_ids, dtype=torch.float)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,     # [B*G]
        }

    # 
    def compute_loss(self, model, inputs):
        """
        逐 token 计算：
          per_token_logps     : 当前模型 πθ 的 log p
          per_token_kl        : KL 近似：exp(Δ) - Δ - 1,  Δ = log p_ref - log p_theta
          per_token_loss_core : exp(logp - logp.detach()) * A   （梯度等价于  A * ∇logp）
          loss                : -( core - beta * KL )
        """
        # 取得 πθ 的 per-token logp
        per_token_logps = self._get_per_token_logps(
            model,
            inputs["prompt_ids"],
            inputs["completion_ids"],
            inputs["prompt_mask"],
            inputs["completion_mask"],
        )                                # 形状: [B*G, T]

        # 参考模型 per-token logp（已在 _prepare_inputs 中算好）
        ref_per_token_logps = inputs["ref_per_token_logps"]                  # [B*G, T]

        # KL 的稳定实现（逐 token）
        delta = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(delta) - delta - 1.0                        # [B*G, T]

        # 组内优势 A，与 “stop-grad 比率” trick
        advantages = inputs["advantages"].unsqueeze(-1)                       # [B*G, 1]
        # 等价于对 -A * logπ 求梯度的实现：前向值≈A，但反向梯度是 A * ∇logπ
        per_token_core = torch.exp(per_token_logps - per_token_logps.detach()) * advantages  # [B*G, T]

        # 拼合最终 per-token loss，再第 50 行用 mask 做平均
        completion_mask = inputs["completion_mask"]                           # [B*G, T]
        per_token_loss = -(per_token_core - self.beta * per_token_kl)         # 负号：做最小化
        loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1.0)
        # 取 batch 平均
        return loss.mean()

    # 计算当前模型的 per-token logp（示例桩）
    def _get_per_token_logps(self, model, prompt_ids, completion_ids, prompt_mask, completion_mask):
        # 实际：拼接 prompt+completion -> logits -> 对应 token 的 log_softmax -> 取 label 对应位置
        return torch.zeros_like(completion_ids, dtype=torch.float)
