[TOC]

### GRPOTrainer 代码详解（按执行流程排序）

```python
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
        prompts = [x["prompt"] for x in batch]
        # 1) 用当前模型生成多个回答（这里省略采样细节，你可以替换为 vLLM 或 HF generate）
        #    结果拼接成：B * G 个样本
        completions = []
        for p in prompts:
             # 为每个提示词生成 num_generations 个回答
            gens = [p + " ...dummy completion..." for _ in range(self.num_generations)]
            completions.extend(gens)  #[B,G] = 2*4 = 8

        # 2) 计算每个 completion 的多路奖励，并加权求和
        # 禁用梯度计算，因为奖励计算不需要更新模型参数，这样可以节省内存并加速计算
        with torch.no_grad():
            rewards_per_func = []
            # 遍历每个奖励函数，计算其对所有回答的奖励值 【奖励函数列表】
            for f in self.reward_funcs:
                # 对每个回答计算奖励，并转换成张量，对每个回答c应用奖励函数f，并取结果的第一个元素
                rewards_per_func.append(torch.tensor([f([c])[0] for c in completions], dtype=torch.float))
            # 将多个奖励函数的结果从列表转换为二维张量
            # 假设我们有 8 个回答（B×G=8）和 2 个奖励函数（R=2）
            # 堆叠后形状为[8, 2]，即每行是一个回答，每列是一个奖励函数的结
            rewards_per_func = torch.stack(rewards_per_func, dim=1)         # [B*G, R]
            # .to(rewards_per_func.device)：确保权重张量和奖励张量在同一设备（CPU/GPU），.unsqueeze(0)：增加一个维度，使权重形状从[R]变为[1, R]，以便与[B*G, R]的奖励张量相乘，相乘后进行sum(dim=1)：按行求和，得到每个回答的总奖励
            rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).sum(dim=1)  # [B*G]

            # 3) 组内归一化得到 advantages 
            B = len(prompts)
            G = self.num_generations
            rewards = rewards.view(B, G)                                    # [B, G]
            # # dim=1：沿第1维（行内的列方向）计算均值（即每个提示词组内的平均奖励）
            # keepdim=True：保持维度不变（结果形状为 [B, 1] 而非 [B]）
            mean_grouped = rewards.mean(dim=1, keepdim=True)                # [B, 1]
            std_grouped = rewards.std(dim=1, keepdim=True)                  # [B, 1]
            advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)    # [B, G]
            advantages = advantages.view(-1)                                 # [B*G]

        #  这里演示写法：真实代码应把 prompt+completion 拼成 input_ids，再喂 ref_model 得到 per-token logp
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
        # 填充token
        # 如果某个样本的有效 token 数为 0，clamp_min(1.0) 后变为 1.0，确保除法有效
        loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1.0)
        # 取 batch 平均
        return loss.mean()

    # 计算当前模型的 per-token logp（示例桩）
    def _get_per_token_logps(self, model, prompt_ids, completion_ids, prompt_mask, completion_mask):
        # 实际：拼接 prompt+completion -> logits -> 对应 token 的 log_softmax -> 取 label 对应位置
        return torch.zeros_like(completion_ids, dtype=torch.float)

```

#### 一、初始化与核心参数（`__init__` 方法）

```python
class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, reward_funcs=[reward_len], reward_weights=None, beta=0.02, num_generations=4):
        self.model = model                   # 训练中的策略模型 πθ
        self.ref_model = ref_model           # 参考模型 π_ref（冻结，用于计算KL散度）
        self.tok = tokenizer                 # 分词器
        self.reward_funcs = reward_funcs     # 奖励函数列表（默认含长度奖励reward_len）
        self.reward_weights = torch.tensor(reward_weights or [1.0]*len(reward_funcs), dtype=torch.float)           # 奖励函数权重
        self.beta = beta                     # KL惩罚系数（控制模型更新幅度）
        self.num_generations = num_generations  # 每个prompt生成的回答数量（G）
```

**核心解析**：

- `reward_funcs`：默认包含 `reward_len`（长度奖励函数），可自定义添加其他函数（如质量奖励 `reward_quality`）。
- `reward_weights`：若传入多个奖励函数，通过权重控制其影响（默认均等权重）。
- `num_generations`：每个提示词生成 G 个回答，用于后续组内比较（如 G=4 表示每个 prompt 生成 4 个回答）。

#### 二、数据准备（`_prepare_inputs` 方法）

该方法将原始数据（prompt）转换为模型可训练的输入格式，核心输出为 `inputs` 字典。

##### 1. 生成回答（completions）

```python
prompts = [x["prompt"] for x in batch]  # 提取所有prompt
completions = []
for p in prompts:
    gens = [p + " ...dummy completion..." for _ in range(self.num_generations)]  # 每个prompt生成G个回答
    completions.extend(gens)  # 总长度为 B×G（B为prompt数量，G为每个prompt的回答数）
```

**解析**：

- 假设 `B=2`（2 个 prompt）、`G=4`（每个生成 4 个回答），则 `completions` 长度为 `2×4=8`。

##### 2. 计算奖励（rewards）

```python
with torch.no_grad():  # 禁用梯度计算（奖励不参与模型更新）
    rewards_per_func = []
    for f in self.reward_funcs:  # 遍历每个奖励函数（默认仅reward_len）
        # 计算每个回答的奖励，转换为张量
        rewards_per_func.append(torch.tensor([f([c])[0] for c in completions], dtype=torch.float))
    rewards_per_func = torch.stack(rewards_per_func, dim=1)  # 形状 [B×G, R]，R为奖励函数数量（这里R=1）
    # 加权求和得到总奖励（R=1时直接等于rewards_per_func）
    rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).sum(dim=1)     # [B×G]
```

**解析**：

- 以 `reward_len` 为例（奖励公式：`-abs(20 - len(c))`），8 个回答的奖励可能为 `[-2, -2, -5, 0, -5, -1, -1, -3]`。
- `torch.stack(..., dim=1)`：将多个奖励函数的结果堆叠（R=1 时形状为 `[8, 1]`）。
- 加权求和：因 `R=1` 且权重为 1.0，`rewards` 最终为 `[-2, -2, -5, 0, -5, -1, -1, -3]`（形状 `[8]`）。

##### 3. 组内归一化计算优势值（advantages）

```python
B = len(prompts)  # prompt数量（B=2）
G = self.num_generations  # 每个prompt的回答数（G=4）
rewards = rewards.view(B, G)  # 重塑为 [B, G]（如 [2, 4]）
mean_grouped = rewards.mean(dim=1, keepdim=True)  # 每组均值 [B, 1]（如 [[-2.25], [-2.5]]）
std_grouped = rewards.std(dim=1, keepdim=True)    # 每组标准差 [B, 1]（如 [[1.785], [1.658]]）
advantages = (rewards - mean_grouped) / (std_grouped + 1e-4)  # 组内归一化 [B, G]
advantages = advantages.view(-1)  # 重塑为 [B×G]（如 [0.14, 0.14, -1.54, 1.26, -1.51, 0.90, 0.90, -0.30]）
```

**解析**：

- 组内归一化：将每个回答的奖励转换为「相对同组平均的优势」（均值 0，标准差 1），消除不同 prompt 间的奖励分布差异。
- 优势值为正：该回答优于同组平均；为负：劣于同组平均。

##### 4. 占位张量（参考模型概率与输入标识）

```python
# 参考模型的逐token对数概率（占位，真实场景需用ref_model计算）
ref_per_token_logps = torch.zeros((len(completions), 16))  # [B×G, T]，T为回答token长度（如16）

# prompt和回答的token id（占位，真实场景需用tokenizer编码）
prompt_ids = torch.zeros((len(completions), 8), dtype=torch.long)  # [B×G, P]，P为prompt token长度（如8）
completion_ids = torch.zeros((len(completions), 16), dtype=torch.long)  # [B×G, T]

# 掩码（过滤填充token）
prompt_mask = (prompt_ids != 0).to(torch.float)  # [B×G, P]，1.0=有效token，0.0=填充
completion_mask = torch.ones_like(completion_ids, dtype=torch.float)  # [B×G, T]，简化为全1（假设无填充）
```

**解析**：

- `ref_per_token_logps`：参考模型对每个回答 token 的对数概率，用于计算 KL 散度（衡量与当前模型的差异）。
- `prompt_mask`/`completion_mask`：掩码用于忽略填充 token 的损失（填充 token 不参与模型学习）。

##### 5. 返回输入字典

```python
return {
    "prompt_ids": prompt_ids,
    "prompt_mask": prompt_mask,
    "completion_ids": completion_ids,
    "completion_mask": completion_mask,
    "ref_per_token_logps": ref_per_token_logps,
    "advantages": advantages
}
```

**解析**：打包所有训练所需数据，传递给 `compute_loss` 方法。

#### 三、损失计算（`compute_loss` 方法）

核心逻辑：结合优势值和 KL 惩罚，计算模型的强化学习损失。

##### 1. 获取当前模型的逐 token 对数概率

```python
per_token_logps = self._get_per_token_logps(
    model, inputs["prompt_ids"], inputs["completion_ids"], inputs["prompt_mask"], inputs["completion_mask"]
)  # [B×G, T]，当前模型对每个回答token的对数概率
```

**解析**：

- 真实场景中，通过模型计算回答 token 的对数概率（`log_softmax` 处理 logits），此处用全 0 占位。

##### 2. 计算 KL 散度（稳定训练）

```python
ref_per_token_logps = inputs["ref_per_token_logps"]  # [B×G, T]，参考模型的对数概率
delta = ref_per_token_logps - per_token_logps  # 对数概率差
per_token_kl = torch.exp(delta) - delta - 1.0  # 稳定的KL散度近似（避免数值问题）
```

**解析**：

- KL 散度衡量当前模型与参考模型的差异，`per_token_kl` 越大，差异越大。
- 公式 `exp(delta) - delta - 1` 是 KL 散度的近似，当 `delta=0` 时 KL=0，差异增大时 KL 增大。

##### 3. 计算核心损失项（引导模型优化）

```python
advantages = inputs["advantages"].unsqueeze(-1)  # [B×G, 1]，扩展维度以匹配token级张量
# 核心技巧：前向值≈advantages，反向梯度=advantages×∇per_token_logps
per_token_core = torch.exp(per_token_logps - per_token_logps.detach()) * advantages  # [B×G, T]
```

**解析**：

- `unsqueeze(-1)`：将优势值从 `[8]` 扩展为 `[8, 1]`，便于与 `[8, 16]` 的 token 级张量广播相乘。
- `detach()`：切断 `per_token_logps` 的梯度，确保优势值不参与梯度传播，仅作为权重引导模型更新。
- 效果：高优势值回答的概率被强化，低优势值回答的概率被削弱。

##### 4. 计算最终损失

```python
per_token_loss = -(per_token_core - self.beta * per_token_kl)  # 负号转为最小化目标
# 过滤填充token，计算每个样本的平均损失
loss = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp_min(1.0)
return loss.mean()  # 返回batch平均损失
```

**解析**：

- `per_token_loss`：结合核心损失和 KL 惩罚（`beta` 控制惩罚强度），负号将「最大化目标」转为「最小化损失」。
- 掩码过滤：`per_token_loss * completion_mask` 忽略填充 token 的损失。
- 样本平均：按有效 token 数求平均，避免长度差异影响损失。
- 最终返回整个 batch 的平均损失，用于反向传播更新模型。

#### 四、辅助方法（`_get_per_token_logps`）

```python
def _get_per_token_logps(self, model, prompt_ids, completion_ids, prompt_mask, completion_mask):
    # 实际逻辑：拼接prompt+completion→logits→log_softmax→提取回答部分的logp
    return torch.zeros_like(completion_ids, dtype=torch.float)  # 占位，返回全0张量
```

**解析**：占位方法，演示计算当前模型对数概率的流程，真实场景需实现完整逻辑。

### 核心逻辑总结

GRPO 训练的核心是通过「优势值引导模型优化」和「KL 散度稳定训练」：

1. 对每个 prompt 生成多个回答，计算奖励并归一化得到优势值；
2. 对比当前模型与参考模型的差异（KL 散度），控制更新幅度；
3. 结合优势值和 KL 惩罚计算损失，让模型倾向于生成高优势值的回答，同时保持与参考模型的稳定性。

通过上述流程，模型能在强化学习中逐步优化生成质量。