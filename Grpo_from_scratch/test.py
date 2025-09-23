import torch
import torch.nn

def reward_len(completions,**kw):
    return(-abs(20-len(c)) for c in completions)


class GRPOTrainer:
    def __init__(self,model,ref_model,tokenizer,reward_funcs=[reward_len],reward_weights=None,beta = 0.2,num_generations= 4):
        self.model = model
        self.ref_model = ref_model
        self.tok = tokenizer
        self.reward_funcs = reward_funcs
        self.weights = torch.tensor(reward_weights or [1.0]*len(reward_funcs),dtype = torch.float)
        self.beta = beta
        self.num_generations = num_generations
    
    def __prepare_inputs(self,batch):
        prompts = [x['prompt'] for x in batch]
        completions = []
        for p in prompts:
            gens = [p+ "...dummy completion..." for _ in range(self.num_generations)]
            completions.extend(gens)

        with torch.no_grad():
            rewards_per_func = []
            for f in self.reward_funcs:
                rewards_per_func.append(torch.tensor([f(c)[0] for c in completions],dtype = torch.float))
            reward_per_func = torch.stack(reward_per_func,dim = 1)
            rewards =(reward_per_func * self.reward_weights.to(reward_per_func.devic).unsqueeze(0)).sum(dim=1)

            B = len(prompts)
            G = self.num_generations
            rewards = rewards.view(B,G)  # [B,G]
            mean_grouped = rewards.mean(dim = 1,keepdim = True) # [B,1]
            std_grouped = rewards.std(dim=1,keepdim = True) . #[B,1]
            advantages = (rewards - mean_grouped)/(std_grouped + 1e4) #[B,G]
            advantages = advantages.vies(-1) #[B*G]

        ref_per_token_logps = torch.zeros((len(completions),16))
        prompt_ids = torch.zeros((len(prompts),8),dtype = torch.long)
        completions_ids = torch.zeros((len(completions),16),dtype = torch.long)
        prompt_mask = (prompts_ids !=0).to(torch.float)
        completions_mask = torch.ones_like(completion_ids,dtype = torch.float)

        return {
            "prompt_ids":prompt_ids,
            "prompt_mask":prompt_mask,
            "completions_ids":completions_ids,
            "completions_mask":completions_mask,
            "ref_per_token_logps":ref_per_token_logps,
            "advantages":advantages,
        }
    def compute_loss(self,model,inputs):
        per_token_logps = self._get_per_token_logps(
            model,
            inputs["prompt_ids"],
            inputs["completions_ids"],
            inputs["prompt_mask"],
            inputs["completions_mask"],
        )
        ref_per_token_logps = inputs["ref_per_token_logps"]
        delta = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(delta) - delta -1.0
        advantages = inputs["advantages"].unsqueeze(-1)
        per_token_core = torch.exp(per_token_logps - per_token_logps.detach()) * advantages

        completions_mask = inputs["completions_mask"]
        per_token_loss =(-per_token_core - self.beta* per_token_kl)

        loss= (per_token_loss * completions_mask).sum(dim = 1) /completion_mask.sum(dim = 1).clanp_min(1.0)
        return loss.mean()

    def _get_per_token_logps(self,model,prompt_ids,completions_ids,prompt_mask,completions_mask):
        return torch.zeros_like(completions_ids,dtype = torch.float)



