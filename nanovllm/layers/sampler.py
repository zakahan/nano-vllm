import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # 采样
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        logits.div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)
        """
        这里很有意思哈，
        按照常规思路
        一般是，if xx 计算A else 计算B
        但这里却相反————
        先计算A并计算B，再“判断”我要谁
        
        这里涉及到了GPU并行计算的特点和PyTorch执行的机制
        GPU擅长计算，而非选择控制逻辑，所以这么干

        这里还有一个点可以将那就是`torch.where`，看上去最后还是要有控制逻辑？
        并非如此，torch.where其实还是计算逻辑，这是一种并行执行的向量化计算（无分支跳转）
        所以他只是长得像控制逻辑
        """  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
