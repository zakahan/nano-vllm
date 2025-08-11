import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,        # key的内存位置（临时的）(这个位置是在GPU全局内存中)
    key_stride,     # key向量之间的内存间隔（步长）
    value_ptr,
    value_stride,   
    k_cache_ptr,    # 指向kv缓存区的内存地址
    v_cache_ptr,
    slot_mapping_ptr,   # kvcache对应槽位（这是一个特殊的区域，通过slot管理，更高效）
    D: tl.constexpr,    # 常量维度D，表示单个kv向量的维度 num_heads * head_dim(用来提升gpu执行效率）
):
    """
    ### 本函数作用
    读取 -> 映射 -> 存储
    用于将key/value高效的存储到kv缓存中，
    将kv向量规整的存储到kv-cache里，而具体的坐标就是slot_mapping_ptr

    ### 关于线程块
    在GPU编程中，特别是使用CUDA或Triton这样的并行计算框架时，
    计算任务通常被组织成一个由线程块（Thread Block）构成的网格（Grid）结构来执行。
    每个线程块包含多个线程（Thread），这些线程共同协作完成一部分计算任务
    """
    # 获取坐标
    idx = tl.program_id(0)      # 获取当前GPU线程块的索引
    key_offsets = idx * key_stride + tl.arange(0, D)        # 计算偏移量
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 读kv向量
    key = tl.load(key_ptr + key_offsets)        # 加载kv向量（这一步就相当于从gpu中读取向量吗？这是一个读处理
    value = tl.load(value_ptr + value_offsets)  # tl.load() 从这些地址中将数据加载到当前线程块的局部存储（GPU寄存器或者是GPU L1/L2缓存）

    slot = tl.load(slot_mapping_ptr + idx)      # 获取当前 token 应该存储到的缓存槽位索引。
    cache_offsets = slot * D + tl.arange(0, D)   
    tl.store(k_cache_ptr + cache_offsets, key)      # 将kv cache存储到slot所在位置
    tl.store(v_cache_ptr + cache_offsets, value)    


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 并行启动N个线程块（Block） Triton 提供的特殊语法（装饰器） 
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
