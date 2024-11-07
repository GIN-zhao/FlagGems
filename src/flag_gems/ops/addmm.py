import torch

import triton
import triton.language as tl

def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128}, num_warps=4),
    ]
def get_autotune_config():
    return get_cuda_autotune_config()


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit(do_not_specialize=["alpha","beta"])
def triton_addmm(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    alpha,
    beta,
    am_stride,
    ak_stride,
    bk_stride,
    bn_stride,
    cm_stride,
    cn_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 
    
    offset_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    

    a_ptrs = a_ptr + offset_m[:, None] * am_stride + offset_k[None, :] * ak_stride
    b_ptrs = b_ptr + offset_k[:, None] * bk_stride + offset_n[None, :] * bn_stride
    bias_ptrs = bias_ptr + offset_n   
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) 
    
    for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,mask=offset_m[:,None]<M & offset_n[None,:]<K-k*BLOCK_SIZE_K , other=0.)
        b = tl.load(b_ptrs,mask=offset_m[:,None]<K-k*BLOCK_SIZE_K & offset_n[None,:]<N , other=0.) 
        
        accumulator += tl.dot(a,b)
        a_ptrs += am_stride*BLOCK_SIZE_M 
        b_ptrs += bk_stride*BLOCK_SIZE_N 
    bias = tl.load(bias_ptrs,mask=offset_n<N , other=0.)
    accumulator = accumulator*alpha + bias*beta 
    
    c_ptrs = c_ptr + offset_m[:, None] * cm_stride + offset_n[None, :] * cn_stride
    tl.store(c_ptrs, accumulator, mask=offset_m[:,None]<M & offset_n[None,:]<N)
    
def addmm(bias, mat1, mat2, alpha=1.0, beta=1.0):
    assert mat1.shape[1] == mat2.shape[0],"mat1.shape[1] != mat2.shape[0]"
    
    M = mat1.shape[0]
    K = mat2.shape[1]
    N = mat2.shape[0] 
    
    mat3 = torch.empty((M, N), dtype=mat1.dtype, device=mat1.device)
    
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))
    triton_addmm[grid](
        mat1,
        mat2,
        mat3,
        bias,
        M,
        N,
        K,
        alpha,
        beta,
        mat1.stride(0),
        mat1.stride(1),
        mat2.stride(0),
        mat2.stride(1),
        mat1.stride(0),
        mat1.stride(1))
    return mat3