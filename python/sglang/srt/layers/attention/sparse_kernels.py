import torch
import triton
import triton.language as tl
from functools import lru_cache

@triton.jit
def compress_k_complete_kernel(
    key_ptr,
    cu_seqlens_ptr,
    compressed_k_ptr,
    cu_seqlens_compressed_ptr,
    batch_size,
    max_chunks_per_seq,
    head_num_k: tl.constexpr,
    head_dim: tl.constexpr,
    kernel_size: tl.constexpr,
    kernel_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Complete single-kernel implementation combining chunk calculation and key compression.
    
    This kernel performs all computations entirely within the kernel:
    1. Computes sequence lengths and chunk counts
    2. Computes prefix sums for output positioning
    3. Validates chunk boundaries
    4. Performs mean pooling compression
    5. Updates cumulative sequence lengths
    
    Grid: (batch_size, max_chunks_per_seq, head_num_k)
    Each thread processes one (batch, chunk_in_seq, head) combination.
    
    Args:
        key_ptr: Input key tensor [total_seq_len, head_num_k, head_dim]
        cu_seqlens_ptr: Input cumulative sequence lengths [batch_size + 1]
        compressed_k_ptr: Output compressed key tensor [batch_size * max_chunks_per_seq, head_num_k, head_dim]
        cu_seqlens_compressed_ptr: Output cumulative chunk counts [batch_size + 1]
        batch_size: Number of sequences in batch
        max_chunks_per_seq: Maximum possible chunks per sequence (conservative)
        head_num_k: Number of attention heads
        head_dim: Dimension per head
        kernel_size: Tokens per chunk for compression
        kernel_stride: Stride between chunk starts
        BLOCK_SIZE: Vectorized load/store width
    """
    batch_idx = tl.program_id(0)
    chunk_in_seq = tl.program_id(1)
    head_idx = tl.program_id(2)
    
    # ====================================================================
    # PHASE 1: Get sequence info and validate chunk
    # ====================================================================
    
    if batch_idx >= batch_size or head_idx >= head_num_k:
        return
    
    seq_start = tl.load(cu_seqlens_ptr + batch_idx)
    seq_end = tl.load(cu_seqlens_ptr + batch_idx + 1)
    seq_len = seq_end - seq_start
    
    # Compute how many chunks this sequence actually has
    actual_chunks_in_seq = tl.maximum((seq_len - kernel_size) // kernel_stride + 1, 0)
    
    # Skip if this chunk doesn't exist for this sequence
    if chunk_in_seq >= actual_chunks_in_seq:
        return
    
    chunk_start_pos = seq_start + chunk_in_seq * kernel_stride
    chunk_end_pos = chunk_start_pos + kernel_size
    
    # Additional validation to match original implementation:
    # Only include chunks that are fully contained within the sequence
    if chunk_end_pos > seq_end:
        return
    
    # ====================================================================
    # PHASE 2: Compute prefix sums and global chunk index
    # ====================================================================
    
    # Compute prefix sum of chunks up to this sequence (replicated in each block)
    chunks_before_this_seq = 0
    for s in range(batch_idx):
        s_start = tl.load(cu_seqlens_ptr + s)
        s_end = tl.load(cu_seqlens_ptr + s + 1)
        s_len = s_end - s_start
        s_chunks = tl.maximum((s_len - kernel_size) // kernel_stride + 1, 0)
        chunks_before_this_seq += s_chunks
    
    global_chunk_idx = chunks_before_this_seq + chunk_in_seq
    
    # Update cu_seqlens_compressed (only first thread per sequence)
    if chunk_in_seq == 0 and head_idx == 0:
        # Compute cumulative chunks up to this sequence (inclusive)
        chunks_up_to_this_seq = chunks_before_this_seq + actual_chunks_in_seq
        tl.store(cu_seqlens_compressed_ptr + batch_idx + 1, tl.cast(chunks_up_to_this_seq, tl.int32))
    
    # ====================================================================
    # PHASE 3: Perform mean pooling compression
    # ====================================================================
    
    # Accumulate over all tokens in this chunk
    acc = tl.zeros([head_dim], dtype=tl.float32)
    
    for token_offset in range(kernel_size):
        token_pos = chunk_start_pos + token_offset
        
        # Compute key memory offset: token_pos * head_num_k * head_dim + head_idx * head_dim
        key_base_offset = token_pos * head_num_k * head_dim + head_idx * head_dim
        
        # Vectorized load of head_dim values
        x = tl.load(
            key_ptr + key_base_offset + tl.arange(0, BLOCK_SIZE),
            mask=tl.arange(0, BLOCK_SIZE) < head_dim,
            other=0.0
        ).to(tl.float32)
        
        acc += x
    
    # Compute mean over the chunk
    acc = acc / kernel_size
    
    # ====================================================================
    # PHASE 4: Store compressed result
    # ====================================================================
    
    # Store compressed key at global position: global_chunk_idx * head_num_k * head_dim + head_idx * head_dim
    out_offset = global_chunk_idx * head_num_k * head_dim + head_idx * head_dim
    tl.store(
        compressed_k_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
        acc,
        mask=tl.arange(0, BLOCK_SIZE) < head_dim
    )

