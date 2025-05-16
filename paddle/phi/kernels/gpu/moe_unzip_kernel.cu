// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/moe_unzip_kernel.h"
#include "paddle/phi/kernels/funcs/moe_utils.h"

namespace phi {
    
#define CUMSUM_BLOCK_SIZE 48   // cumsum开销和并行度之间的tradeoff的结果，勿动
#define CUMSUM_INVALID_TAG -1  // 用于标记无效的cumsum，尝试过-114514但失败了
#ifndef MAX_NUM_EXPERTS
#define MAX_NUM_EXPERTS 32
#endif
// 多阶段算法，控制每block处理的行数来权衡额外开销
//  首先解析routemap来更新专家当前所收到的token数，然后check前一个block给的前缀和并更新给下一个block
//  随后，目的行号的信息已获取，立即开始搬运工作，直至任务完全完成
template <typename X_T, typename routemap_T, typename probs_T, bool has_scale>
__global__ void tokens_unzip_stable_kernel(
    const X_T *__restrict__ X,
    const routemap_T *__restrict__ routemap_topk,
    const probs_T *__restrict__ probs_topk,
    const float *__restrict__ XScale,
    X_T *__restrict__ X_unzipped,
    int *__restrict__ zipped_expertwise_rowmap,
    probs_T *__restrict__ probs_unzipped,
    float *__restrict__ XScale_unzipped,
    int *global_expertwise_block_cumsum,
    const int total_zipped_tokens_num,
    const int max_tokens_per_expert,
    const int token_length,
    const int scale_length,
    const int num_experts,
    const int topk) {
  const int block_row_base = blockIdx.x * CUMSUM_BLOCK_SIZE;
  int cumsum_offset = (blockIdx.x != 0)* CUMSUM_INVALID_TAG;  // 除了第0个block，其他的都以非法值初始化,因为atomic忙等要用
  int expert_offset = threadIdx.x % num_experts * max_tokens_per_expert;
  int local_cumsum = 0;
  const int base_row_idx = blockIdx.x * CUMSUM_BLOCK_SIZE;
  __shared__ int shared_expert_rowmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS];
  __shared__ probs_T shared_expert_probmap[CUMSUM_BLOCK_SIZE][MAX_NUM_EXPERTS];

  // --------------------- num_experts个线程处理不同的experts 要保证blockDim.x>=nums_experts -------------------------
  if (threadIdx.x < num_experts) [[unlikely]] {
    int local_expert_rowmap[CUMSUM_BLOCK_SIZE];
    probs_T local_expert_probs[CUMSUM_BLOCK_SIZE];
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
      local_expert_rowmap[i] = -1;  // 以非法值初始化，方便后续shared mem写入
      local_expert_probs[i] = (probs_T)0;
    }
    // 将乱序访存限制在寄存器级别，后续shared_mem规整写入
    for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
         row++) {
      if (row >= total_zipped_tokens_num) break;
      const int internal_row = row - block_row_base;
#pragma unroll
      for (int k = 0; k < topk; k++) {
        const int expert = routemap_topk[row * topk + k];
        if (expert == -1) continue;
        if (threadIdx.x == expert) {
          local_expert_rowmap[internal_row] = local_cumsum + expert_offset;
          local_expert_probs[internal_row] = probs_topk[row * topk + k];
          local_cumsum += 1;
        }
      }
    }
// -------------------------- 块间通信逻辑 -----------------------------
    if (blockIdx.x != 0) [[likely]] {
      while (cumsum_offset == CUMSUM_INVALID_TAG) [[likely]] {
        cumsum_offset = atomicExch(
            &global_expertwise_block_cumsum[blockIdx.x * num_experts + threadIdx.x],
            CUMSUM_INVALID_TAG);
      }
    }
    const int proposed_offset = cumsum_offset + local_cumsum;
    global_expertwise_block_cumsum[(blockIdx.x + 1) * num_experts + threadIdx.x] =
        proposed_offset;
    // 至此，给下一个block的cumsum已经更新完毕，下一个block可以开始cumsum的计算了

// -------------------------- 块内通信逻辑 -----------------------------
#pragma unroll
    for (int i = 0; i < CUMSUM_BLOCK_SIZE; i++) {
      const int proposed_row =
          (local_expert_rowmap[i] == -1)
              ? -1
              : (local_expert_rowmap[i] + cumsum_offset);
      shared_expert_rowmap[i][threadIdx.x] = proposed_row;
      shared_expert_probmap[i][threadIdx.x] = local_expert_probs[i];
    }
  }  // 至此，本线程块内的shared_mem已经规整完毕，接下来是向量化的数据搬运
  __syncthreads();  // 其余线程等到了thread0，工作安排在shared_mem上
  // ------------------------- 所有block内线程 -------------------------
  for (int row = block_row_base; row < block_row_base + CUMSUM_BLOCK_SIZE;
       row++) {
    if (row >= total_zipped_tokens_num) return;
    const int internal_row = row - block_row_base;
#pragma unroll
    for (int expert = 0; expert < num_experts; expert++) {
      const int unzipped_row_idx = shared_expert_rowmap[internal_row][expert];
      if (threadIdx.x == 0) {
        zipped_expertwise_rowmap[row * num_experts + expert] = unzipped_row_idx;
      }
      if (unzipped_row_idx == -1) continue;
      // 更新三个核心数据结构
      if (threadIdx.x == 0) {
        probs_unzipped[unzipped_row_idx] =
            shared_expert_probmap[internal_row][expert];
      }
      if constexpr (has_scale) {
        phi::funcs::vectorized_memcpy(&XScale[row * scale_length],
                          &XScale_unzipped[unzipped_row_idx * scale_length],
                          scale_length);
      }
      phi::funcs::vectorized_memcpy(&X[row * token_length],
                        &X_unzipped[unzipped_row_idx * token_length],
                        token_length);
    }
  }
}

template <typename T, typename Context>
void MoeUnzipKernel(const Context& dev_ctx,
                    const DenseTensor& X,
                    const DenseTensor& XScale,
                    const DenseTensor& expert_routemap_topk,
                    const DenseTensor& expert_prob_topk,
                    const Scalar& max_tokens_per_expert,
                    int topk,
                    int num_experts,
                    DenseTensor* X_unzipped,
                    DenseTensor* zipped_expertwise_rowmap,
                    DenseTensor* token_prob_unzipped,
                    DenseTensor* XScale_unzipped,
                    DenseTensor* global_expertwise_block_cumsum) {
  auto X_dims = X.dims();
  auto XScale_dims = XScale.dims();
  int seq_len = X_dims[0];
  int token_length = X_dims[1];
  dev_ctx.template Alloc<T>(X_unzipped);
  dev_ctx.template Alloc<float>(XScale_unzipped);
  dev_ctx.template Alloc<int>(global_expertwise_block_cumsum);
  auto stream = dev_ctx.stream();
  //为global_expertwise_block_cumsum赋值：
  auto global_expertwise_block_cumsum_ptr =
      reinterpret_cast<void *>(global_expertwise_block_cumsum->data<int>());
  const int cumsum_blocknum = (seq_len + CUMSUM_BLOCK_SIZE - 1) / CUMSUM_BLOCK_SIZE;
  cudaMemsetAsync(global_expertwise_block_cumsum_ptr,
                CUMSUM_INVALID_TAG,
                sizeof(int) * (cumsum_blocknum + 1) * num_experts,
                stream); // cuda流的设置

  int scale_length = XScale_dims.size() > 1 ? XScale_dims[1] : 0;
  auto max_tokens_per_expert_v = max_tokens_per_expert.to<int>();

  dim3 grid, block;
  grid.x = cumsum_blocknum;
  block.x = 256;
  // 定义类型获取宏
  #define DTYPE_CASE(dtype, type) dtype == phi::DataType::type
  #define GET_DATA(tensor, type) tensor.data<type>()
  #define GET_PTR_DATA(tensor, type) tensor->data<type>()
// 分发处理不同的类型组合
#define DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)                       \
  auto kernel = tokens_unzip_stable_kernel<TOKEN_T, INT_T, PROB_T, HAS_SCALE>; \
  kernel<<<grid, block, 0, stream>>>(                                       \
      GET_DATA(X, TOKEN_T),                                                    \
      GET_DATA(expert_routemap_topk, INT_T),                                   \
      GET_DATA(expert_prob_topk, PROB_T),                                      \
      XScale.data<float>(),                                                    \
      GET_PTR_DATA(X_unzipped, TOKEN_T),                                           \
      GET_PTR_DATA(zipped_expertwise_rowmap, INT_T),                               \
      GET_PTR_DATA(token_prob_unzipped, PROB_T),                                   \
      XScale_unzipped->data<float>(),                                           \
      global_expertwise_block_cumsum->data<int>(),                              \
      seq_len,                                                 \
      max_tokens_per_expert_v,                                                   \
      token_length,                                                            \
      scale_length,                                                            \
      num_experts,                                                             \
      topk);

// 可扩展：处理特定的topk和num_experts组合,可根据之后需求进行扩展
#define HANDLE_EXPERT_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE) \
  DISPATCH_CASE(TOKEN_T, PROB_T, INT_T, HAS_SCALE)

#define HANDLE_TOKEN_TYPE(PROB_T, INT_T)                       \
  if (DTYPE_CASE(X.dtype(), BFLOAT16)) {                       \
    HANDLE_EXPERT_CASE(T, PROB_T, INT_T, false)                   \
  } else if (DTYPE_CASE(X.dtype(), FLOAT8_E4M3FN)) {           \
    HANDLE_EXPERT_CASE(T, PROB_T, INT_T, true)                    \                   
  }

#define HANDLE_PROB_TYPE(INT_T)                                \
  if (DTYPE_CASE(expert_prob_topk.dtype(), BFLOAT16)) {        \
    dev_ctx.template Alloc<phi::bfloat16>(token_prob_unzipped);\
    HANDLE_TOKEN_TYPE(phi::bfloat16, INT_T)                    \
  } else if (DTYPE_CASE(expert_prob_topk.dtype(), FLOAT32)) {  \
    dev_ctx.template Alloc<float>(token_prob_unzipped);        \
    HANDLE_TOKEN_TYPE(float, INT_T)                            \
  }

  // 可扩展：根据整型类型控制派发，未来可支持int8，但int64不行，因为下标开销太重了，建议在外面直接cast到int32
  if (DTYPE_CASE(zipped_expertwise_rowmap->dtype(), INT32)) {
    dev_ctx.template Alloc<int>(zipped_expertwise_rowmap);
    HANDLE_PROB_TYPE(int)
  }
  #undef DTYPE_CASE
  #undef GET_DATA
  #undef DISPATCH_CASE
  #undef HANDLE_EXPERT_CASE
  #undef HANDLE_TOKEN_TYPE
  #undef HANDLE_PROB_TYPE
}

}  // namespace phi

PD_REGISTER_KERNEL(moe_unzip,
                   GPU,
                   ALL_LAYOUT,
                   phi::MoeUnzipKernel,
                   int,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float8_e4m3fn) {}
