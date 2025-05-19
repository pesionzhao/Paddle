import paddle
import paddle.nn.functional as F
from paddle import _C_ops
from paddle.base.framework import in_dynamic_or_pir_mode
from paddle.base.layer_helper import LayerHelper
from paddle import Tensor
import numpy as np
def moe_unzip(
    X: Tensor,
    XScale: Tensor,
    expert_routemap_topk: Tensor,
    expert_prob_topk: Tensor,
    max_tokens_per_expert: int,
    topk: int,
    num_experts: int,
    name: str | None = None,
):
    # 为了突出重点，省略部分代码
    # 动静统一分支，直接调用算子对应的 Python C 函数
    if in_dynamic_or_pir_mode():
        X_unzipped, zipped_experwise, token_prob_unzipped, XScale_unzipped, _ = \
        _C_ops._moe_unzip(X, XScale, expert_routemap_topk, expert_prob_topk, max_tokens_per_expert, topk, num_experts)
        return (X_unzipped, zipped_experwise, token_prob_unzipped, XScale_unzipped)

    # 老静态图分支
    ## 输入参数检查
    # __check_input

    ## 构造输出，添加 op，返回输出
    helper = LayerHelper('_moe_unzip', **locals())
    X_unzipped = helper.create_variable_for_type_inference(dtype=X.dtype)
    zipped_experwise = helper.create_variable_for_type_inference(dtype=expert_routemap_topk.dtype)
    token_prob_unzipped = helper.create_variable_for_type_inference(dtype=expert_prob_topk.dtype)
    XScale_unzipped = helper.create_variable_for_type_inference(dtype=XScale.dtype)
    global_expertwise_block_cumsum = helper.create_variable_for_type_inference(dtype=expert_routemap_topk.dtype)

    inputs = {
        'X': X, 
        'XScale': XScale, 
        'expert_routemap_topk': expert_routemap_topk, 
        'expert_prob_topk': expert_prob_topk, 
    }

    outputs = {
        'X_unzipped': X_unzipped, 
        'zipped_experwise': zipped_experwise, 
        'token_prob_unzipped': token_prob_unzipped, 
        'XScale_unzipped': XScale_unzipped,
        'global_expertwise_block_cumsum': global_expertwise_block_cumsum,
    }

    helper.append_op(
        type='_moe_unzip',
        inputs=inputs,
        attrs={'topk': topk, 'num_experts': num_experts,'max_tokens_per_expert': max_tokens_per_expert},
        outputs=outputs,
    )
    return (X_unzipped, zipped_experwise, token_prob_unzipped, XScale_unzipped)

if __name__ == "__main__":
    H1 = 7168
    H2 = 2048
    topk = 8

    paddle.seed(42)
    topk_ind = np.load("/work/PaddleNLP/tests/ops/topk_indice.npy")
    reci_x = paddle.randn( [ topk_ind.shape[0], H1], dtype="bfloat16")
    reci_x_fp8 = reci_x.cast("float8_e4m3fn")
    print("reci_x_fp8 dtype: ", reci_x_fp8.dtype)
    reci_x_scale = paddle.randn((reci_x.shape[0], int((H1 + 127) / 128)), dtype="float32")
    print("reci_x scale shape: ", reci_x_scale.shape)
    topk_ind_base = paddle.to_tensor(topk_ind, dtype="int32")
    probs = paddle.ones( topk_ind_base.shape, dtype="bfloat16") # uses ones as topk_ind_base???
    print( topk_ind.shape)
    print( "recv x", reci_x.shape)

    total_num = int((topk_ind != -1).astype("int64").sum())

    e0_num = int( (topk_ind == 0).astype("int64").sum() )
    e1_num = int((topk_ind == 1).astype("int64").sum())
    e2_num = int( (topk_ind == 2).astype("int64").sum())
    e3_num = int( (topk_ind == 3).astype("int64").sum())
    token_per_expert = [  e0_num, e1_num, e2_num, e3_num ]
    print("token per expert: ", token_per_expert)
    max_tokens = max(token_per_expert)
    print("############## FP8 ################")

    (unzipped_tokens, zipped_expertwise_rowmap, unzipped_probs, unzipped_scales) = moe_unzip(
        reci_x_fp8,
        reci_x_scale,
        topk_ind_base, 
        probs,
        max_tokens,
        topk,
        4)
    
    print("zipped_expertwise_rowmap_fp8: ", zipped_expertwise_rowmap)
    np.savetxt("zipped_expertwise_rowmap_fp8_zps.csv", zipped_expertwise_rowmap, delimiter=",", fmt='%d')
    # np.savetxt("unzippedtokens_fp8_zps.csv", unzipped_tokens, delimiter=",", fmt='%d')
    np.savetxt("topk_ind_fp8.csv", topk_ind, delimiter=",", fmt='%d')
    np.savetxt("unzipped_scales_fp8_zps.csv", unzipped_scales[:10, :], delimiter=",", fmt='%d')


