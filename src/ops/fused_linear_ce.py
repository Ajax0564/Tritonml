import torch
import triton
import triton.language as tl


@triton.jit
def linear_softmax_ce_fwd(
    X, W, B, T,
    LOSS,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_b,
    stride_t,
    stride_l,
    K: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IGNORE_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)

    target = tl.load(T + pid * stride_t)

    if target == IGNORE_INDEX:
        tl.store(LOSS + pid * stride_l, 0.0)
        return

    offs_k = tl.arange(0, K)
    x = tl.load(X + pid * stride_xm + offs_k * stride_xk)

    max_logit = -float("inf")

    # ---- Pass 1: max ----
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N

        w = tl.load(
            W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask[None, :],
            other=0.0,
        )
        logits = tl.sum(x[:, None] * w, axis=0)

        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)

        max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))

    denom = 0.0
    target_logit = -float("inf")  # scalar fp32

    # ---- Pass 2: sum + target ----
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask = offs_n < N

        w = tl.load(
            W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask[None, :],
            other=0.0,
        )
        logits = tl.sum(x[:, None] * w, axis=0)

        if B is not None:
            logits += tl.load(B + offs_n * stride_b, mask=mask, other=0.0)

        exp_logits = tl.exp(logits - max_logit)
        denom += tl.sum(exp_logits, axis=0)

        # extract target logit safely
        is_target = offs_n == target
        tgt = tl.where(is_target, logits, -float("inf"))
        target_logit = tl.maximum(target_logit, tl.max(tgt, axis=0))

    log_prob = target_logit - max_logit - tl.log(denom + 1e-12)
    loss = -log_prob

    tl.store(LOSS + pid * stride_l, loss)


def triton_linear_cross_entropy(
    inputs, weight, bias, targets, ignore_index=-100
):
    M, K = inputs.shape
    N = weight.shape[0]

    loss_buf = torch.empty((M,), device=inputs.device, dtype=torch.float32)

    grid = (M,)

    linear_softmax_ce_fwd[grid](
        inputs, weight, bias, targets,
        loss_buf,
        inputs.stride(0), inputs.stride(1),
        weight.stride(1), weight.stride(0),
        bias.stride(0),
        targets.stride(0),
        loss_buf.stride(0),
        K=K,
        N=N,
        BLOCK_N=256,
        IGNORE_INDEX=ignore_index,
        num_warps=4,
        num_stages=2,
    )

    valid_mask = targets != ignore_index
    return loss_buf[valid_mask].mean()

