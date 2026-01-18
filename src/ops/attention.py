import torch
import triton
import triton.language as tl
import math

# ============================================================
# Forward kernel
# ============================================================
@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Mask, sm_scale, L, Out,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    stride_ob, stride_oh, stride_om, stride_ok,
    BATCH, HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hz = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rk = tl.arange(0, HEAD_DIM)

    q = tl.load(
        Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :],
        mask=rm[:, None] < SEQ_LEN,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    mask_ptr = Mask + (pid_hz // HEADS) * stride_mb + (pid_hz % HEADS) * stride_mh

    for start_n in range(0, SEQ_LEN, BLOCK_N):
        cols = start_n + rn

        k = tl.load(
            K + pid_hz * stride_kh + cols[None, :] * stride_kn + rk[:, None] * stride_kk,
            mask=cols[None, :] < SEQ_LEN,
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V + pid_hz * stride_vh + cols[:, None] * stride_vn + rk[None, :],
            mask=cols[:, None] < SEQ_LEN,
            other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, k) * sm_scale

        m_tile = tl.load(
            mask_ptr + rm[:, None] * stride_mm + cols[None, :] * stride_mn,
            mask=(rm[:, None] < SEQ_LEN) & (cols[None, :] < SEQ_LEN),
            other=-float("inf"),
        )

        qk += m_tile

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    out = acc / l_i[:, None]

    tl.store(
        Out + pid_hz * stride_oh + rm[:, None] * stride_om + rk[None, :],
        out.to(tl.float16),
        mask=rm[:, None] < SEQ_LEN,
    )

    tl.store(
        L + pid_hz * SEQ_LEN + rm,
        m_i + tl.log(l_i),
        mask=rm < SEQ_LEN,
    )


# ============================================================
# Backward kernel
# ============================================================
@triton.jit
def _attn_bwd_kernel(
    Q, K, V, Mask, sm_scale,
    dO, dQ, dK, dV, L,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_mb, stride_mh, stride_mm, stride_mn,
    BATCH, HEADS, SEQ_LEN,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_hz = tl.program_id(0)
    rk = tl.arange(0, HEAD_DIM)

    mask_ptr = Mask + (pid_hz // HEADS) * stride_mb + (pid_hz % HEADS) * stride_mh

    # ---------------- dQ ----------------
    for start_m in range(0, SEQ_LEN, BLOCK_M):
        rm = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(
            Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :],
            mask=rm[:, None] < SEQ_LEN,
            other=0.0,
        ).to(tl.float32)

        do = tl.load(
            dO + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :],
            mask=rm[:, None] < SEQ_LEN,
            other=0.0,
        ).to(tl.float32)

        lse = tl.load(
            L + pid_hz * SEQ_LEN + rm,
            mask=rm < SEQ_LEN,
            other=0.0,
        )

        dq = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
        Di = tl.zeros([BLOCK_M], tl.float32)

        # First pass: compute Di
        for start_n in range(0, SEQ_LEN, BLOCK_N):
            rn = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                K + pid_hz * stride_kh + rn[None, :] * stride_kn + rk[:, None] * stride_kk,
                mask=rn[None, :] < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)

            v = tl.load(
                V + pid_hz * stride_vh + rn[:, None] * stride_vn + rk[None, :],
                mask=rn[:, None] < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)

            qk = tl.dot(q, k) * sm_scale
            m_tile = tl.load(
                mask_ptr + rm[:, None] * stride_mm + rn[None, :] * stride_mn,
                mask=(rm[:, None] < SEQ_LEN) & (rn[None, :] < SEQ_LEN),
                other=-float("inf"),
            )
            qk += m_tile

            p = tl.exp(qk - lse[:, None])
            Di += tl.sum(p * tl.dot(do, tl.trans(v)), axis=1)

        # Second pass: compute dQ
        for start_n in range(0, SEQ_LEN, BLOCK_N):
            rn = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                K + pid_hz * stride_kh + rn[None, :] * stride_kn + rk[:, None] * stride_kk,
                mask=rn[None, :] < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)

            v = tl.load(
                V + pid_hz * stride_vh + rn[:, None] * stride_vn + rk[None, :],
                mask=rn[:, None] < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)

            qk = tl.dot(q, k) * sm_scale
            m_tile = tl.load(
                mask_ptr + rm[:, None] * stride_mm + rn[None, :] * stride_mn,
                mask=(rm[:, None] < SEQ_LEN) & (rn[None, :] < SEQ_LEN),
                other=-float("inf"),
            )
            qk += m_tile

            p = tl.exp(qk - lse[:, None])
            dp = (tl.dot(do, tl.trans(v)) - Di[:, None]) * p * sm_scale
            dq += tl.dot(dp, tl.trans(k))

        tl.store(
            dQ + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :],
            dq.to(tl.float16),
            mask=rm[:, None] < SEQ_LEN,
        )

        # ---------------- dK / dV ----------------
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)

        dk = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)
        dv = tl.zeros([BLOCK_N, HEAD_DIM], tl.float32)

        k = tl.load(
            K + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :],
            mask=rn[:, None] < SEQ_LEN,
            other=0.0,
        ).to(tl.float32)

        v = tl.load(
            V + pid_hz * stride_vh + rn[:, None] * stride_vn + rk[None, :],
            mask=rn[:, None] < SEQ_LEN,
            other=0.0,
        ).to(tl.float32)

        for start_m in range(0, SEQ_LEN, BLOCK_M):
            rm = start_m + tl.arange(0, BLOCK_M)

            q = tl.load(
                Q + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :],
                mask=rm[:, None] < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)

            do = tl.load(
                dO + pid_hz * stride_qh + rm[:, None] * stride_qm + rk[None, :],
                mask=rm[:, None] < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)

            lse = tl.load(
                L + pid_hz * SEQ_LEN + rm,
                mask=rm < SEQ_LEN,
                other=0.0,
            )

            # -------- PASS 1: compute Di over ALL n --------
            Di = tl.zeros([BLOCK_M], tl.float32)

            for start_n2 in range(0, SEQ_LEN, BLOCK_N):
                rn2 = start_n2 + tl.arange(0, BLOCK_N)

                k2 = tl.load(
                    K + pid_hz * stride_kh + rn2[None, :] * stride_kn + rk[:, None] * stride_kk,
                    mask=rn2[None, :] < SEQ_LEN,
                    other=0.0,
                ).to(tl.float32)

                v2 = tl.load(
                    V + pid_hz * stride_vh + rn2[:, None] * stride_vn + rk[None, :],
                    mask=rn2[:, None] < SEQ_LEN,
                    other=0.0,
                ).to(tl.float32)

                qk = tl.dot(q, k2) * sm_scale
                m_tile = tl.load(
                    mask_ptr + rm[:, None] * stride_mm + rn2[None, :] * stride_mn,
                    mask=(rm[:, None] < SEQ_LEN) & (rn2[None, :] < SEQ_LEN),
                    other=-float("inf"),
                )
                qk += m_tile

                p = tl.exp(qk - lse[:, None])
                Di += tl.sum(p * tl.dot(do, tl.trans(v2)), axis=1)

            # -------- PASS 2: accumulate dk / dv --------
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            m_tile = tl.load(
                mask_ptr + rm[:, None] * stride_mm + rn[None, :] * stride_mn,
                mask=(rm[:, None] < SEQ_LEN) & (rn[None, :] < SEQ_LEN),
                other=-float("inf"),
            )
            qk += m_tile

            p = tl.exp(qk - lse[:, None])

            dv += tl.dot(tl.trans(p), do)

            dp = (tl.dot(do, tl.trans(v)) - Di[:, None]) * p * sm_scale
            dk += tl.dot(tl.trans(dp), q)

        tl.store(
            dK + pid_hz * stride_kh + rn[:, None] * stride_kn + rk[None, :],
            dk.to(tl.float16),
            mask=rn[:, None] < SEQ_LEN,
        )
        tl.store(
            dV + pid_hz * stride_vh + rn[:, None] * stride_vn + rk[None, :],
            dv.to(tl.float16),
            mask=rn[:, None] < SEQ_LEN,
        )

# ============================================================
# Test
# ============================================================
def test():
    B, H, S, D = 1, 4, 512, 64
    q, k, v = [
        torch.randn((B, H, S, D), device="cuda", dtype=torch.float16, requires_grad=True)
        for _ in range(3)
    ]

    mask = torch.tril(torch.ones((B, H, S, S), device="cuda"))
    mask = torch.where(mask > 0, 0.0, float("-inf")).to(torch.float16)

    do = torch.randn_like(q)
    scale = 1.0 / math.sqrt(D)

    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out_ref.backward(do)
    ref_grads = [q.grad.clone(), k.grad.clone(), v.grad.clone()]

    q.grad = k.grad = v.grad = None

    out_tri = FlashAttentionSafe.apply(q, k, v, mask, scale)
    out_tri.backward(do)

    print("Output Diff:", (out_ref - out_tri).abs().max().item())
    print("dQ Diff:", (ref_grads[0] - q.grad).abs().max().item())
    print("dK Diff:", (ref_grads[1] - k.grad).abs().max().item())
    print("dV Diff:", (ref_grads[2] - v.grad).abs().max().item())


if __name__ == "__main__":
    test()
