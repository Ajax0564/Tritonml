import torch
import torch.nn.functional as F


class MemoryEfficientLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias, targets,
                chunk_size=256, ignore_index=-100):
        """
        inputs:  [M, K]
        weight:  [N, K]
        targets: [M]
        """
        M, K = inputs.shape
        device = inputs.device

        lse = torch.empty(M, device=device, dtype=torch.float32)
        tgt_logits = torch.empty(M, device=device, dtype=torch.float32)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)

            x = inputs[start:end]          # [c, K]
            t = targets[start:end]         # [c]

            logits = F.linear(x, weight, bias)  # [c, N]
            lse_chunk = torch.logsumexp(logits, dim=-1) #logj∑​ezj​

            lse[start:end] = lse_chunk

            # target logit extraction
            mask = t != ignore_index
            if mask.any():
                tgt_logits[start:end].zero_()
                tgt_logits[start:end][mask] = logits[
                    mask, t[mask]
                ]
            else:
                tgt_logits[start:end].zero_()

        valid_mask = targets != ignore_index
        num_valid = valid_mask.sum()

        loss = (lse[valid_mask] - tgt_logits[valid_mask]).sum()
        loss = loss / num_valid.clamp(min=1)

        ctx.save_for_backward(inputs, weight, bias, targets, lse)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.num_valid = num_valid

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias, targets, lse = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        ignore_index = ctx.ignore_index

        M, K = inputs.shape
        N = weight.shape[0]
        device = inputs.device

        grad_input = torch.empty_like(inputs)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        scale = grad_output / ctx.num_valid.clamp(min=1)

        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)

            x = inputs[start:end]
            t = targets[start:end]
            lse_chunk = lse[start:end]

            # compute logits 
            logits = F.linear(x, weight, bias)


            probs = torch.exp(logits - lse_chunk.unsqueeze(1)) #softmax(z)

            mask = t != ignore_index
            if mask.any():
                # subtract 1 from correct class
                # softmaxij​−1[j=yi​]
                probs.scatter_add_(
                    1,
                    t.clamp(min=0).unsqueeze(1),
                    (-mask.float()).unsqueeze(1)
                )
                probs *= mask.unsqueeze(1)
            else:
                probs.zero_()

            probs.mul_(scale)

            # gradients
            grad_input[start:end] = probs @ weight
            grad_weight.addmm_(probs.t(), x)

            if grad_bias is not None:
                grad_bias.add_(probs.sum(0))

        return grad_input, grad_weight, grad_bias, None, None, None
        
class MemoryEfficientLinearCEModule(torch.nn.Module):
    def __init__(self, chunk_size=256, ignore_index=-100):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(self, inputs, weight, bias, targets):
        return MemoryEfficientLinearCE.apply(
            inputs, weight, bias, targets,
            self.chunk_size, self.ignore_index
        )
def test_memory_efficient_linear_ce():
    torch.manual_seed(42)

    # Parameters
    M, K, N = 128, 64, 10  # batch, input dim, num classes
    chunk_size = 32
    ignore_index = -100

    # Random inputs
    inputs = torch.randn(M, K, requires_grad=True)
    weight = torch.randn(N, K, requires_grad=True)
    bias = torch.randn(N, requires_grad=True)
    targets = torch.randint(0, N, (M,), dtype=torch.long)

    # Randomly set some targets to ignore_index
    mask_ignore = torch.rand(M) < 0.1
    targets[mask_ignore] = ignore_index

 
    inputs_pt = inputs.clone().detach().requires_grad_(True)
    weight_pt = weight.clone().detach().requires_grad_(True)
    bias_pt = bias.clone().detach().requires_grad_(True)

    # Compute logits
    logits = F.linear(inputs_pt, weight_pt, bias_pt)
    loss_pt = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='mean')
    loss_pt.backward()

    grads_pt = (inputs_pt.grad.clone(), weight_pt.grad.clone(), bias_pt.grad.clone())

  
    inputs_me = inputs.clone().detach().requires_grad_(True)
    weight_me = weight.clone().detach().requires_grad_(True)
    bias_me = bias.clone().detach().requires_grad_(True)

    loss_me = MemoryEfficientLinearCE.apply(inputs_me, weight_me, bias_me, targets, chunk_size, ignore_index)
    loss_me.backward()

    grads_me = (inputs_me.grad.clone(), weight_me.grad.clone(), bias_me.grad.clone())


    atol = 1e-6
    rtol = 1e-5
    print("Forward pass difference:", abs(loss_pt.item() - loss_me.item()))
    assert torch.allclose(loss_pt, loss_me, atol=atol, rtol=rtol), "Forward pass mismatch!"
    assert torch.allclose(grads_pt[0], grads_me[0], atol=atol, rtol=rtol), "Grad input mismatch!"
    assert torch.allclose(grads_pt[1], grads_me[1], atol=atol, rtol=rtol), "Grad weight mismatch!"
    assert torch.allclose(grads_pt[2], grads_me[2], atol=atol, rtol=rtol), "Grad bias mismatch!"
    print("✅ All checks passed: forward and backward match PyTorch native CE.")


if __name__ == "__main__":
    test_memory_efficient_linear_ce()

