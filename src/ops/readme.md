

### Mathematical Formulation
## Softmax

online approach to compute softmax stably:

1. **Running Max & Sum Update**:
   For each tile $k$, we update the running max $m^{(k)}$ and sum $d^{(k)}$:
   $$m^{(k)} = \max(x_{tile})$$
   $$\alpha = e^{m^{(k-1)} - m^{(k)}}$$
   $$d^{(k)} = d^{(k-1)} \cdot \alpha + \sum e^{x_{tile} - m^{(k)}}$$

2. **Final Output**:
   Once the global row statistics $m^{(N)}$ and $d^{(N)}$ are found:
   $$y_{i,j} = \frac{e^{x_{i,j} - m^{(N)}}}{d^{(N)}}$$


### Mathematical Proof
To ensure numerical stability without multiple passes, we use the following identity to update the denominator $d$ when the running maximum changes from $m_{old}$ to $m_{new}$:

$$d_{new} = d_{old} \cdot e^{m_{old} - m_{new}} + \sum e^{x_{tile} - m_{new}}$$

**Derivation:**
Since $d_{old} = \sum e^{x_{prev} - m_{old}}$, multiplying by $e^{m_{old} - m_{new}}$ shifts the baseline of the previous sum:
$$\left(\sum e^{x_{prev} - m_{old}}\right) e^{m_{old} - m_{new}} = \sum e^{x_{prev} - m_{new}}$$
This allows us to simply add the new tile's values as they now share a common denominator base ($m_{new}$).




## Attention

### Forward
$$
\begin{aligned}
S &= \frac{QK^\top}{\sqrt{d_k}} \\
A &= \text{softmax}(S) \\
O &= AV
\end{aligned}
$$

#### Backward
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial V} &= A^\top G \\
\frac{\partial \mathcal{L}}{\partial A} &= G V^\top \\
\frac{\partial \mathcal{L}}{\partial S} &= \text{softmax\_backward}(A, \frac{\partial \mathcal{L}}{\partial A}) \\
\frac{\partial \mathcal{L}}{\partial Q} &= \frac{1}{\sqrt{d_k}} \frac{\partial \mathcal{L}}{\partial S} K \\
\frac{\partial \mathcal{L}}{\partial K} &= \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top Q
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial V}
&= A^\top G
\\[10pt]
\frac{\partial \mathcal{L}}{\partial Q}
&=
\frac{1}{\sqrt{d_k}}
\left[
\left(
A \odot
\left(
GV^\top
-
\left(
(A \odot GV^\top)\mathbf{1}
\right)
\right)
\right)
K
\right]
\\[10pt]
\frac{\partial \mathcal{L}}{\partial K}
&=
\frac{1}{\sqrt{d_k}}
\left[
\left(
A \odot
\left(
GV^\top
-
\left(
(A \odot GV^\top)\mathbf{1}
\right)
\right)
\right)^\top
Q
\right]
\end{aligned}
$$

### Online Softmax Update (Iterative)
To process the attention scores in blocks, we use the following update rules for each row block $i$ and column block $j$:

$$
\begin{aligned}
m_i^{new} &= \max(m_i^{old}, \max(QK_{ij})) \\
\alpha &= \exp(m_i^{old} - m_i^{new}) \\
l_i^{new} &= l_i^{old} \cdot \alpha + \sum \exp(QK_{ij} - m_i^{new}) \\
\text{acc}^{new} &= \text{acc}^{old} \cdot \alpha + \text{matmul}(\exp(QK_{ij} - m_i^{new}), V_j)
\end{aligned}
$$

### Log-Sum-Exp (LSE) Computation
The kernel stores the LSE for the backward pass. The identity used is:
$$LSE = m_i + \log(l_i)$$

**Proof**: 
Given $l_i = \sum \exp(x_j - m_i)$, then:
$$m_i + \log(\sum \exp(x_j - m_i)) = \log(\exp(m_i)) + \log(\sum \exp(x_j - m_i))$$
Using $\log(A) + \log(B) = \log(A \cdot B)$:
$$\log(\exp(m_i) \cdot \sum \exp(x_j - m_i)) = \log(\sum \exp(x_j - m_i + m_i)) = \log(\sum \exp(x_j))$$
