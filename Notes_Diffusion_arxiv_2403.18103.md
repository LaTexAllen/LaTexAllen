# Score-Matching Langevin Dynamics(SMLD) 

[Reference](https://arxiv.org/pdf/2403.18103)

## I. Langevin Dynamics

The **Langevin dynamics** for sampling from a known distribution $p(\bold{x})$ is an iterative procedure for $t = 1,\dots , T$ :

$$\bold{x}_{t+1} = \bold{x}_t + \tau \nabla_\bold{x} \log p(\bold{x}_t) +\sqrt{2\tau} \bold{z}, \bold{z}~\sim \mathcal{N}(0,\bold I),$$

where $\tau$ is the step size which users can control, and $\bold{x}_0$ is white noise, $\lim\limits_{t \rightarrow T}\bold{x}_t = p(\bold{x})$.

Without $\sqrt{2\tau} \bold{z}$ (the noise term), the equation is **gradient descent**. With $\sqrt{2\tau} \bold{z}$, the equation is **stochastic gradient descent**.

| Problem | Sampling  |	Maximum Likelihood |
|:--------:| :---------:|:--------:|
| Optimization Target | A sample $\bold{x}$ |Model parameter $\bm{\theta}$ |
| Formulation | $\bold{x}^*=\arg \mathop{\max}\limits_{\bold{x}} \log p(\bold{x}; \bm{\theta})$ |$\bm{\theta}^*=\arg \mathop{\max}\limits_{\bold{x}} \log p(\bold{x}; \bm{\theta}) $|

Langevin Dynamic Equation is easily obtained by applying Brownian Motion to the Netwon second order equation.

## II. (Stein’s) Score Function
The Stein’s score function is the gradient wrt the data point $\bold{x}$.
$$\bold{s}_{\bm{\theta}}(\bold{x}) \coloneqq \nabla_\bold{x} \log p_{\bm{\theta}}(\bold{x}).$$

Do not confuse it with the **ordinary score function** $\bold{s_x}(\bm{\theta}) \coloneqq \nabla_\bold{x} \log p_{\bm{\theta}}(\bold{x})$ (the gradient of log-likelihood wrt $\bm{\theta}$). Maximum likelihood estimation uses the ordinary score function, whereas Langevin dynamics uses Stein’s score function. 

$$\nabla_\bold{x} \log p_{\bm{\theta}}(\bold{x}) = \textit{a vector field} = [\frac{\partial \log p(\bold{x})}{\partial x}, \frac{\partial \log p(\bold{x})}{\partial y}]^T.$$

The data points *drift* in the direction of the vector field. 

## III. Score Matching Techniques
**Explicit Score Matching**
Given a dataset $\chi=\bold{x}_1, \dots, \bold{x}_M$, we consider the kernel density estimation by defining a distribution 

$$q(\bold{x})=\frac{1}{M} \sum_{m=1}^M \frac{1}{h}K(\frac{\bold{x}-\bold{x}_m}{h}),$$ 

where $h$ is just some hyperparameter for the kernel function $K(·)$, and $\bold{x}_m$ is the m-th sample in the training set.

Since $q(\bold{x})$ is an approximation to $p(\bold{x})$ which is never accessible, we can learn $\bold{s}_{\bm{\theta}}(\bold{x})$ based on $q(\bold{x})$. This leads to the following definition of the a loss function which can be used to train a network.

The **explicit score matching loss** is

$$\begin{aligned}
J_{\mathrm{ESM}}(\bm{\theta}) &\coloneqq \mathbb E_{q(\bold{x})}\parallel \bold{s}_{\bm{\theta}}(\bold{x}) - \nabla_\bold{x} \log q(\bold{x})  \parallel^2.\\
&= \int \parallel \bold{s}_{\bm{\theta}}(\bold{x}) - \nabla_\bold{x} \log q(\bold{x})  \parallel^2 [\frac{1}{M} \sum_{m=1}^M \frac{1}{h}K(\frac{\bold{x}-\bold{x}_m}{h})] d\bold{x} \\
&= \frac{1}{M} \sum_{m=1}^M \int \parallel \bold{s}_{\bm{\theta}}(\bold{x}) - \nabla_\bold{x} \log q(\bold{x})  \parallel^2 \frac{1}{h}K(\frac{\bold{x}-\bold{x}_m}{h}) d\bold{x}.
\end{aligned}
$$

So, we have derived a loss function that can be used to train the network. Once we train the network $\bold{s}_{\bm{\theta}}$, we can replace it in the Langevin dynamics equation to obtain the recursion:

$$\bold{x}_{t+1} = \bold{x}_t + \tau \bold{s}_{\bm{\theta}}(\bold{x}_t) +\sqrt{2\tau} \bold{z}.$$

**Remark**: a fairly poor non-parameter estimation of the true distribution.

**Denoising Score Matching (DSM)** more popular than ESM

$$\begin{aligned}
J_{\mathrm{DSM}}(\bm{\theta}) \coloneqq \mathbb E_{q(\bold{x, x'})}[\frac{1}{2}\parallel \bold{s}_{\bm{\theta}}(\bold{x}) - \nabla_\bold{x} \log q(\bold{x|x'})  \parallel^2],
\end{aligned}
$$

where we replace the distribution $q(\bold{x})$ by a conditional distribution $q(\bold{x|x'})$.

In the special case where $q(\bold{x|x'}) = \mathcal{N}(\bold{x|x'}, \sigma^2)$ and $\bold{x}= \bold{x'} + \sigma \bold{z}$, DSM's loss function is 

$$\begin{aligned}
J_{\mathrm{DSM}}(\bm{\theta}) &= \mathbb E_{q(\bold{x'})}[\frac{1}{2}\parallel \bold{s}_{\bm{\theta}}(\bold{x'+\sigma z}) + \frac{\bold{z}}{\sigma^2} \parallel^2], \\
&=\mathbb E_{p(\bold{x})}[\frac{1}{2}\parallel \bold{s}_{\bm{\theta}}(\bold{x+\sigma z}) + \frac{\bold{z}}{\sigma^2} \parallel^2]
\end{aligned}
$$

if we replace the dummy variable $\bold{x}'$ by $\bold{x}$, and we note that sampling from $q(\bold{x})$ can be replaced by sampling from $p(\bold{x})$ when we are given a training dataset.

The **training step** can simply described as follows: You give us a training dataset $\{\bold{x}^{(\mathcal l)}\}_{\mathcal l=1}^L$, we train a network $\bm{\theta}$ with the goal to

$$
\bm{\theta}^* = \arg \mathop{\min}\limits_{\bm{\theta}} \frac{1}{L}\sum_{\mathcal l=1}^L \frac{1}{2}\parallel \bold{s}_{\bm{\theta}}(\bold{x}^{(\mathcal l)}+\sigma \bold{z}^{(\mathcal l)}) + \frac{\bold{z}^{(\mathcal l)}}{\sigma^2} \parallel^2, \bold{z}^{(\mathcal l)} \sim \mathcal{N}(0,\bold I).
$$

**Vincent Theorem**: For up to a constant $C$ which is independent of the variable $\bm{\theta}$, it holds that $J_{\mathrm{DSM}}(\bm{\theta}) = J_{\mathrm{ESM}}(\bm{\theta}) + C$.
[For proof, see Vincent's PDF.](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)

# Stochastic Differential Equation (SDE)

## I. Motivating Examples
1. ODE for analytic solution.
2. Gradient Descent in convex function.

## II. Forward and Backward Iterations in SDE
If we introduce a noise term $\bold{z}_t~\sim \mathcal{N}(0,\bold I)$ to the gradient descent algorithm, then the ODE will become a stochastic differential equation (SDE).

Define $\bold{x}$ as a continuous function for $0 \leq t \leq 1$ and a sequence $\{\frac{i}{N}|i=0,1,\dots,N\}$. Then we have $\bold x_i=\bold x(\frac{i}{N})$, $\Delta t=\frac{1}{N}$.
$$
\begin{aligned}
\bold{x}_i &= \bold{x}_{i-1} - \tau \nabla f(\bold{x}_{i-1}) + \bold{z}_{i-1}, \\
\Rightarrow \bold{x}(t+\Delta t) &= \bold{x}(t) - \tau \nabla f(\bold{x}(t)) + \bold{z}(t).
\end{aligned}
$$

Define a random process $\bold{w}(t)$ such that $\bold{z}(t) = \bold{w}(t + \Delta t) - \bold{w}(t) \approx \frac{d \bold{w}(t)}{dt} \Delta t$. Then we have
$$d\bold x = -\tau \nabla f(\bold{x}) dt + d\bold{w}.$$

In summary, we have **Forward Diffusion**:
$$d\bold x = \underbrace{\bold f(\bold{x}, t)}_{\rm drift} dt + \underbrace{g(t)}_{\rm diffusion} d\bold{w}.$$

1. $\bold f(\bold{x}, t)$: how molecules in a closed system would move in the absence of random effects.
2. $g(t)$: how the molecules would randomly walk from one position to another.

**Reverse SDE**: [See Anderson's PDF](https://www.sciencedirect.com/science/article/pii/0304414982900515)
$$d\bold x = [\underbrace{\bold f(\bold{x}, t)}_{\rm drift} dt - g(t)^2 \underbrace{\nabla_{\bold{x}}\log p_t(\bold{x})}_{\rm{score \ function}}] dt + \underbrace{g(t) d\bar{\bold{w}}}_{\rm {reverse-time \ diffusion}},$$

where $p_t(\bold{x})$ is the probability distribution of $\bold{x}$ at time $t$, and $\bar{\bold{w}}$ is the Wiener process when time flows backward.
