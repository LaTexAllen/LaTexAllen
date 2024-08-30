# Score-Matching Langevin Dynamics(SMLD) 
  
[Reference](https://arxiv.org/pdf/2403.18103 )
  
## I. Langevin Dynamics
  
The **Langevin dynamics** for sampling from a known distribution <img src="https://latex.codecogs.com/gif.latex?p(\bold{x})"/> is an iterative procedure for <img src="https://latex.codecogs.com/gif.latex?t%20=%201,\dots%20,%20T"/> :
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\bold{x}_{t+1}%20=%20\bold{x}_t%20+%20\tau%20\nabla_\bold{x}%20\log%20p(\bold{x}_t)%20+\sqrt{2\tau}%20\bold{z},%20\bold{z}~\sim%20\mathcal{N}(0,\bold%20I),"/></p>  
  
  
where <img src="https://latex.codecogs.com/gif.latex?\tau"/> is the step size which users can control, and <img src="https://latex.codecogs.com/gif.latex?\bold{x}_0"/> is white noise, <img src="https://latex.codecogs.com/gif.latex?\lim\limits_{t%20\rightarrow%20T}\bold{x}_t%20=%20p(\bold{x})"/>.
  
Without <img src="https://latex.codecogs.com/gif.latex?\sqrt{2\tau}%20\bold{z}"/> (the noise term), the equation is **gradient descent**. With <img src="https://latex.codecogs.com/gif.latex?\sqrt{2\tau}%20\bold{z}"/>, the equation is **stochastic gradient descent**.
  
| Problem | Sampling  |	Maximum Likelihood |
|:--------:| :---------:|:--------:|
| Optimization Target | A sample <img src="https://latex.codecogs.com/gif.latex?\bold{x}"/> |Model parameter <img src="https://latex.codecogs.com/gif.latex?\bm{\theta}"/> |
| Formulation | <img src="https://latex.codecogs.com/gif.latex?\bold{x}^*=\arg%20\mathop{\max}\limits_{\bold{x}}%20\log%20p(\bold{x};%20\bm{\theta})"/> |<img src="https://latex.codecogs.com/gif.latex?\bm{\theta}^*=\arg%20\mathop{\max}\limits_{\bold{x}}%20\log%20p(\bold{x};%20\bm{\theta})"/>|
  
Langevin Dynamic Equation is easily obtained by applying Brownian Motion to the Netwon second order equation.
  
## II. (Stein’s) Score Function
The Stein’s score function is the gradient wrt the data point <img src="https://latex.codecogs.com/gif.latex?\bold{x}"/>.
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\bold{s}_{\bm{\theta}}(\bold{x})%20\coloneqq%20\nabla_\bold{x}%20\log%20p_{\bm{\theta}}(\bold{x})."/></p>  
  
  
Do not confuse it with the **ordinary score function** <img src="https://latex.codecogs.com/gif.latex?\bold{s_x}(\bm{\theta})%20\coloneqq%20\nabla_\bold{x}%20\log%20p_{\bm{\theta}}(\bold{x})"/> (the gradient of log-likelihood wrt <img src="https://latex.codecogs.com/gif.latex?\bm{\theta}"/>). Maximum likelihood estimation uses the ordinary score function, whereas Langevin dynamics uses Stein’s score function. 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\nabla_\bold{x}%20\log%20p_{\bm{\theta}}(\bold{x})%20=%20\text{a%20vector%20field}%20=%20[\frac{\partial%20\log%20p(\bold{x})}{\partial%20x},%20\frac{\partial%20\log%20p(\bold{x})}{\partial%20y}]^T."/></p>  
  
  
The data points *drift* in the direction of the vector field. 
  
## III. Score Matching Techniques
**Explicit Score Matching**
Given a dataset <img src="https://latex.codecogs.com/gif.latex?\chi=\bold{x}_1,%20\dots,%20\bold{x}_M"/>, we consider the kernel density estimation by defining a distribution 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?q(\bold{x})=\frac{1}{M}%20\sum_{m=1}^M%20\frac{1}{h}K(\frac{\bold{x}-\bold{x}_m}{h}),"/></p>  
  
  
where <img src="https://latex.codecogs.com/gif.latex?h"/> is just some hyperparameter for the kernel function <img src="https://latex.codecogs.com/gif.latex?K(·)"/>, and <img src="https://latex.codecogs.com/gif.latex?\bold{x}_m"/> is the m-th sample in the training set.
  
Since <img src="https://latex.codecogs.com/gif.latex?q(\bold{x})"/> is an approximation to <img src="https://latex.codecogs.com/gif.latex?p(\bold{x})"/> which is never accessible, we can learn <img src="https://latex.codecogs.com/gif.latex?\bold{s}_{\bm{\theta}}(\bold{x})"/> based on <img src="https://latex.codecogs.com/gif.latex?q(\bold{x})"/>. This leads to the following definition of the a loss function which can be used to train a network.
  
The **explicit score matching loss** is
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}J_{\mathrm{ESM}}(\bm{\theta})%20&amp;\coloneqq%20\mathbb%20E_{q(\bold{x})}\parallel%20\bold{s}_{\bm{\theta}}(\bold{x})%20-%20\nabla_\bold{x}%20\log%20q(\bold{x})%20%20\parallel^2.\\&amp;=%20\int%20\parallel%20\bold{s}_{\bm{\theta}}(\bold{x})%20-%20\nabla_\bold{x}%20\log%20q(\bold{x})%20%20\parallel^2%20[\frac{1}{M}%20\sum_{m=1}^M%20\frac{1}{h}K(\frac{\bold{x}-\bold{x}_m}{h})]%20d\bold{x}%20\\&amp;=%20\frac{1}{M}%20\sum_{m=1}^M%20\int%20\parallel%20\bold{s}_{\bm{\theta}}(\bold{x})%20-%20\nabla_\bold{x}%20\log%20q(\bold{x})%20%20\parallel^2%20\frac{1}{h}K(\frac{\bold{x}-\bold{x}_m}{h})%20d\bold{x}.\end{aligned}"/></p>  
  
  
So, we have derived a loss function that can be used to train the network. Once we train the network <img src="https://latex.codecogs.com/gif.latex?\bold{s}_{\bm{\theta}}"/>, we can replace it in the Langevin dynamics equation to obtain the recursion:
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\bold{x}_{t+1}%20=%20\bold{x}_t%20+%20\tau%20\bold{s}_{\bm{\theta}}(\bold{x}_t)%20+\sqrt{2\tau}%20\bold{z}."/></p>  
  
  
**Remark**: a fairly poor non-parameter estimation of the true distribution.
  
**Denoising Score Matching (DSM)** more popular than ESM
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}J_{\mathrm{DSM}}(\bm{\theta})%20\coloneqq%20\mathbb%20E_{q(\bold{x,%20x&#39;})}[\frac{1}{2}\parallel%20\bold{s}_{\bm{\theta}}(\bold{x})%20-%20\nabla_\bold{x}%20\log%20q(\bold{x|x&#39;})%20%20\parallel^2],\end{aligned}"/></p>  
  
  
where we replace the distribution <img src="https://latex.codecogs.com/gif.latex?q(\bold{x})"/> by a conditional distribution <img src="https://latex.codecogs.com/gif.latex?q(\bold{x|x&#39;})"/>.
  
In the special case where <img src="https://latex.codecogs.com/gif.latex?q(\bold{x|x&#39;})%20=%20\mathcal{N}(\bold{x|x&#39;},%20\sigma^2)"/> and <img src="https://latex.codecogs.com/gif.latex?\bold{x}=%20\bold{x&#39;}%20+%20\sigma%20\bold{z}"/>, DSM's loss function is 
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}J_{\mathrm{DSM}}(\bm{\theta})%20&amp;=%20\mathbb%20E_{q(\bold{x&#39;})}[\frac{1}{2}\parallel%20\bold{s}_{\bm{\theta}}(\bold{x&#39;+\sigma%20z})%20+%20\frac{\bold{z}}{\sigma^2}%20\parallel^2],%20\\&amp;=\mathbb%20E_{p(\bold{x})}[\frac{1}{2}\parallel%20\bold{s}_{\bm{\theta}}(\bold{x+\sigma%20z})%20+%20\frac{\bold{z}}{\sigma^2}%20\parallel^2]\end{aligned}"/></p>  
  
  
if we replace the dummy variable <img src="https://latex.codecogs.com/gif.latex?\bold{x}&#39;"/> by <img src="https://latex.codecogs.com/gif.latex?\bold{x}"/>, and we note that sampling from <img src="https://latex.codecogs.com/gif.latex?q(\bold{x})"/> can be replaced by sampling from <img src="https://latex.codecogs.com/gif.latex?p(\bold{x})"/> when we are given a training dataset.
  
The **training step** can simply described as follows: You give us a training dataset <img src="https://latex.codecogs.com/gif.latex?\{\bold{x}^{(\mathcal%20l)}\}_{\mathcal%20l=1}^L"/>, we train a network <img src="https://latex.codecogs.com/gif.latex?\bm{\theta}"/> with the goal to
  
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\bm{\theta}^*%20=%20\arg%20\mathop{\min}\limits_{\bm{\theta}}%20\frac{1}{L}\sum_{\mathcal%20l=1}^L%20\frac{1}{2}\parallel%20\bold{s}_{\bm{\theta}}(\bold{x}^{(\mathcal%20l)}+\sigma%20\bold{z}^{(\mathcal%20l)})%20+%20\frac{\bold{z}^{(\mathcal%20l)}}{\sigma^2}%20\parallel^2,%20\bold{z}^{(\mathcal%20l)}%20\sim%20\mathcal{N}(0,\bold%20I)."/></p>  
  
  
**Vincent Theorem**: For up to a constant <img src="https://latex.codecogs.com/gif.latex?C"/> which is independent of the variable <img src="https://latex.codecogs.com/gif.latex?\bm{\theta}"/>, it holds that <img src="https://latex.codecogs.com/gif.latex?J_{\mathrm{DSM}}(\bm{\theta})%20=%20J_{\mathrm{ESM}}(\bm{\theta})%20+%20C"/>.
[For proof, see Vincent's PDF.](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf )
  
# Stochastic Differential Equation (SDE)
  
## I. Motivating Examples
1. ODE for analytic solution.
2. Gradient Descent in convex function.
  
## II. Forward and Backward Iterations in SDE
If we introduce a noise term <img src="https://latex.codecogs.com/gif.latex?\bold{z}_t~\sim%20\mathcal{N}(0,\bold%20I)"/> to the gradient descent algorithm, then the ODE will become a stochastic differential equation (SDE).
  
Define <img src="https://latex.codecogs.com/gif.latex?\bold{x}"/> as a continuous function for <img src="https://latex.codecogs.com/gif.latex?0%20\leq%20t%20\leq%201"/> and a sequence <img src="https://latex.codecogs.com/gif.latex?\{\frac{i}{N}|i=0,1,\dots,N\}"/>. Then we have <img src="https://latex.codecogs.com/gif.latex?\bold%20x_i=\bold%20x(\frac{i}{N})"/>, <img src="https://latex.codecogs.com/gif.latex?\Delta%20t=\frac{1}{N}"/>.
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}\bold{x}_i%20&amp;=%20\bold{x}_{i-1}%20-%20\tau%20\nabla%20f(\bold{x}_{i-1})%20+%20\bold{z}_{i-1},%20\\\Rightarrow%20\bold{x}(t+\Delta%20t)%20&amp;=%20\bold{x}(t)%20-%20\tau%20\nabla%20f(\bold{x}(t))%20+%20\bold{z}(t).\end{aligned}"/></p>  
  
  
Define a random process <img src="https://latex.codecogs.com/gif.latex?\bold{w}(t)"/> such that <img src="https://latex.codecogs.com/gif.latex?\bold{z}(t)%20=%20\bold{w}(t%20+%20\Delta%20t)%20-%20\bold{w}(t)%20\approx%20\frac{d%20\bold{w}(t)}{dt}%20\Delta%20t"/>. Then we have
<p align="center"><img src="https://latex.codecogs.com/gif.latex?d\bold%20x%20=%20-\tau%20\nabla%20f(\bold{x})%20dt%20+%20d\bold{w}."/></p>  
  
  
In summary, we have **Forward Diffusion**:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?d\bold%20x%20=%20\underbrace{\bold%20f(\bold{x},%20t)}_{\rm%20drift}%20dt%20+%20\underbrace{g(t)}_{\rm%20diffusion}%20d\bold{w}."/></p>  
  
  
1. <img src="https://latex.codecogs.com/gif.latex?\bold%20f(\bold{x},%20t)"/>: how molecules in a closed system would move in the absence of random effects.
2. <img src="https://latex.codecogs.com/gif.latex?g(t)"/>: how the molecules would randomly walk from one position to another.
  
**Reverse SDE**: [See Anderson's PDF](https://www.sciencedirect.com/science/article/pii/0304414982900515 )
<p align="center"><img src="https://latex.codecogs.com/gif.latex?d\bold%20x%20=%20[\underbrace{\bold%20f(\bold{x},%20t)}_{\rm%20drift}%20dt%20-%20g(t)^2%20\underbrace{\nabla_{\bold{x}}\log%20p_t(\bold{x})}_{\rm{score%20\%20function}}]%20dt%20+%20\underbrace{g(t)%20d\bar{\bold{w}}}_{\rm%20{reverse-time%20\%20diffusion}},"/></p>  
  
  
where <img src="https://latex.codecogs.com/gif.latex?p_t(\bold{x})"/> is the probability distribution of <img src="https://latex.codecogs.com/gif.latex?\bold{x}"/> at time <img src="https://latex.codecogs.com/gif.latex?t"/>, and <img src="https://latex.codecogs.com/gif.latex?\bar{\bold{w}}"/> is the Wiener process when time flows backward.
  
## III. Stochastic Differential Equation for DDPM
To draw the connection between DDPM and SDE, we consider the discrete-time DDPM iteration.
For <img src="https://latex.codecogs.com/gif.latex?i=0,1,\dots,N"/>:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\bold{x}_i%20=%20\sqrt{1-\beta_i}\bold{x}_{i-1}+\sqrt{\beta_i}\bold{z}_{i-1},%20\bold{z}_{i-1}~\sim%20N(0,\bold{I})."/></p>  
  
  
The forward sampling equation of **DDPM** can be written as an SDE via
<p align="center"><img src="https://latex.codecogs.com/gif.latex?d\bold{x}%20=%20\underbrace{−\frac{\beta(t)}{2}\bold{x}}_{\bold{f}(\bold{x},t)}dt%20+%20\underbrace{\sqrt{\beta(t)}}_{g(t)}d\bold{w}."/></p>  
  
  