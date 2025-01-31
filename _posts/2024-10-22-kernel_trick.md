---
title: The Kernel Trick 
subtitle: I write about the kernel trick and show an informal derivation when applied to a regularized least squares example
shorttitle: Kernel Trick
layout: default
date: 2024-10-23
keywords: blogging, writing
published: true
---

In this post, I will demonstrate an intuitive derivation of the kernelized form of L2-regularized least squares as an illustration of how the kernel trick can be useful.

Let's begin by solving the basic L2-regularized least squares problem.

Solving the L2-regularized Least Squares Problem
---
The regularized least squares problem is:
{% katexmm %}
$$
\underset{w \space \epsilon \space R_M}\text{Min} \space \frac{1}{2N} [ \Vert \Phi w - y \Vert^2 + \lambda \Vert w \Vert^2 ]
$$
{% endkatexmm %}

Recall that $\Phi_{ij} = \phi_j (x_i)$, <b>whereby our design matrix $\Phi$ has dimensions $N×M$ (this is an important detail).</b>

This is also known as ridge regression.

Now, to derive the solution to the ridge regression empirical risk minimization:

{% katexmm %}
$$
\begin{aligned}
J(\hat{w}) &= (y - \Phi \hat{w}) (y - \Phi \hat{w}) + \lambda \hat{w}^T \hat{w} \\
&= y^Ty - 2 y^T \Phi \hat{w} + \hat{w}^T \Phi^T \Phi \hat{w} + \lambda \hat{w}^T \hat{w} \\
\frac{dJ(\hat{w})}{d\hat{w}} &= -2 y^T \Phi + 2 \Phi \Phi \hat{w} + 2 \lambda \hat{w} \\
&= 0 \\
\Phi^T \Phi \hat{w} + \lambda \hat{w} &= \Phi^T y \\
(\Phi^T \Phi + \lambda I_m)\hat{w} &= \Phi^T y \\
\end{aligned}
$$
{% endkatexmm %}

{% katexmm %}
$$
\boxed{\hat{w} = (\Phi^T \Phi + \lambda I_m)^{-1} \Phi^T y}
$$
{% endkatexmm %}

Note that in the above solution for $\hat{w}$, the dataset $(x_i,y_i)_{i=N}$ is "memorized" by $\hat{w}$ and is not needed for new predictions.

Next, let's see how rewriting the regularized least squares solution can produce a fascinatingly different result.


Reframing the regularized least squares solution
---
We will begin with the claim that:
{% katexmm %}
$$
\hat{w} = (\Phi^T \Phi + \lambda I_m)^{-1} \Phi^T y = \fcolorbox{orange}{white}{$\Phi^T(\Phi \Phi^T + \lambda I_N)^{-1}y$}
$$
{% endkatexmm %}

And here is the (informal?) proof:

{% katexmm %}
$$
\begin{aligned}
(\Phi^T \Phi + \lambda I_m) \overbrace{\Phi^T (\Phi \Phi^T + \lambda I_N)^{-1}y}^{\color{orange} \circledast} &= (\Phi^T \Phi \Phi^T + \lambda \Phi^T) (\Phi \Phi^T + \lambda I_N)^{-1} y \\
&= \Phi^T (\Phi \Phi^T + \lambda I_N)(\Phi \Phi^T + \lambda I_N)^{-1}y \\
&= \Phi^T y \\
\text{So, } \space \overbrace{\Phi^T (\Phi \Phi^T + \lambda I_N)^{-1}y}^{\color{orange} \circledast} &= (\Phi^T \Phi + \lambda I_m)^{-1} \Phi y \\
~ \\
\text{Finally, } \space \hat{f}(x) &= \hat{w}^T \phi(x) \\
&= \phi(x)^T \hat{w} \\
&= \phi(x)^T \Phi^T(\Phi \Phi^T + \lambda I_N)^{-1} y \\
&= \displaystyle\sum_{i=1}^N \alpha_i \phi(x)^T \phi(x_i) \\
~ \\
\text{Whereby } \space \alpha &= (\Phi \Phi^T + \lambda I_N)^{-1}y \\
&= (\underbrace{G}_{\text{Gram Matrix}} + \lambda I_N)^{-1} y \\
~ \\
\text{And} \space G_{ij} &= \phi(x_i)^T \phi(x_j)
\end{aligned}
$$
{% endkatexmm %}

Let's now summarize our findings for the reframed ridge regression solution:

{% katexmm %}
$$
\fcolorbox{orange}{white}{$\hat{f}(x) = \displaystyle\sum_{i=1}^N \alpha_i \phi(x_i)^T \phi(x)$} \space \text{, where} \space
\alpha = (G + \lambda I_N)^{-1}y
$$
{% endkatexmm %}

We can make two key observations:
1. The input data {$x_i$} participates in predictions
2. For each new prediction, we have $O(N)$ operations

Now, to see how this reframing is useful, let's make some comparisons with the original ridge regression solution.


Comparing two formulations of the solution
---

{% katexmm %}
$$
\begin{array}{c:c}
\text{Original Solution} & \text{Reformulated Solution} \\
~ & ~ \\
\hat{f}(x) = \displaystyle\sum_{j=0}^{M-1} \hat{w}_j^T \phi_j(x) &
\hat{f}(x) = \displaystyle\sum_{i=1}^{N} \alpha_i \phi(x_i)^T \phi(x) \\
~ & ~ \\
\text{Where} \space \hat{w} = (\underbrace{\Phi^T \Phi}_{\text{M×M}} + \lambda I_m)^{-1} \Phi^T y &
\text{Where} \space \alpha = (G + \lambda I_N)^{-1}y \space \text{ , } \space G = \underbrace{\phi(x_i)^T \phi(x_j)}_{\text{N×N}} \\ ~ & ~
\end{array}
$$
{% endkatexmm %}

These two formulations are the exact same function, but we can make the following key comparisons:
1. The original solution has $O(M)$ operations, while the reformulated solution only has $O(N)$ operations. There are quite some cases where M >> N for the N×M design matrix $\Phi$. Thus, the N×N gram matrix $\Phi \Phi^T$ is much more computationally feasible. 
2. Most importantly, the reformulated solution only depends on feature maps {$\phi_j$} through dot product $\phi(x)^T \phi(x')$


Next, let's see how we can generalize this solution to kernel form

Generalizing to Kernel Form
---

{% katexmm %}
$$
\text{So, we have our reformulated solution: } \\
\hat{f}(x) = \displaystyle\sum_{i=1}^{N} \alpha_i \phi(x_i)^T \phi(x) \\
\Downarrow \\
\text{Notice we have a similarity measure in the form of } \phi(x)^T \phi(x') \space \text{,} \\
\text{which involves computing the product of the two feature mappings} \\
\Downarrow \\
\text{What if we forget about the feature map } \phi \\
\text{and define a kernel } K(x,x') = \phi(x)^T\phi(x') 
~ \\
\Downarrow \\
\text{Such that} \\
\fcolorbox{orange}{white}{$\hat{f}(x) = \displaystyle\sum_{i=1}^{N} \alpha_i K(x_i,x)$}, \\
~ \\
\fcolorbox{orange}{white}{$\alpha = (G + \lambda I_N)^{-1} y,
G_{ij} = K(x_i, x_j)$} \\
~ \\
$$
{% endkatexmm %}

Why is this useful? Given $K(x,x') = \phi(x)^T\phi(x')$, $K(x,x')$ computes the same value as $\phi(x)^T\phi(x')$, but does so without explicitly computing $\phi(x)$. In other words, the kernel encapsulates all the information about the feature space $\phi$ and inner product in it. We don't need to explicitly map $x$ and $x'$ to the feature space via computing $\phi(x)$, then compute the inner product, because $K(x,x')$ provides the result directly in terms of $x$ and $x'$ in the input space.

This might seem suspiciously convenient. Why are we able to swap $\phi(x)^T\phi(x')$ for $K(x,x')$? Well, <a href="https://en.wikipedia.org/wiki/Mercer%27s_theorem" target="_blank">Mercer's Theorem</a> states that if $K$ is a Symmetric Positive Definite kernel, there will exist a feature space $H$ and feature map $\phi$ such that $K(x,x') = \phi(x)^T\phi(x')$. 

In other words, any kernel function that satisfies Mercer's Theorem implicitly corresponds to a valid feature map $\phi(x)$. 

In a sense, this is the core idea of the Kernel Trick.


Constructing a Kernel
---
There are a few ways to construct a kernel. 

Firstly, you could construct a valid kernel by defining your basis function (feature mapping) explicitly, and then finding your kernel.

Next, you could directly define a kernel function as long as it is known to be valid. Some examples of SPD kernels include:

{% katexmm %}
$$
\begin{aligned}
~ \\
\text{Linear Kernel: } k(x,x') &= x^Tx' \space \text{or} \space k(x,x') = 1+x^Tx' \\
~ \\
\text{Polynomial Kernel: } k(x,x') &= (1+x^Tx')^m \space , \space m \space \epsilon \space Z_{+} \\
~ \\
\text{Gaussian (RBF) Kernel: } \space k(x,x') &= \exp (\frac{- \Vert x-x' \Vert^2}{2s^2}) \space , \space s>0 \\
~ \\
\end{aligned}
$$
{% endkatexmm %}

Finally, you could also construct kernels from simpler kernels, by using them as building blocks. For example:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image24.png" width="40%">
</div>

