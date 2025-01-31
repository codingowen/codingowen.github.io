---
title: Hessian Matrices for Convex Optimization
subtitle: I try to construct a geometric intuition for why Hessian matrices must be Positive Semi-Definite in convex optimization.
shorttitle: Hessian Matrices
layout: default
date: 2025-01-17
keywords: blogging, writing
published: true
---

In one 'Convex Optimization Algorithms' class, we learned about the necessary conditions for being a 'local minimizer' of a given function $f(x)$. One of the conditions stated that, for a point $x_0$ to be a local minimizer of $f(x)$, the Hessian $H_f (x_0)$ has to be positive semidefinite. 

This struck me as being related to some powerful properties in linear algebra I had learned about before. Given there was not much further explanation about why, I decided to do some reading about it, and found some nice, intuitive geometric explanations for this condition. I will attempt to explain my learnings as simply as possible.

To begin, we need to first define some of the terms and the properties we're going to work with. Then, we'll explore some explanations for the necessary condition, and end off with the explanation I find most intuitive.

1\. What is a local minimizer?
---
Well, given the context of the Optimization Algorithms class, we're usually trying to minimize a given function $f(x)$. The graph of $f(x)$ against $x$ might look something like this:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image81.png" width="60%">
</div>

And if we were to label the local and global minimizers on the graph:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image82.png" width="60%">
</div>

So, we can observe that a 'minimizer' is a particular $x$ value/point that has a corresponding minimum value of $f(x)$.
We can also see that the global minimizer is a local minimizer. However, the converse is not true in general.



2\. What is a Hessian?
---
The Hessian matrix $H_f(x)$ is the matrix of all second partial derivatives of $f(x)$. More formally:

{% katexmm %} 
$$ 
H_f (x)
= 
\begin{bmatrix}
\frac{∂^2 f}{∂ x_1^2} & \frac{∂^2 f}{∂ x_1 ∂ x_2} & \cdots \\
\frac{∂^2 f}{∂ x_2 ∂ x_1} & \frac{∂^2 f}{∂ x_2^2} & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$ 
{% endkatexmm %}

The $ij$-entry of $H_f(x)$ is $\frac{∂^2 f}{∂ x_i ∂ x_j}$. In general, $H(x)$ is not symmetric. However, if $f(x)$ has continuous second order derivatives, then the Hessian matrix is symmetric.



3\. What is a Positive Semi-Definite Matrix? 
---
A Positive Semi-Definite Matrix can be simply defined as a square, symmetric matrix where all its eigenvalues are non-negative.

It has a not-so-obvious-to-me mathematical definition like so:

{% katexmm %} 
$$ 
\boxed{\text{A is Positive Semi-Definite } \Longleftrightarrow x^TAx \ge 0 \space \forall \space x \epsilon \Reals^n} \\
~ \\
$$ 
{% endkatexmm %}

Why is the conditional quadratic form $x^TAx$ required for PSD matrices? 

The quadratic form $x^T Ax$ is a scalar value that captures how A transforms $x$ in an inner-product kind of way. For instance, if $A$ has an eigenvalue decomposition $A = QΛQ^T$, then:

{% katexmm %} 
$$ 
\boxed{x^TAx = x^TQΛQ^Tx = (Q^Tx)^T Λ (Q^Tx)} \\
~ \\
$$ 
{% endkatexmm %}

Since $Λ$ is a diagonal matrix with eigenvalues as its entries, the expression above simplifies to:

{% katexmm %} 
$$ 
\boxed{x^TAx = \displaystyle\sum_{i} \lambda_i (z_i)^2 \space \text{where } \space z=Q^Tx} \\
~ \\
$$ 
{% endkatexmm %}

Hence, the quadratic form, $x^TAx$ sums the eigenvalues weighted by squared terms. If all eigenvalues are non-negative, the sum is also non-negative. 

We're now ready to answer our original question. Moving on, we'll see why this quadratic form is important for describing Hessians and finding minimum points.


4\. So, why must the Hessian be PSD?
---
A common explanation starts with the fact that given a twice-differentiable function $f(x)$, we can approximate it around a point $x_0$, using the second-order Taylor Expansion:

{% katexmm %} 
$$ 
\boxed{f(x) \approx f(x_0) + \nabla f(x_0)^T (x-x_0) + \frac{1}{2} (x-x_0)^T H(x_0) (x-x_0)}
~ \\
$$ 
{% endkatexmm %}

Where $\nabla f(x_0)^T$ is the gradient of $f(x)$ at $x_0$, and $H(x_0)$ is the Hessian matrix.

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

To check if $x_0$ is a local minimum, we examine how f(x) behaves near $x_0$. Then, the logic goes:

- Since $x_0 $ is a minimum point, the gradient $\nabla f(x_0) = 0$.

- Then, what remains is the quadratic term $(x-x_0)^T H(x_0) (x-x_0)$. The function $f(x)$ has to curve upwards around $x_0$, so we need $(x-x_0)^T H(x_0) (x-x_0) \ge 0$.

In order to satisfy $(x-x_0)^T H(x_0) (x-x_0) \ge 0$,  we see that this fits the definition of having a PSD Hessian $H(x_0)$.

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

<b><u>To add on to this explanation, I found that thinking about it geometrically (with the help of some eigenvectors) was also helpful:</u></b>

1. When our Hessian is a symmetric matrix, the Hessian has a finite number of eigenvectors which form an orthogonal basis for the entire $n$-th dimensional space.

2. These eigenvectors describe the principal axes of curvature for the given function.

3. Any arbitrary direction $v$ in the space can be expressed as a linear combination of these eigenvectors. For example, $v = c_1 e_1 + c_2 e_2 + \ldots + c_n e_n$. 

4. Moving from arbitrary direction $v$ to curvature $q(v)$, the curvature of our function along an arbitrary direction $v$ is given by the quadratic form $q(v) = v^THv$.

5. By using the eigen-decomposition of $H$ and substituting $v = c_1 e_1 + \ldots + c_n e_n$, we get: 
{% katexmm %} 
$$ 
q(v) = \displaystyle\sum_{i}^n \lambda_i (c_i)^2 \space \text{for each direction } v
$$ 
{% endkatexmm %}

So, even though $H$ has a finite number of eigenvectors, the decomposition above guarantees non-negative curvature for any direction $v$, not just the eigenvector directions. 

Let's visualize this:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image83.png" width="50%">
</div>

We can see the contour plot of a convex quadratic function $f(x,y) = ax^2 + bxy + c y^2$. On the contour plot, The orthogonal eigenvectors of the Hessian matrix $H$ are plotted as red arrows. These eigenvectors indicate the principal directions of curvature of the function. 

Since the Hessian is PSD with non-negative eigenvalues, moving along any eigenvector in the +ve direction results in increasing or zero second-order derivative, ensuring no negative curvature.

Here's the 3D plot of what our example $f(x,y)$ might look like:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image84.png" width="50%">
</div>

Hopefully the red eigenvectors are still visible on the 3D plot.

I hope the geometric explanation involving eigenvectors and the visualizations helped make clear why curvature from the local minimizer is non-negative if the Hessian is Positive Semi-Definite.

Thank you for reading!