---
title: Visualizing some essential transformation matrices
subtitle: I write about some essential matrix transformations that help build an understanding of matrix multiplication as a composition of matrix transformations.
shorttitle: Visualizing Essential Matrices
layout: default
date: 2024-11-01
keywords: blogging, writing
published: true
---

In keeping with the geometric perspective of matrices as linear transformations, this post will introduce some matrices that produce essential transformations.


Identity Matrix
---
The identity matrix is a square matrix (n x n), with $1's$ on the diagonal entries and $0's$ everywhere else. 

{% katexmm %}
$$
\begin{bmatrix} 1 & 0 \\ 0 & 1 \\\end{bmatrix}
$$
{% endkatexmm %}

This matrix is called the 'identity matrix' because when you multiply any vector with the identity matrix, you get the same vector back. 


Scalar Matrix
---
The scalar matrix is a square matrix with constant values on the diagonal and $0's$ everywhere else.

{% katexmm %}
$$
\begin{bmatrix} 2 & 0 \\ 0 & 2 \\\end{bmatrix}
$$
{% endkatexmm %}

We can see the scalar matrix as $n(Im)$ where $Im$ is the identity matrix and $n$ is the scaling factor. The scalar matrix scales your input vector exactly by $n$.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image14.png" width="40%">
</div>


"Off by One" Matrix
---
The "Off by One" matrix is like the identity matrix, but one number on the diagonal is not a $1$ entry. 

{% katexmm %}
$$
\begin{bmatrix} 1 & 0 \\ 0 & 3 \\\end{bmatrix}
$$
{% endkatexmm %}

This matrix keeps all dimensions unchanged except the one with a non-$1$ valued entry, which creates a scaling effect (like stretching or squishing) along only one axis.

For example, the matrix above keeps one dimension unchanged but scales the other dimension by 3.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image15.png" width="40%">
</div>

The "Off by One" matrix also works predictably in higher dimensions:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image16.png" width="40%">
</div>


Reflection Matrix
---
The reflection matrix is similar to the identity matrix in form, except that the diagonal entries are $-1$ instead of $1$. 

{% katexmm %}
$$
\begin{bmatrix} -1 & 0 \\ 0 & -1 \\\end{bmatrix}
$$
{% endkatexmm %}

Having a negative sign in our diagonal entry causes the input region to be flipped for the corresponding axis. Having all diagonal entries be $-1$ in the reflection matrix thus causes the input region to be flipped along all its axes.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image17.png" width="70%">
</div>


Diagonal Matrix
---
The diagonal matrix builds upon the identity and "off by one" matrix above, but now we remove some constraints, such that the diagonal entries of our square matrix can be any value. 

{% katexmm %}
$$
\begin{bmatrix} a & 0 \\ 0 & b \\\end{bmatrix}
$$
{% endkatexmm %}

In the diagonal matrix, we scale each axis according to their corresponding diagonal entries. For example, in the matrix above, we scale one axis by $a$, and the other by $b$.

We can also decompose our diagonal matrix into constituent matrices by viewing matrix multiplication as a composition of sequential transformations, as seen below:

{% katexmm %}
$$
\begin{bmatrix} -2.5 & 0 \\ 0 & 0.6 \\\end{bmatrix} = \underbrace{\begin{bmatrix} -1 & 0 \\ 0 & 1 \\\end{bmatrix}}_{\text{A}} \underbrace{\begin{bmatrix} 1 & 0 \\ 0 & 0.6 \\\end{bmatrix}}_{\text{B}} \underbrace{\begin{bmatrix} 2.5 & 0 \\ 0 & 1 \\\end{bmatrix}}_{\text{C}}
\\
\\
$$
{% endkatexmm %}

{% katexmm %}
$$
\text{A: reflect X about Y axis}
\\
\\
\text{B: scale Y by 0.6}
\\
\\
\text{C: scale X by 2.5}
$$
{% endkatexmm %}


Zero Matrix
---
The zero matrix simply makes every input vector the zero vector, by bringing all points to origin.

{% katexmm %}
$$
\begin{bmatrix} 0 & 0 \\ 0 & 0 \\\end{bmatrix}
$$
{% endkatexmm %}


Shear Matrix
---
The shear matrix is a square matrix that applies a shearing transformation. We can visualize a (horizontal) shearing transformation as horizontal layers of an input square being laterally shifted relative to each other.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image18.png" width="60%">
</div>

{% katexmm %}
$$
\begin{bmatrix} 1 & 1 \\ 0 & 1 \\\end{bmatrix} \begin{bmatrix} x \\ y \\\end{bmatrix} = \begin{bmatrix} x+y \\ y \\\end{bmatrix}
$$
{% endkatexmm %}

We can see that every X coordinate is shifted right by Y, while Y coordinates stay the same, creating the shearing effect.

Here are some other essential shear matrices:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image19.png" width="60%">
</div>


Orthogonal Matrix
---
An orthogonal matrix is a square matrix where all column vectors are unit vectors, and all column vectors are orthogonal. 

For example, 

{% katexmm %}
$$
\begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} \\[0.3em] \frac{-\sqrt{2}}{2} & \frac{\sqrt{2}}{2}\\\end{bmatrix}
$$
{% endkatexmm %}

To elaborate, this means that each column in the matrix forms a vector with magnitude $\lvert 1 \rvert$ , and that the dot product between any two column vectors produces a value of $0$ .

Orthogonal matrices are significant because they produce 'pure rotation operations', whereby an input region is rotated without any change in total area or shape.

We can see that the example matrix above actually produces a pure $45^{\circ}$ rotation clockwise:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image20.png" width="60%">
</div>

It turns out that finding the correct direction to rotate is the 'gateway to unlock all complexity with linear transformations'. This makes sense, as we have been working only with simple scaling, flipping and shearing matrices so far.

Inductively, we can see that a $3$x$3$ orthogonal matrix produces rotations in $3$ dimensions. Let's use this knowledge to emphasize the perspective that matrix multiplication is a composition of sequential transformations. 

For example, given the following two matrices with the following rotation effects:

{% katexmm %}
$$ 
\begin{bmatrix} 1 & 0 & 0 \\[0.3em]
0 & \frac{1}{2} & \frac{-\sqrt{3}}{2} \\[0.3em]
0 & \frac{\sqrt{3}}{2} & \frac{1}{2} \end{bmatrix}
\text{A: Rotates 3D region by 60 degrees about X-axis}
$$
{% endkatexmm %}

{% katexmm %}
$$
\begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{-\sqrt{2}}{2} & 0 \\[0.3em] 
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} & 0 \\[0.3em]
0 & 0 & 1 \end{bmatrix}
\text{B: Rotates 3D region by 45 degrees about Z-axis}
$$
{% endkatexmm %}

We can find the matrix multiplication product $C$ = $BA$, where:

{% katexmm %}
$$
\underbrace{\begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{-\sqrt{2}}{4} & \frac{\sqrt{6}}{4} \\[0.3em]
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{4} & \frac{-\sqrt{6}}{4} \\[0.3em]
0 & \frac{\sqrt{3}}{2} & \frac{1}{2} \end{bmatrix}}_{\text{C}}
= 
\underbrace{\begin{bmatrix} \frac{\sqrt{2}}{2} & \frac{-\sqrt{2}}{2} & 0 \\[0.3em] 
\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} & 0 \\[0.3em]
0 & 0 & 1 \end{bmatrix}}_{\text{B}}
\times
\underbrace{\begin{bmatrix} 1 & 0 & 0 \\[0.3em]
0 & \frac{1}{2} & \frac{-\sqrt{3}}{2} \\[0.3em]
0 & \frac{\sqrt{3}}{2} & \frac{1}{2} \end{bmatrix}}_{\text{A}}
$$
{% endkatexmm %}

We actually see that applying transformation matrix $A$ to an input 3D region first, followed by matrix $B$, actually produces the same transformation as matrix $C$ produces itself.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image21.png" width="100%">
</div>

This emphasizes that matrix $C$ is a sequential composition of its two constituent matrices. However, we need to also emphasize the importance of the 'sequential' term - The order in which matrices $A$ and $B$ are applied matters, as $AB$ produces a different net rotation than $BA$.

Recall that matrix multiplication is not commutative - which means that $BA$ is not always = $AB$.


Projection Matrix
---
A projection matrix  is one that moves every vector to its closest point on a defined subspace. So, the projection matrix moves every single vector outside of a subspace into it.

Let's define a subspace first. We will use examples to understand what it is, since the formal definition is rather long. 

In 2D space, an infinitely long line crossing through the origin is a subspace of $R^2$.

In 3D space, an infinitely large plane passing through origin is a subspace of $R^3$.

For each subspace, there exists a projection matrix that can move every vector outside the subspace into it. 

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image22.png" width="90%">
</div>

In effect, a projection matrix compresses an input into a lower dimension by projecting the vectors to their closest point on the subspace.


Inverse of a Matrix
---
An inverse matrix allows us to "reverse" the effects of a particular transformation matrix. For example, a transformation matrix that scales has an inverse matrix that exactly "unscales", and a transformation matrix that rotates has an inverse matrix that exactly "unrotates".

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image23.png" width="100%">
</div>


We notice that after applying the inverse to a transformed matrix, we get back the original input region, just like we went through transformation with the identity matrix. That's why, when we apply a transformation, then apply its inverse, we get back the identity matrix:

{% katexmm %}
$$ 
\underbrace{A^{-1}}_{\text{Inverse transformation}} \overbrace{A}^{\text{Some transformation}} = \underbrace{I}_{\text{Identity matrix}}
$$
{% endkatexmm %}


However, some matrices cannot be inverted. The "Zero Matrix" and "Projection Matrix" cannot be un-transformed. This makes sense because by going through the transformation, you have lost information about the original input.



References
---
Please refer to Visual Kernel's excellent video on these essential transformation matrices <a href="https://youtu.be/wciU07gPqUE?si=sH8jLyVQXhHmP2li" target="_blank">here</a>. I found this topic helpful in crystallizing my geometric understanding of matrices as linear transformations.