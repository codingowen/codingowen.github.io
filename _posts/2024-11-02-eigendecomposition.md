---
title: An Intuition of Eigenvalue (Spectral) Decomposition
subtitle: I attempt to construct a geometric understanding of how eigendecomposition works.
shorttitle: Eigenvalue (Spectral) Decomposition
layout: default
date: 2024-11-02
keywords: blogging, writing
published: true
---

Eigenvalue Decomposition, also called Spectral Decomposition, is a mathematical technique used in linear algebra to express a symmetric matrix as a product of three components:

{% katexmm %} 
$$ 
\fcolorbox{orange}{white}{$A = QΛQ^T$} \\
~ \\
\text{A is the original symmetric matrix} \\
\text{Q is a matrix whose columns are the eigenvectors of A} \\
\text{Λ is a diagonal matrix whose entries are the eigenvalues of A} \\
~ \\
$$ 
{% endkatexmm %}

Eigenvalue Decomposition is extremely important because it simplifies linear transformations. It is commonly used in transforming high-dimensional data into a lower dimensional space while retaining its most significant patterns, enabling much more efficient storage and computation. 

In this blog post, I will attempt to construct an informal geometric intuition for how Eigenvalue Decomposition works, and why it is so important in decomposing complicated matrices. 

We'll begin this blog post by answering some fundamental questions, and gradually build up to Eigenvalue Decomposition.


1\. What is a Symmetric Matrix?
---
Symmetric matrices have entries that are symmetric about the diagonal line:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image42.png" width="40%">
</div>

Only square matrices can be symmetrical, any other rectangular matrices are not.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image43.png" width="25%">
</div>


2\. Special Properties of the Matrix Transpose
---
Transpose is an action you perform on a matrix, where you make the rows of the matrix the columns, and vice versa.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image44.png" width="40%">
</div>

When you transpose a rectangular matrix, notice that the dimensionality of the matrix changes:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image45.png" width="40%">
</div>

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

The concept of the matrix transpose is crucial to Eigendecomposition because of its special properties relating to symmetric and orthogonal matrices. Next, we'll learn about these special transpose properties:

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

<b><u>1. When you transpose a symmetric matrix, you get the exact same matrix back</u></b>

Recall the definition and characteristics of a symmetric matrix. If we visualize the matrix transpose happening, we can see why the transposed symmetric matrix is identical to the original matrix:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image46.png" width="40%">
</div>

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

<b><u>2. When you transpose an orthogonal matrix, you get its inverse</u></b>

As it turns out, when you transpose an orthogonal matrix, the transposed matrix is equal to the inverse of the orthogonal matrix:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image47.png" width="55%">
</div>

Here's an explanation of why this property is so special, laid out step-by-step:

1. Recall that an orthogonal matrix produces a 'pure rotation' transformation

2. The inverse of a matrix reverses the transformation by the original matrix, so it 'untransforms'

3. The 'untransformation' of a rotation means to rotate in the reverse direction

4. The transpose of an orthogonal matrix is equal to its inverse matrix (established in the above section)

5. Therefore, the transpose of an orthogonal matrix produces a 'reverse rotation' transformation, or, a rotation in the reverse direction.


So, for example, if an orthogonal matrix $Q$ produces a $45°$ clockwise rotation, its transpose $Q^T$ is equal to its matrix $Q^{-1}$, which produces a $45°$ anti-clockwise rotation:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image48.png" width="60%">
</div>


3\. What is Matrix Decomposition?
---
Let's begin by recalling what Matrix Composition is. When we do matrix multiplication of let's say 3 matrices, we're essentially <b>composing</b> their distinct transformations together into one net/resultant transformation.

For example: 

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image49.png" width="70%">
</div>

So, Matrix Decomposition is essentially going in the reverse direction to Matrix Composition. Suppose you have the transformation of a matrix, doing matrix decomposition means you re-express that as a sequence of much simpler transformations. 

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

$\text{Composition: Start with multiplication of some matrices → Produce resultant matrix}$

$\text{Decomposition: Start with one matrix → Produce sequence of simpler matrix multiplication}$

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

We can imagine that matrix decomposition is much more difficult than matrix composition - if you were given a random matrix, how would you find a correct sequence of constituent matrices that multiply together to form it?

Hopefully, we will be able to show that spectral decomposition can make this process easier.


4\. Symmetric Matrices and Orthogonal Eigenvectors
---
A crucial property of symmetric matrices is that their eigenvectors are orthogonal. 

Recall that we have a standard basis, $\hat{i}$ and $\hat{j}$ which are orthogonal. 

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image50.png" width="25%">
</div>

If we apply the property of symmetric matrices having orthogonal eigenvectors, we can observe that there exists a Symmetric Matrix $Q$ which produces a 'pure rotation transformation', which happens to rotate the standard basis $\hat{i}$ and $\hat{j}$ to align with its eigenvectors.

In the same case, there would exist an inverse matrix $Q^{-1}$ which rotates the eigenvectors to align with the standard basis instead.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image51.png" width="70%">
</div>

(Note: by 'rotation', I really mean 'transformation from one basis to another', eg. transformation from the standard basis into the eigenvector basis. But for this informal blog post, I will keep things simple and use the term 'rotation'.)

5\. Spectral Decomposition
---
Now, we're ready to understand spectral decomposition. 

The Spectral Decomposition Theorem states that for any real $n \times n$ symmetric matrix $A$ can be decomposed into a sequence of three simple matrices, $QΛQ^T$, whereby:

{% katexmm %} 
$$ 
\fcolorbox{orange}{white}{$A = QΛQ^T$} \\
~ \\
$$ 
{% endkatexmm %}

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image52.png" width="70%">
</div>

{% katexmm %} 
$$ 
\text{A is the original symmetric matrix} \\
\text{Q is an orthogonal matrix whose columns are the eigenvectors of A} \\
\text{Λ is a diagonal matrix whose entries are the eigenvalues of A} \\
~ \\
$$ 
{% endkatexmm %}

<b><u>Let's visualize the transformation effect of each decomposed matrix product.</u></b>

Starting with Lambda, $Λ$, it is a diagonal matrix whose entries are the eigenvalues of A. Assuming our original matrix $A$ is $2 \times 2$, our $Λ$ matrix has diagonal entries $\lambda_1$ and $\lambda_2$. So, the $Λ$ matrix scales the $x$ axis by $\lambda_1$, and scales the $y$ axis by $\lambda_2$.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image53.png" width="40%">
</div>

Next, for $Q$, it is an orthogonal matrix, so we know it does a 'pure rotation' transformation. Because its entries are the eigenvectors of A, $Q$ rotates the standard basis to align with the respective eigenvectors.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image54.png" width="49%">
</div>

Of course, for $Q^T$, whereby $Q^T = Q^{-1}$, it does the opposite of $Q$, and rotates the eigenvectors to align with the standard basis.

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

<b><u>Let's go through an example to see why this decomposition makes sense!</u></b>

Let's say we have a $2 \times 2$ symmetric matrix $A$, which can be decomposed using the spectral decomposition theorem, like so:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image55.png" width="55%">
</div>


{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}


Next, let's say we know that matrix $A$ produces the following transformation effect:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image56.png" width="70%">
</div>

Since we know matrix $A$ can be decomposed, we know that this resultant transformation is the composition of three simpler transformation effects. Let's visualize this:

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

Firstly, using matrix $Q^T$, we identify the eigenvectors of $A$, and rotate them onto the standard basis:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image57.png" width="70%">
</div>

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

Next, using the matrix $Λ$, we scale the x-axis by 6, and the y-axis by 2:

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image58.png" width="75%">
</div>

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

And finally, using the matrix $Q$, we rotate the standard basis to match the original location of the eigenvectors.

<div class="centered-image">
    <img class="post-image" src="/assets/blog_images/image59.png" width="75%">
</div>

{% katexmm %} $$ ~ \\ $$ {% endkatexmm %}

Why does this make so much intuitive sense? 

Recall one of the definitions of eigenvectors - they are vectors that do not get knocked off their span during a linear transformation, such that the effect of linear transformation is just to scale the eigenvectors. This scaling factor is also known as the eigenvalue. 

You may also question, why do we even "rotate" the eigenvector basis to match the standard basis? If we were to remain in the standard basis, it would be very difficult to understand the linear transformation action of the matrix, because it may have complicated motions (eg. rotating and scaling simultaneously). By transforming into the eigenvector basis, the linear transfomation action is constrained to only scaling along each principal axis. Not only does this allow us to understand the transformation clearly, it also makes many computations easier.

Spectral Decomposition is a powerful (and beautiful) tool, but it suffers from poor generalizability - how often do we even get to work with a symmetric matrix? You'll need a square matrix that happens to be symmetric about its diagonal line. 

To make spectral decomposition more versatile, we'll need to learn about Singular Value Decomposition. See you in the next post!


References
---
This blog post was entirely based on Visual Kernel's <a href="https://youtu.be/mhy-ZKSARxI?si=QZ36fP7WF47oDUPS" target="_blank">excellent video on Spectral Decomposition!</a>