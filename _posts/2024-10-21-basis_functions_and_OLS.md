---
title: Basis Functions in Linear Regression Models, with an OLS example
subtitle: I give an overview of how basis functions are used in linear regression models. Then, I derive the solution to an OLS problem with basis functions involved.
shorttitle: Basis Functions
layout: default
date: 2024-10-22
keywords: blogging, writing
published: true
---

In the field of supervised learning, one of the most fundamental linear models is the Linear Regression model. 

Let's begin with a quick recap on how to mathematically set up a linear regression model:

Basic Linear Regression
---

<b><u>Step 1: Define hypothesis space H</u></b>
{% katexmm %}
$$
H = {f: f(x) = w_0 + w_1 x}
$$
{% endkatexmm %}

Whereby $w_0$ and $w_1$ are parameters of the model, and $w_0 , w_1 \space \epsilon \space R$

The goal is to find the optimal values of $w_0$ and $w_1$ that best fit the data.


<b><u>Step 2: Define the loss function and ERM</u></b>

Firstly, we need to define a loss function:

{% katexmm %}
$$
L(y',y) = \frac{1}{2} (y' - y)^2
$$
{% endkatexmm %}


Then, the empirical risk minimization problem becomes:

{% katexmm %}
$$
\underset{f \space \epsilon \space H}\text{Min} \space R_{\text{emp}}(f) = \underset{w_0, w_1}\text{Min} \space \frac{1}{2N} \displaystyle\sum_{i=1}^N (\underbrace{w_0 + w_1x_i}_{y'_i} - y_i)^2
$$
{% endkatexmm %}


<b><u>Step 3: Solve for the best approximation</u></b>

To minimize loss, we find the derivative of the empirical loss function, and recognize that loss is minimum at point where derivative value = 0. 

So, we need to find the derivative of the loss function with respect to $w_0$ and $w_1$.


General Linear Basis Models
---
The simple linear regression we have seen is quite limited - it is only suitable for 1-dimensional inputs, and can only fit linear relationships. Now, we'll see how linear basis models help us generalize the simple linear regression approach.

Consider an input $x$ that lies in a higher-dimensional space. We can apply a feature map $\phi (x)$ to transform this input into a new space where we use $\phi (x)$ to model the problem. This transformation allows us to handle non-linear relationships from the original input space while maintaining a linear relationship between the transformed features and target $y$. 

<b><u>Let's use a simple example to illustrate what this all means:</u></b>

Let's say the relationship between your $x$ and $y$ variables resembles something like $y = x^2$. whereby the relationship between $x$ and $y$ is non-linear. So, the $x$ here is considered as being in a 'space' where it is not straightforward to model $y$ linearly.

So, we use a feature map $\phi (x) = x^2$ to transform $x$ into a new feature space, where $\phi (x)$ and $y$ have a linear relationship. 

In this example,
1. Original relationship: $y = x^2$ 
2. Feature map: $\phi (x)$ = $x^2$ 
3. Transformed relationship: y = $\phi (x)$

In the transformed space $\phi (x)$, the relationship is now linear because $y$ is directly proportional to $\phi (x)$.

From a higher-level perspective, we can see that basis functions allow us to work with non-linear $x$ inputs while having linear parameters that are easy to compute. 

<b><u>Let's mathematically formulate the hypothesis space for the feature map</u></b>

Considering $x \space \epsilon \space R^d$, 

{% katexmm %}
$$
H_M = {f: f(x) = \displaystyle\sum_{j=0}^{M-1} w_j \phi_j (x)}
$$
{% endkatexmm %}

Whereby each $\phi_j : R^d \rightarrow R$ is called a basis function or a feature map. Our feature map can also generate multiple features, depending on the value of $M$. 

For example, our feature map $\phi (x)$ could generate $\phi_1 (x) = x$, $\phi_2 (x) = x^2$, and more. 


Applying Linear Basis Models to Ordinary Least Squares
---
Now, let's see how the linear basis models can be used to solve an OLS problem. 

Remember, due to the feature mapping, our $x$ input is modified to be $\phi_j (x_i)$, so the input to our ordinary least squares equation becomes $w_j \phi_j (x_i)$, where loss = $w_j \phi_j (x_i) - y_i$

{% katexmm %}
$$
\begin{aligned}
\underset{f \space \epsilon \space H_M}\text{Min} \space R_{\text{emp}}(f) &= \underset{w \space \epsilon \space R^M}\text{Min} \space R_{\text{emp}}(w) \\
&= \underset{w \space \epsilon \space R^M}\text{Min} \frac{1}{2N} \displaystyle\sum_{i=1}^{N} (f(x_i)-y_i)^2 \\
&= \underset{w \space \epsilon \space R^M}\text{Min} \frac{1}{2N} \displaystyle\sum_{i=1}^{N} ( \displaystyle\sum_{j = 0}^{M-1} w_j \phi_j (x_i) - y_i)^2
\end{aligned}
$$
{% endkatexmm %}

We can also rewrite the end result in compact form:

{% katexmm %}
$$
\underset{w \space \epsilon \space R^M}\text{Min} \frac{1}{2N} \displaystyle\sum_{i=1}^{N} ( \displaystyle\sum_{j = 0}^{M-1} w_j \phi_j (x_i) - y_i)^2 = \underset{w \space \epsilon \space R^M}\text{Min} \frac{1}{2N} \Vert \Phi w - y \Vert^2
$$
{% endkatexmm %}

Whereby

{% katexmm %}
$$
\begin{array}{ccc}
\Phi = \begin{pmatrix} \phi_0 (x_1) & \dots & \phi_{M-1}(x_1) \\ \phi_0 (x_2) & \dots & \phi_{M-1}(x_2) \\ \vdots & \ddots & \vdots \\ \phi_0 (x_N) & \dots & \phi_{M-1}(x_N) \end{pmatrix} 
&
w = \begin{pmatrix} w_0 \\ w_1 \\ \vdots \\ w_{M-1} \end{pmatrix}
&
y = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_N \end{pmatrix}
\end{array}
$$
{% endkatexmm %}


<b><u>Now, let's solve for the least squares solution</u></b>

{% katexmm %}
$$
\begin{aligned}
\text{Squared Error} &= (\Phi w - y)^T (\Phi w - y) \\
&= w^T \Phi^T \Phi w - \Phi^T w^T y - y^T \Phi w + y^T y \\
&= y^T y - 2 w^T \Phi^T y + w^T \Phi^T \Phi w
\end{aligned} 
$$
{% endkatexmm %}

{% katexmm %}
$$
\begin{aligned}
\vartriangle \text{Squared Error} &= \frac{d}{dw} (y^Ty - 2 w^T \Phi^T y + w^T \Phi^T \Phi w) \\
&= -2 \Phi^T y + 2 \Phi^T \Phi w
\end{aligned} 
$$
{% endkatexmm %}

Let $\vartriangle \text{Squared Error}$ = 0, then: 

{% katexmm %}
$$
\begin{aligned}
2 \Phi^T \Phi w &= 2 \Phi^T y \\
\Phi^T \Phi w &= \Phi^T y \\
w &= (\Phi^T \Phi)^{-1} \Phi^T y
\end{aligned}
$$
{% endkatexmm %}

Here, we can see how linear basis models can model more complex relationships than simple linear regression models, and we have shown how the least squares solution can be derived relatively simply for the linear basis models too.