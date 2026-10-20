---
tags:
  - tinygpt
  - machine-learning
  - python
  - backpropagation
  - gradient-descent
created: 2026-10-18
status: completed
---

# Training with Multiple Examples

> [!summary] Key idea
> Mean squared error averages squared prediction errors, while the gradient averages signed error contributions across examples.

$$
L=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

$$
\frac{\partial L}{\partial w}=\frac{2}{N}\sum_{i=1}^{N}(\hat y_i-y_i)x_i
$$

$$
\frac{\partial L}{\partial b}=\frac{2}{N}\sum_{i=1}^{N}(\hat y_i-y_i)
$$

Notebook: [05-training-multiple-examples.ipynb](../notebooks/05-training-multiple-examples.ipynb).

## Original worked exercises

## Exercise 1

Assume our model has:
$$
\begin{aligned}
\hat{y}&=wx+b \\

w&=1,\quad b=0
\end{aligned}
$$
And our dataset is:

| $x$ | Target $y$ |
| :-: | :--------: |
|  1  |     2      |
|  2  |     4      |

Calculate the following by hand:

1. What is the model's prediction $\hat{y}$ for each input?
    $$
    \begin{aligned}
    let \quad x &= 1 \\
    \hat{y} &= 1\cdot1+0\\
    &= 1\\
    let \quad x &= 2 \\
    \hat{y} &= 1\cdot2+0\\
    &= 2\\
    \end{aligned}
    $$
2. What is the squared error for each prediction?
    $[ L_i=(\hat y_i-y_i)^2 ]$

$$
\begin{aligned}
L1 &= (2 - 1)^2\\
&= 1\\
L2 &= (4 - 2)^2\\
&= 4\\
\end{aligned}
$$
3. What is the Mean Squared Error across both examples?

$$
\begin{aligned}
\boxed{\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2} \\\\
MSE=\frac{1}{2}\cdot5 = \boxed{2.5}
\end{aligned}
$$

## Exercise 2: Gradients with Multiple Examples
Our model is:

$\hat y=wx+b$

Our loss function is:

$L=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2$

$L=(wx+b-y)^2$

$$
\begin{aligned}
u&=wx+b-y \\
L&=(wx+b-y)^2 \\ 
&= u^2 \\

\frac{dL}{du} &= 2u \\
\frac{\partial{u}}{\partial{w}} &= x \\\\

\frac{\partial{L}}{\partial{w}} &= \frac{dL}{du} \cdot \frac{\partial{u}}{\partial{w}} \\
&= 2u \cdot x \\
&= 2(wx+b-y)\cdot x

\end{aligned}
$$
with initial values:
$w=1,\qquad b=0$

and dataset:

| $x$ | Target $y$ |
| :-: | :--------: |
|  1  |     2      |
|  2  |     4      |

I want you to find three values:

1. $\frac{\partial L_1}{\partial w}$
    $$ \begin{aligned} 
    \frac{\partial L_1}{\partial w} &= 2(1 \cdot 1+0-2)\cdot 1 \\
    &= -2
    \end{aligned}$$
2. $\frac{\partial L_2}{\partial w}$
    $$ \begin{aligned} 
    \frac{\partial L_2}{\partial w} &= 2(1 \cdot 2+0-4)\cdot 2 \\
    &= -8
    \end{aligned}$$
3. $\frac{\partial L}{\partial w}$ (the average gradient)
$$\begin{aligned} 
\frac{\partial L}{\partial w} &=\frac12\left( \frac{\partial L_1}{\partial w} +\frac{\partial L_2}{\partial w} \right) \\
&= \frac{1}{2}(-2 -8) \\
&= 1/2 \cdot -10 \\ 
&= \boxed{-5}
\end{aligned}$$


Your task: Calculate these three values by hand.

1. $\frac{\partial L_1}{\partial b}$
   $$ \begin{aligned} 
    \frac{\partial L_1}{\partial b} &= 2(1 \cdot 1+0-2) \\
    &= -2
    \end{aligned}$$
    
2. $\frac{\partial L_2}{\partial b}$
   $$ \begin{aligned} 
    \frac{\partial L_2}{\partial b} &= 2(1 \cdot 2+0-4) \\
    &= -4
    \end{aligned}$$
    
3. $\frac{\partial L}{\partial b}$ — the average of the first two gradients.
$$\begin{aligned} 
\frac{\partial L}{\partial b} &=\frac12\left( \frac{\partial L_1}{\partial b} +\frac{\partial L_2}{\partial b} \right) \\
&= \frac{1}{2}(-2 -4) \\
&= 1/2 \cdot -6 \\ 
&= \boxed{-3}
\end{aligned}$$

## Exercise 3: One Complete Gradient Descent Update
|Parameter|Value|
|---|---|
|Weight \(w\)|1|
|Bias \(b\)|0|
|MSE|2.5|
|Weight gradient \(\frac{\partial L}{\partial w}\)|-5|
|Bias gradient \(\frac{\partial L}{\partial b}\)|-3|
Part A — Update the parameters

Calculate the new values of \(w\) and \(b\) using the gradients above.
$$
\begin{aligned}
w_{new} &= w - \eta \cdot \frac{\partial{L}}{\partial{w}} \\
&= 1 - 0.1 \cdot -5 \\
&= \boxed{1.5}
\\\\
b_{new} &= b - \eta \cdot \frac{\partial{L}}{\partial{b}} \\
&= 0 - 0.1 \cdot -3 \\
&= \boxed{0.3}
\end{aligned}
$$

Part B — Evaluate the updated model

Using your new parameters, calculate:

1. The prediction \(\hat y\) for \(x=1\).
    
2. The prediction \(\hat y\) for \(x=2\).
    
3. The new MSE across both examples.

| $x$ | Target $y$ |
| :-: | :--------: |
|  1  |     2      |
|  2  |     4      |


$$
\begin{aligned}
\hat{y}&=wx+b \\

w&=1.5,\quad b=0.3 \\\\
let \quad x &= 1 \\
\hat{y}&=1.5\cdot1+0.3 \\
&= 1.8

\\\\

let \quad x &= 2 \\
\hat{y}&=1.5\cdot2+0.3 \\
&= 3.3
\end{aligned}
$$

$$
\begin{aligned}
L1 &= (2 - 1.8)^2\\
&= 0.04\\
L2 &= (4 - 3.3)^2\\
&= 0.49\\

MSE &= \frac{1}{2} \cdot (0.04 + 0.49) \\
&= \frac{1}{2} \cdot 0.53 \\
&= 0.265
\end{aligned}
$$
## Exercise 4: Increasing the Learning Rate

Suppose we start again with:

$w=1,\quad b=0$

And the same gradients:

$\frac{\partial L}{\partial w}=-5$

$\frac{\partial L}{\partial b}=-3$

But now we use:

$\boxed{\eta=0.3}$

### Your task

1. Calculate the new $w$ and $b$.
$$
\begin{aligned}
w_{new} &= w - \eta \cdot \frac{\partial{l}}{\partial{w}} \\
&= 1-0.3\cdot-5 \\
&= 1 + 1.5 \\
&= \boxed{2.5}
\\\\

b_{new} &= b - \eta \cdot \frac{\partial{l}}{\partial{b}} \\
&= 0-0.3\cdot-3 \\
&= 0 + 0.9 \\
&= \boxed{0.9}
\end{aligned}
$$
2. Calculate the predictions for $x=1$ and $x=2$.
$$
\begin{aligned}
\hat{y}&=wx+b \\

w&=2.5,\quad b=0.9 \\\\
let \quad x &= 1 \\
\hat{y}&=2.5\cdot1+0.9 \\
&= 3.4

\\\\

let \quad x &= 2 \\
\hat{y}&=2.5\cdot2+0.9 \\
&= 5.9
\end{aligned}
$$
2. Calculate the new MSE.
$$
\begin{aligned}
L_1 &= (2 - 3.4)^2 \\
&= (-1.4)^2 \\
&= \boxed{1.96} \\
L_2 &= (4 - 5.9)^2 \\
&= (-1.9)^2 \\
&= \boxed{3.61}
\\\\
MSE &= \frac{1}{2} \cdot (1.96 + 3.61) \\
&= \frac{1}{2} \cdot 5.57 \\
&= \boxed{2.785}
\end{aligned}
$$

### Conceptual Question

Is it possible for a gradient descent update to increase the loss beyond its original value of $2.5$, even though we're moving in the direction of the negative gradient?
> Yes, if we overshoot far enough. (Didn't need to really do the manual calculations to come to this conclusion but alas)
