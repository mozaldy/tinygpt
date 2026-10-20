---
tags:
  - tinygpt
  - machine-learning
  - python
  - backpropagation
  - gradient-descent
created: 2026-10-20
status: completed
---

# Mini-batch Gradient Descent

> [!summary] Key idea
> Update parameters after each mini-batch; keep those updated parameters for the next mini-batch. An epoch processes every training example once.

For a batch of $m$ examples, calculate signed errors $\hat y_i-y_i$ for gradients and squared errors for MSE. Evaluate full-dataset loss after each epoch for consistent progress tracking.

Notebook: [05-mini-batch-training.ipynb](../notebooks/05-mini-batch-training.ipynb).

## Original worked exercises

## Exercise 8 — First mini-batch

Assume our model has:
$$
\begin{aligned}
\hat{y}&=wx+b \\

w&=0,\quad b=0
\end{aligned}
$$
With the following dataset:

|\(x\)|\(y\)|
|---|---|
|1|2|
|2|4|
|3|6|
|4|8|
Our starting parameters are:

$w=0,\quad b=0,\quad \eta=0.05$

We use Mean Squared Error:

$L=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2$

The gradients are:

$\frac{\partial L}{\partial w}=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)x_i$

$\frac{\partial L}{\partial b}=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)$

### Your task — Part A

Our batch size is 2, and we are not shuffling the data.

Therefore, the first mini-batch contains:

$(x_1,y_1)=(1,2)$

$(x_2,y_2)=(2,4)$

Calculate:

1. Both predictions using the initial $w$ and $b$.
$$ \begin{aligned}
\hat{y}&=wx+b \\

w&=0,\quad b=0 \\\\
\hat{y}_1 &= 0\cdot1+0\\
&= 0 \\
\hat{y}_2 &= 0\cdot2+0\\
&= 0 \\
\hat{y}_3 &= 0\cdot3+0\\
&= 0 \\
\hat{y}_4 &= 0\cdot4+0\\
&= 0 \\
\end{aligned}$$
    
2. The average loss of this mini-batch.
    $$ \begin{aligned}


L_1 &= (0 - 2)^2 = (-2)^2 = 4\\
L_2 &= (0 - 4)^2 = (-4)^2 = 16\\\\
L&=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2 \\\\
 &= \frac{1}{2} \cdot (4 + 16)\\
&= \frac{1}{2} \cdot20=\boxed{10}

\end{aligned}$$
2. Both gradients, $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$.
    $$ \begin{aligned}
    \frac{\partial L}{\partial w}&=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)x_i  \\
    &= 2/2\cdot(-2\cdot1) + (-4\cdot2) \\
    &= 1\cdot-2-8 \\
    &= \boxed{-10}
    \\\\

	\frac{\partial L}{\partial b}&=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)\\
	&= 2/2\cdot-2 -4 \\
    &= 1\cdot-6 \\
    &= \boxed{-6}
	
	
	\end{aligned}$$

2. The updated parameters $w_{\text{new}}$ and $b_{\text{new}}$, using learning rate $0.05$.
$$\begin{aligned}
w_{new} &= w - \eta\cdot \frac{\partial L}{\partial w} \\
&= 0 - 0.05\cdot-10\\
&= 0 + 0.5\\
&= \boxed{0.5} \\

b_{new} &= b - \eta\cdot \frac{\partial L}{\partial b} \\
&= 0 - 0.05\cdot-6\\
&= 0 + 0.3\\
&= \boxed{0.3} \\
\end{aligned}$$
## Exercise 8 — Part B: The second mini-batch

Now comes the important part of mini-batch gradient descent.

We don't reset the parameters to zero. Instead, we carry the updated parameters from our first mini-batch into the second.

Our model now has:

$w=0.5,\quad b=0.3$

Our second mini-batch contains:

| \(x\) | \(y\) |
| ----- | ----- |
| 3     | 6     |
| 4     | 8     |

Learning rate remains:

$\eta=0.05$

### Your task

Calculate the same four things:

1. Both predictions using \(w=0.5\) and \(b=0.3\).
$$ \begin{aligned}
\hat{y}&=wx+b \\

w&=0.5,\quad b=0.3 \\\\
\hat{y}_3 &= 0.5\cdot3+0.3\\
&= 1.8 \\
\hat{y}_4 &= 0.5\cdot4+0.3\\
&= 2.3 \\
\end{aligned}$$
2. The average loss of this mini-batch.
    $$ \begin{aligned}


L_1 &= (1.8 - 6)^2 = (-4.2)^2 = 17.64\\
L_2 &= (2.3 - 8)^2 = (-5.7)^2 = 32.49\\\\
L&=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2 \\\\
 &= \frac{1}{2} \cdot (17.64+32.49)\\
&= \frac{1}{2} \cdot50.13=\boxed{25.065}

\end{aligned}$$
2. Both gradients with respect to \(w\) and \(b\).
    $$ \begin{aligned}
    \frac{\partial L}{\partial w}&=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)x_i  \\
    &= 2/2\cdot(-4.2\cdot3) + (-5.7\cdot4) \\
    &= 1\cdot-12.6-22.8\\
    &= \boxed{-35.4}
    \\\\

	\frac{\partial L}{\partial b}&=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)\\
	&= 2/2\cdot-4.2 - 5.7\\
    &= 1\cdot-9.9 \\
    &= \boxed{-9.9}
	
	
	\end{aligned}$$
    
3. The new parameters after one gradient descent update.

$$\begin{aligned}
w_{new} &= w-\eta\cdot\frac{\partial{L}}{\partial{w}}\\
&= 0.5-0.05\cdot-35.4\\
&= 0.5+1.77 \\ 
&= 2.27
\\
b_{new} &= b-\eta\cdot\frac{\partial{L}}{\partial{b}}\\
&= 0.3 - 0.05 \cdot-9.9\\
&= 0.3 + 0.495\\
&=  0.795\\
\end{aligned}$$

## Exercise 9 — Python mini-batch implementation

Implemented in [05-mini-batch-training.ipynb](../notebooks/05-mini-batch-training.ipynb): two updates per epoch, ten epochs, and full-dataset MSE evaluated after each epoch.

## Exercise 10 — Final Stage 5 Challenge

This is your final understanding check before we move to Stage 6.

No coding required. Answer these questions using your understanding of gradient descent, batches, and epochs.

### Question 1 — Batch size

Our dataset contains 4 training examples, and we train for 10 epochs.

Complete this table:

| Batch size | Updates per epoch | Total updates |
| ---------- | ----------------- | ------------- |
| 1          | 4                 | 40            |
| 2          | 2                 | 20            |
| 4          | 1                 | 10            |

### Question 2 — Why can the loss increase?

During Exercise 8, our first mini-batch had an MSE of 10, while our second mini-batch had an MSE of 25.065.

Does this necessarily mean that our model became worse after training on the first mini-batch? Explain why or why not.

no, because you can't really compare the mse of the first 2 dataset to the last 2 dataset to see if the models are getting better. You need to compare the first 2 dataset in the first epoch and the first 2 dataset in the second epoch to really see if the models are becoming worse after training.
### Question 3 — Data ordering

Currently, our training loop processes the dataset in this order:

```
X = [1, 2, 3, 4]
Y = [2, 4, 6, 8]
```

Suppose we shuffle the examples, preserving each input-target pair, before every epoch.

Could shuffling change the final weights and bias after 10 epochs, even if the dataset and learning rate remain identical? Why?

yes, because the order will determine the gradient descent

### Question 4 — Full-batch gradient descent

Suppose we change our batch size from 2 to 4.

Our starting parameters are:

$w=0,\quad b=0,\quad\eta=0.05$

Using all four examples, calculate the gradients and perform one parameter update.

What are the new \(w\) and \(b\)?

Calculate:

1. Both predictions using the initial $w$ and $b$.
$$ \begin{aligned}
\hat{y}&=wx+b \\

w&=0,\quad b=0 \\\\
\hat{y}_1 &= 0\cdot1+0\\
&= 0 \\
\hat{y}_2 &= 0\cdot2+0\\
&= 0 \\
\hat{y}_3 &= 0\cdot3+0\\
&= 0 \\
\hat{y}_4 &= 0\cdot4+0\\
&= 0 \\
\end{aligned}$$
    
2. The average loss of this mini-batch.
    $$ \begin{aligned}


L_1 &= (0 - 2)^2 = (-2)^2 = 4\\
L_2 &= (0 - 4)^2 = (-4)^2 = 16\\
L_3 &= (0 - 6)^2 = (-6)^2 = 36\\
L_4 &= (0 - 8)^2 = (-8)^2 = 64\\
L&=\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2 \\\\
&= \frac{1}{4} \cdot (4 + 16 + 36 + 64)\\
&= \frac{1}{4} \cdot120=\boxed{30}

\end{aligned}$$
2. Both gradients, $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$.
    $$ \begin{aligned}
    \frac{\partial L}{\partial w}&=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)x_i  \\
    &= 2/4\cdot((-2\cdot1) + (-4\cdot2) + (-6\cdot3) + (-8\cdot4)) \\
    &= 1/2\cdot(-2-8-18-32) \\
    &= \boxed{-30}
    \\\\

	\frac{\partial L}{\partial b}&=\frac{2}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)\\
	&= 2/4\cdot(-2 -4-6-8) \\
    &= 1/2\cdot-20 \\
    &= \boxed{-10}
	
	
	\end{aligned}$$

3. The updated parameters $w_{\text{new}}$ and $b_{\text{new}}$, using learning rate $0.05$.
$$\begin{aligned}
w_{new} &= w - \eta\cdot \frac{\partial L}{\partial w} \\
&= 0 - 0.05\cdot-30\\
&= 0 + 1.5\\
&= \boxed{1.5} \\

b_{new} &= b - \eta\cdot \frac{\partial L}{\partial b} \\
&= 0 - 0.05\cdot-10\\
&= 0 + 0.5\\
&= \boxed{0.5} \\
\end{aligned}$$


You can answer Questions 1–3 conceptually and calculate Question 4 by hand.

Once you've demonstrated your understanding of these four questions, we'll close Stage 5 and prepare for Stage 6.