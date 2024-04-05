# Schedule-Free Learning - A New Way to Train
Schedule-Free Optimizers in PyTorch.

Authors: Aaron Defazio, Xingyu Yang, Konstantin Mishchenko, Ashok Cutkosky, Harsh Mehta, Ahmed Khaled

**TLDR** Faster training without schedules - no need to specify the stopping time/steps in advance!

## Approach
Schedule-Free learning replaces the momentum of an underlying optimizer with a combination of interpolation and averaging. In the case of gradient descent, the Schedule-free update is:

$$
\begin{align*}
y_{t} & = (1-\beta)z_{t} + \beta x_{t},\\
z_{t+1} & =z_{t}-\gamma\nabla f(y_{t}),\\
x_{t+1} & =\left(1-\frac{1}{t}\right)x_{t}+\frac{1}{t}z_{t+1},
\end{align*}
$$

Here $x$ is the sequence that evaluations of test/val loss should occur at, which differs from the primary iterates $z$ and the gradient evaluation locations $y$. The updates to $z$ correspond to the underlying optimizer, in this case a simple gradient step.

As the name suggests, Schedule-free learning does not require a decreasing learning rate schedule, yet typically out-performs, or at worst matches, SOTA schedules such as cosine-decay and linear decay. Only two sequences need to be stored at a time (the third can be computed from the other two on the fly) so this method has the same memory requires as the base optimizer (parameter buffer + momentum).

We provide both AdamW and SGD versions in this repo.

## How to Use
Since our optimizer uses two different points for gradient calls and test/val loss calculations, it's necessary to switch the param buffer between the two during training. This is done by calling `optimizer.train()` at the same place you call `model.train()` and `optimizer.eval()` at the same place you call `model.eval()`.

If your code supports PyTorch Optimizer step closures, you can use the closure forms of the optimizers, which do not require the `.train()` and `.eval()` calls.

## Caveats 
- If your model uses BatchNorm, additional modifications are required for test/val evaluations to work correctly. Right before eval, something like the following:
  
 ```python
  model.train()
  optimizer.eval()
  for batch in itertools.islice(train_loader, 50):
    _ = self.model(batch)
  model.eval()
```
This will replace the `training_mean`/`training_var` cache (which is updated in each forward pass when in model.train() mode) with values calculated at $x$ instead of $y$. Using PreciseBN will also avoid this isssue.


 - Many code bases use additional features that may not be compatible without additonal changes. For instance, if the parameters are cached in fp16, the cached versions will need to be updated manually to ensure the correct $x$ sequence is used for evaluation, not the $y$ sequence. Some GradScalers do this.
 - Training is more sensitive to the choice of $\beta$ than you may expect from standard momentum. Our default of $0.9$ works on most problems but it may be necessary to increase the value to $0.95$ or $0.98$ particually for very long training runs.
 - There is no need to use a learning rate scheduler, however the code is compatible with one.
 - Using learning rate warmup is recommended. This is supported through the `warmup_steps` parameter.
 - This method does require tuning - it won't necessarily out-perform a schedule approach without also tuning regularization and learning rate parameters.
 - For SGD, a learning rate 10x-50x larger than classical rates seems to be a good starting point.
 - For AdamW, learnings rates in the range 1x-10x larger than with schedule based approaches seem to work.
 - Our method can also be implemented as a wrapper around a base optimizer, where the momentum of the base optimizer is disabled. We didn't do that as PyTorch's Adam implementation would still allocate memory for it's momentum buffer `exp_avg` even if we don't use it.

# License
See the [License file](/LICENSE).

# Related Work

Schedule-Free learning can be seen as an interpolation between primal averaging ($\beta=1$) and Polyak-Ruppert averaging ($\beta=0)$. The advantage of this interpolation is that it allows us to get the best of both worlds. We can achieve the fast early stage convergence of Polyak-Ruppert averaging (since the $z$ sequence moves quicker than the $x$ sequence), without the $x$ sequence straying too far from the $z$ sequence, which causes instability.

Our method is also related to Nesterov's accelerated method (Nesterov, 1983), which can be written in the following form:

$$
\begin{align*}
y_{t} & =(1-2/(t+1))x_{t} + (2/(t+1))z_{t}\\
z_{t+1} & =z_{t}-\frac{k+1}{2L}\nabla f(y_{t})\\
x_{t+1} & =(1-2/(t+1))x_{t}+(2/(t+1))z_{t+1}
\end{align*}
$$

Our approach has the same three sequences, but uses very different weights, and crucially, does not include an increasing learning rate over time, which is essential for accelerated rates with Nesterov's method. We also use different weight sequences for the interpolation operation versus the averaging operation.

Tail averaging approaches such as Stochastic Weight Averaging (Izmailov et al., 2018) and LAtest Weight Averaging (Kaddour, 2022; Sanyal et al., 2023) combine averaging with large or cyclic learning rates. They still require the use of a schedule, introduce additional hyper-parameters to tune, and require additional memory compared to our technique. It is also possible to use SWA and LAWA on top of our approach, potentially giving further gains.

Portes Et. Al. (2022) use cyclic learning rate schedules with increasing cycle periods to give a method that explores multiple points along the Pareto frontier of training time vs eval performance. Each point at the end of a cycle is an approximation to the model from a tuned schedule ending at that time. Our method gives the entire frontier, rather than just a few points along the path.

Exponential moving averages (EMA) of the iterate sequence are used in the popular Lookahead optimizer (Zhang et al., 2019). The Lookahead method can be seen as the EMA version of primal averaging, just as exponential weight averaging is the EMA version of Polyak-Ruppert averaging. Our extra interpolation step can potentially be used in combination with the lookahead optimizer also.
