# Schedule-Free Learning
[![Downloads](https://static.pepy.tech/badge/schedulefree)](https://pepy.tech/project/schedulefree) [![Downloads](https://static.pepy.tech/badge/schedulefree/month)](https://pepy.tech/project/schedulefree)

Schedule-Free Optimizers in PyTorch.

Preprint: [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)

Authors: Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky

**TLDR** Faster training without schedules - no need to specify the stopping time/steps in advance!

``` pip install schedulefree ```

Primary implementations are `SGDScheduleFree` and `AdamWScheduleFree`. We also have `AdamWScheduleFreeReference` and `SGDScheduleFreeReference` versions which have a simplified implementation, but which use more memory. To combine with other optimizers, use the experimental ScheduleFreeWrapper version.

A [Jax implementation](https://optax.readthedocs.io/en/latest/api/contrib.html#schedule-free) is availiable as part of Optax.

## Approach
Schedule-Free learning replaces the momentum of an underlying optimizer with a combination of interpolation and averaging. In the case of gradient descent, the basic Schedule-Free update is:

$$
\begin{align*}
y_{t} & = (1-\beta)z_{t} + \beta x_{t},\\
z_{t+1} & =z_{t}-\gamma\nabla f(y_{t}),\\
x_{t+1} & =\left(1-\frac{1}{t+1}\right)x_{t}+\frac{1}{t+1}z_{t+1},
\end{align*}
$$

Here $x$ is the sequence that evaluations of test/val loss should occur at, which differs from the primary iterates $z$ and the gradient evaluation locations $y$. The updates to $z$ correspond to the underlying optimizer, in this case a simple gradient step.

As the name suggests, Schedule-Free learning does not require a decreasing learning rate schedule, yet typically out-performs, or at worst matches, SOTA schedules such as cosine-decay and linear decay. Only two sequences need to be stored at a time (the third can be computed from the other two on the fly) so this method has the same memory requirements as the base optimizer (parameter buffer + momentum).

We provide both AdamW and SGD versions in this repo, as we as an experimental
wrapper version that can be used with any base optimizer.

## How to Use
Since our optimizer uses two different points for gradient calls and test/val loss calculations, it's necessary to switch the param buffer between the two during training. This is done by calling `optimizer.train()` at the same place you call `model.train()` and `optimizer.eval()` at the same place you call `model.eval()`. The optimizer should also be placed in eval mode when storing checkpoints.

If your code supports PyTorch Optimizer step closures, you can use the closure forms of the optimizers, which do not require the `.train()` and `.eval()` calls.

## Paper
If you use Schedule-Free training in your work, please cite our [preprint](https://arxiv.org/abs/2405.15682) as:
```
@misc{defazio2024road,
      title={The Road Less Scheduled}, 
      author={Aaron Defazio and Xingyu Yang and Harsh Mehta and Konstantin Mishchenko and Ahmed Khaled and Ashok Cutkosky},
      year={2024},
      eprint={2405.15682},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Releases

*New* Version 1.3 changes the behavior of weight decay during learning rate warmup
to improve stabiliy and be more consistant with the behavior of standard AdamW in PyTorch. The previous implementation is still available as `AdamWScheduleFreePaper`.

### Examples
Examples of using the `schedulefree` package can be found in the `examples` folder. These include:
- [Image classification (MNIST) using Convnets](./examples/mnist/)*
- More examples to be added

*Example is modified from [Pytorch Examples Repo](https://github.com/pytorch/examples).


## Caveats 
- If your model uses BatchNorm, additional modifications are required for test/val evaluations to work correctly. Right before eval, something like the following:
  
 ```python
  model.train()
  optimizer.eval()
  with torch.no_grad():
    for batch in itertools.islice(train_loader, 50):
      model(batch)
  model.eval()
```
This will replace the `training_mean`/`training_var` cache (which is updated in each forward pass when in model.train() mode) with values calculated at $x$ instead of $y$. Using PreciseBN will also avoid this issue.


 - Many code bases use additional features that may not be compatible without additional changes. For instance, if the parameters are cached in fp16, the cached versions will need to be updated manually to ensure the correct $x$ sequence is used for evaluation, not the $y$ sequence. Some GradScalers do this.
 - Training is more sensitive to the choice of $\beta$ than you may expect from standard momentum. Our default of $0.9$ works on most problems but it may be necessary to increase the value to $0.95$ or $0.98$ particularly for very long training runs.
 - There is no need to use a learning rate scheduler, however the code is compatible with one.
 - Using learning rate warmup is recommended. This is supported through the `warmup_steps` parameter.
 - This method does require tuning - it won't necessarily out-perform a schedule approach without also tuning regularization and learning rate parameters.
 - For SGD, a learning rate 10x-50x larger than classical rates seems to be a good starting point.
 - For AdamW, learning rates in the range 1x-10x larger than with schedule-based approaches seem to work.

 # Wrapper Version

We offer a highly experimental wrapper version `ScheduleFreeWrapper` which can wrap any base optimizer. When using this version, you can disable the base optimizer's 
 momentum, as it's no longer necessary when using our wrapper's momentum (although you can use both types of momentum if you want).

 Example usage:
 ```
  base_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0025)
  optimizer = ScheduleFreeWrapper(
    base_optimizer, momentum=0.9, weight_decay_at_y=0.1)
 ```
 If you set weight decay on the base optimizer, it computes weight decay at $z$. We offer the option to compute weight decay at $y$, via the `weight_decay_at_y`
 parameter, which seems to give better results in our experiments.

We also include a ScheduleFreeWrapperReference version which uses more memory but is more numerically stable, we recommended this version for early experimentation or research work. 

# License
See the [License file](/LICENSE).

# Related Work

Schedule-Free learning can be seen as an interpolation between primal averaging ($\beta=1$) and Polyak-Ruppert averaging ($\beta=0)$. The advantage of this interpolation is that it allows us to get the best of both worlds. We can achieve the fast early stage convergence of Polyak-Ruppert averaging (since the $z$ sequence moves quicker than the $x$ sequence), without the $x$ sequence straying too far from the $z$ sequence, which causes instability.

Our method is also related to Nesterov's accelerated method (Nesterov, 1983) in AC-SA form (Ghadimi & Lan 2010):

$$
\begin{align*}
y_{t} & =(1-2/(t+1))x_{t} + (2/(t+1))z_{t}\\
z_{t+1} & =z_{t}-\frac{t}{2L}\nabla f(y_{t})\\
x_{t+1} & =(1-2/(t+1))x_{t}+(2/(t+1))z_{t+1}
\end{align*}
$$

Our approach has the same three sequences, but uses very different weights, and crucially, does not include an increasing learning rate over time, which is essential for accelerated rates with Nesterov's method. We also use different weight sequences for the interpolation operation versus the averaging operation.

Tail averaging approaches such as Stochastic Weight Averaging (Izmailov et al., 2018) and LAtest Weight Averaging (Kaddour, 2022; Sanyal et al., 2023) combine averaging with large or cyclic learning rates. They still require the use of a schedule, introduce additional hyper-parameters to tune, and require additional memory compared to our technique. It is also possible to use SWA and LAWA on top of our approach, potentially giving further gains.

Portes et al. (2022) use cyclic learning rate schedules with increasing cycle periods to give a method that explores multiple points along the Pareto frontier of training time vs eval performance. Each point at the end of a cycle is an approximation to the model from a tuned schedule ending at that time. Our method gives the entire frontier, rather than just a few points along the path.

Exponential moving averages (EMA) of the iterate sequence are used in the popular Lookahead optimizer (Zhang et al., 2019). The Lookahead method can be seen as the EMA version of primal averaging, just as exponential weight averaging is the EMA version of Polyak-Ruppert averaging. Our extra interpolation step can potentially be used in combination with the lookahead optimizer also.
