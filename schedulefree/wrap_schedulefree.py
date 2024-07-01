import math
import torch.optim
from collections import defaultdict
#defaultdict(dict)

class ScheduleFreeWrapper:
    r"""
        Wrap any optimizer to make it Schedule-Free.

        When using this version, you can disable the base optimizer's 
        momentum, as it's no longer necessary when using our wrapper's 
        momentum (although you can use both types of momentum if you want).

        Example usage:
        ```
        base_optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0025)
        optimizer = ScheduleFreeWrapper(
            base_optimizer, momentum=0.9, weight_decay_at_y=0.1)
        ```

        If you set weight decay on the base optimizer, it computes weight decay at $z$. We offer the option to compute weight decay at $y$, via the `weight_decay_at_y`
        parameter, which seems to give better results in our experiments.

        base (torch.optim.Optimizer): 
            PyTorch optimizer object
        momentum (float): Apply momentum on the outer optimizer (default 0.9)
        weight_decay_at_y (float): 
            Weight decay calculated at the y point. Set weight decay on the 
            inner optimizer to instead calculate at z (default: 0).
        r (float): Use polynomial weighting in the average 
            with power r (default 0).
        weight_lr_power (float): During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting
            (default 2.0).
    """
    def __init__(self, base, 
                 momentum=0.9,
                 weight_decay_at_y=0.0,
                 r=0,
                 weight_lr_power=2):

        self.base = base

        self.momentum = momentum
        self.weight_lr_power = weight_lr_power
        self.r = r
        self.train_mode = False

        # Reallocate param group
        for group in self.param_groups:
            group['weight_decay_at_y'] = weight_decay_at_y
            zs = []
            for p in group['params']:
                z = p.detach().clone()
                zs.append(z)

                self.state[z]['model'] = p

            group['params'] = zs

    def add_param_group(self, param_group):
        return self.base.add_param_group(param_group)

    def load_state_dict(self, state_dict):
        return self.base.load_state_dict(state_dict)

    def state_dict(self):
        return self.base.state_dict()
        
    def zero_grad(self, set_to_none=True):
        return self.base.zero_grad(set_to_none)

    @property
    def param_groups(self):
        return self.base.param_groups

    @property
    def state(self):
        return self.base.state

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                for z in group['params']:
                    state = self.state[z]
                    # Set model params y -> x
                    state['model'].lerp_(end=z, weight=1-1/self.momentum)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for z in group['params']:
                    state = self.state[z]
                    # Set model params x -> y
                    state['model'].lerp_(end=z, weight=1-self.momentum)

        self.train_mode = True

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not self.train_mode:
            raise Exception("Please insert .train() and .eval() calls for the "
                            "optimizer. See documentation for details")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            weight_decay_at_y = group['weight_decay_at_y']
            for z in group['params']:
                model = self.state[z]['model']

                # Give the inner optimizer access to the grads calculated at y
                z.grad = model.grad

                if weight_decay_at_y != 0.0:
                    z.sub_(model, alpha=lr*weight_decay_at_y)
                    model.sub_(model, alpha=lr*weight_decay_at_y*(1-self.momentum))
                    
                # Convert model y -> x
                model.lerp_(end=z, weight=1-1/self.momentum)

        #######
        self.base.step()

        ######
        for group in self.param_groups:
            weight_lr_power = self.weight_lr_power
            r = self.r
            k = group.get('k', 0)
            lr = group['lr']
            lr_max = group['lr_max'] = max(lr, group.get('lr_max', 0))
            
            weight = ((k+1)**r) * (lr_max**weight_lr_power)
            weight_sum = group['weight_sum'] = group.get('weight_sum', 0.0) + weight

            ckp1 = weight/weight_sum

            for z in group['params']:
                model = self.state[z]['model']

                z.grad = None

                # Update x then convert x -> y using a fused operation
                model.lerp_(end=z, weight=1-self.momentum*(1-ckp1))

            group['k'] = k+1

        return loss