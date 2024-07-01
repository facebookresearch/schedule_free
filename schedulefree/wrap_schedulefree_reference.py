import math
import torch.optim
from collections import defaultdict
#defaultdict(dict)

class ScheduleFreeWrapperReference:
    r""" 
        Simplified reference implementation of ScheduleFreeWrapper
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

        for group in self.param_groups:
            group['weight_decay_at_y'] = weight_decay_at_y

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
                for p in group['params']:
                    state = self.state[p]
                    if 'x' in state:
                        # Switch p.data to x
                        p.data.copy_(state['x'])
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                for p in group['params']:
                    state = self.state[p]
                    if 'y' in state:
                        p.data.copy_(state['y'])

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
            for p in group['params']:
                state = self.state[p]

                if 'z' not in state:
                    state['z'] = torch.clone(p.data)
                    state['x'] = torch.clone(p.data)
                    state['y'] = torch.clone(p.data) 

                y = state['y']
                z = state['z']

                # Apply weight decay to z, computed at point y
                if weight_decay_at_y != 0.0:
                    z.sub_(y, alpha=lr*weight_decay_at_y)

                # Copy z into param buffer
                p.copy_(z)

        #######

        # Run base optimizer, acting on params p, which are a copy of z
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

            for p in group['params']:
                state = self.state[p]
                
                y = state['y']
                z = state['z']
                x = state['x']

                # Update z to new value after base optimizer step
                z.copy_(p)

                # Update x
                x.lerp_(end=z, weight=ckp1)

                # Compute y
                y.copy_(x).lerp_(end=z, weight=1-self.momentum)

                # Copy y into param buffer
                p.copy_(y)

            group['k'] = k+1

        return loss