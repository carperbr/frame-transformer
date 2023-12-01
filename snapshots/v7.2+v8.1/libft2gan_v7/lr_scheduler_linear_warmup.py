from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import copy

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, target_lr=1e-3, num_steps=16000, current_step=0, verbose_skip_steps=1000):
        self.num_steps = num_steps
        self.starting_step = current_step
        self.current_step = current_step
        self.verbose_skip_steps = verbose_skip_steps

        self.target_lr = copy.deepcopy(optimizer.param_groups)
        self.current_lr = optimizer.param_groups

        super().__init__(optimizer)

    def step(self):
        if self.current_step < self.num_steps+1:
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.current_lr[i]['lr'] = (self.current_step+1) * (self.target_lr[i]['lr'] / (self.num_steps+1))
                param_group['lr'] = self.current_lr[i]['lr']
                if self.current_step % self.verbose_skip_steps == 0:
                    print(' Step {:5d} of {:5d}: increased learning rate'
                                ' of group {} to {:.4e}.'.format(self.current_step, self.num_steps, i, self.current_lr[i]['lr']))
            self.current_step = self.current_step + 1

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

class LinearWarmup:
    def __init__(self, target_lr=1e-3, num_steps=16000, current_step=0, verbose_skip_steps=1000):
        self.num_steps = num_steps
        self.starting_step = current_step
        self.current_step = current_step
        self.verbose_skip_steps = verbose_skip_steps

        self.target_lr = target_lr
        self.current_lr = 0

    def step(self):
        if self.current_step < self.num_steps+1:
            self.current_lr = (self.current_step+1) * (self.target_lr / (self.num_steps+1))

        self.current_step = self.current_step + 1
