"""
Custom learning rate schedulers for iterative training.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupStableDecayScheduler(_LRScheduler):
    """
    A learning rate scheduler with three phases:
    1. Warmup: Linear increase from 0 to target LR
    2. Stable: Constant LR at target value
    3. Decay: Cosine annealing decay to 0
    The three phases repeat continuously for the duration of training.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of steps for warmup phase
        stable_steps: Number of steps for stable phase
        decay_steps: Number of steps for decay phase
        last_epoch: The index of last epoch (default: -1)
    """
    
    def __init__(self, optimizer, warmup_steps, stable_steps, decay_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.cycle_steps = warmup_steps + stable_steps + decay_steps
        
        # Validation
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if stable_steps < 0:
            raise ValueError(f"stable_steps must be >= 0, got {stable_steps}")
        if decay_steps < 0:
            raise ValueError(f"decay_steps must be >= 0, got {decay_steps}")
        if self.cycle_steps <= 0:
            raise ValueError("At least one of warmup_steps, stable_steps, or decay_steps must be > 0")
        
        print(f"WarmupStableDecayScheduler initialized:")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Stable steps: {stable_steps}")
        print(f"  Decay steps: {self.decay_steps}")
        print(f"  Cycle steps: {self.cycle_steps}")
        
        super(WarmupStableDecayScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate based on current step."""
        # Map the global step into its position within the current cycle
        cycle_position = (self.last_epoch % self.cycle_steps) + 1
        
        lrs = []
        for base_lr in self.base_lrs:
            if cycle_position <= self.warmup_steps:
                # Warmup phase: linear increase from 0 to target LR
                if self.warmup_steps > 0:
                    lr = base_lr * cycle_position / self.warmup_steps
                else:
                    lr = base_lr
            elif cycle_position <= self.warmup_steps + self.stable_steps:
                # Stable phase: constant LR
                lr = base_lr
            else:
                # Decay phase: cosine annealing over the remainder of the cycle
                decay_step = cycle_position - self.warmup_steps - self.stable_steps
                if self.decay_steps > 0:
                    # Cosine annealing from target LR to 0
                    lr = 0.5 * base_lr * (1 + math.cos(math.pi * decay_step / self.decay_steps))
                else:
                    lr = base_lr
            
            lrs.append(lr)
        
        return lrs
    
    def get_current_phase(self):
        """Return the current phase name for debugging."""
        cycle_position = (self.last_epoch % self.cycle_steps) + 1
        
        if cycle_position <= self.warmup_steps:
            return "warmup"
        elif cycle_position <= self.warmup_steps + self.stable_steps:
            return "stable"
        else:
            return "decay"
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, "
                f"stable_steps={self.stable_steps}, "
                f"decay_steps={self.decay_steps})")