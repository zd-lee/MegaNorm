"""Learning rate scheduler utilities."""

import torch.optim.lr_scheduler as lr_scheduler

# class StaticLR()


def create_scheduler(optimizer, training_config, total_epochs):
    """
    Create learning rate scheduler based on config.

    Config format:
        training:
          lr_scheduler:
            type: 'cosine'  # or 'step', 'plateau'
            eta_min: 1e-6   # for cosine
            step_size: 30   # for step
            gamma: 0.1      # for step
            patience: 10    # for plateau
            factor: 0.5     # for plateau
          warmup_epochs: 5

    Args:
        optimizer: PyTorch optimizer
        training_config: training section of config dict
        total_epochs: total number of training epochs

    Returns:
        scheduler: LR scheduler instance, or None if not configured
    """
    scheduler_config = training_config.get('lr_scheduler', None)

    if scheduler_config is None:
        return None

    # Get scheduler type from nested dict
    scheduler_type = scheduler_config.get('type', None)

    if scheduler_type is None:
        return None

    warmup_epochs = training_config.get('warmup_epochs', 0)

    if scheduler_type == 'cosine':
        eta_min = float(scheduler_config.get('eta_min', 1e-6))
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=eta_min
        )

    elif scheduler_type == 'static':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=1e6,
            gamma=1
        )

    elif scheduler_type == 'step':
        step_size = int(scheduler_config.get('step_size', 30))
        gamma = float(scheduler_config.get('gamma', 0.1))
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )

    elif scheduler_type == 'plateau':
        mode = scheduler_config.get('mode', 'min')
        patience = int(scheduler_config.get('patience', 10))
        factor = float(scheduler_config.get('factor', 0.5))
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            patience=patience,
            factor=factor
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Supported: cosine, step, plateau")
