import torch.optim.lr_scheduler as lr_scheduler


def create_scheduler(optimizer, training_config, total_epochs):
    scheduler_config = training_config.get("lr_scheduler", None)
    if scheduler_config is None:
        return None
    scheduler_type = scheduler_config.get("type", None)
    if scheduler_type is None:
        return None
    warmup_epochs = training_config.get("warmup_epochs", 0)
    if scheduler_type == "cosine":
        eta_min = float(scheduler_config.get("eta_min", 1e-6))
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
        )
    elif scheduler_type == "static":
        return lr_scheduler.StepLR(optimizer, step_size=1e6, gamma=1)
    elif scheduler_type == "step":
        step_size = int(scheduler_config.get("step_size", 30))
        gamma = float(scheduler_config.get("gamma", 0.1))
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "plateau":
        mode = scheduler_config.get("mode", "min")
        patience = int(scheduler_config.get("patience", 10))
        factor = float(scheduler_config.get("factor", 0.5))
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience, factor=factor
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. Supported: cosine, step, plateau"
        )
