from typing import Optional
from attrs import frozen

@frozen 
class OpenLMConfig:
    complete_train_token_count: int
    model: str
    workers: int
    global_batch_size: int
    log_every_n_steps: int
    grad_clip_norm: float
    lr: float
    warmup: int
    wd: float
    beta2: float
    epochs: int
    report_to: str
    data_key: str
    lr_cooldown_end: float
    z_loss_coefficient: float
    accum_freq: int
    model_norm: str
    val_frequency: int
    global_val_batch_size: int
    val_num_samples: int
    qk_norm: bool
    fsdp: bool
    fsdp_amp: bool
    grad_checkpointing: bool
    seed: Optional[int] = None