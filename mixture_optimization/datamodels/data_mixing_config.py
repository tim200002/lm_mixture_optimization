from attrs import define
from typing import Optional

@define
class DataMixingConfig:
    no_workers: int
    chunk_size: int
    oversample_factor: float
    shard_size: int
    shard_selection_multiplier: Optional[float] = None
    seed: Optional[int] = None