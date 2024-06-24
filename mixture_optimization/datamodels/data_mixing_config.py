from attrs import define

@define
class DataMixingConfig:
    no_workers: int
    chunk_size: int
    oversample_factor: float
    shard_size: int