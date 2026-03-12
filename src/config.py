# config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    # data / labels
    num_classes: int = 4

    # esm / embedding
    esm_backend: str = "transformers"   # "transformers" or "fair-esm"
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D"
    esm_layer: int = -1
    #head_len: int = 80
    head_len: int = 170
    max_len: int = 4096
    use_fp16: bool = True
    cache_dir: str = "cache/embeddings"

    # mixer
    mixer_blocks: int = 2
    token_mlp_ratio: float = 4.36910951366005
    channel_mlp_ratio: float = 4.2463551442789
    dropout: float = 0.344773936377058

    # physchem
    physchem_dim: int = 7
    physchem_mlp_hidden: int = 64

    # classifier
    cls_hidden: int = 448
    cls_dropout: float = 0.353859337696465

    # training
    seed: int = 43
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 0.0000630894422932425
    weight_decay: float = 0.000090343171152051
    epochs: int = 50
    grad_clip: float = 1.0
    early_stop_patience: int = 20

    # loss
    label_smoothing: float = 0.0855144600436398
    class_weight: str = "balanced"  # "balanced" or "none"

    # output
    out_dir: str = "runs/Subhpo"

    def ensure_dirs(self) -> None:
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.out_dir) / "folds").mkdir(parents=True, exist_ok=True)
