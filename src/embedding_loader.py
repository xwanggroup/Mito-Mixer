# embedding_loader.py
import hashlib
from pathlib import Path
import torch

def seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()

class EmbeddingStore:
    def __init__(self, emb_dir: str):
        self.emb_dir = Path(emb_dir)

    def load(self, seq: str) -> torch.Tensor:
        h = seq_hash(seq)
        path = self.emb_dir / f"{h}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Embedding not found: {path}")
        return torch.load(path, map_location="cpu")  # [L, D]
