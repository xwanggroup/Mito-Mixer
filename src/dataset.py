# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class ProteinCSV(Dataset):
    """
    CSV must contain:
      - sequence column (string, amino acids)
      - label column (int 0..C-1)
    """
    def __init__(self, csv_path: str, seq_col: str, label_col: str):
        self.df = pd.read_csv(csv_path)
        if seq_col not in self.df.columns or label_col not in self.df.columns:
            raise ValueError(f"CSV must contain columns: {seq_col}, {label_col}")
        self.seq_col = seq_col
        self.label_col = label_col

        # basic cleanup
        self.df[seq_col] = self.df[seq_col].astype(str).str.strip()
        self.df[label_col] = self.df[label_col].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "sequence": row[self.seq_col],
            "label": int(row[self.label_col]),
        }

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    seqs = [b["sequence"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"sequences": seqs, "labels": labels}
