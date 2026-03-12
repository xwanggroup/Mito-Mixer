
"""
Mito-Mixer (FASTA) FULL training (NO K-fold, NO HPO),
using PRECOMPUTED ESM2 residue embeddings (NO ESM forward during training).

Key features:
1) 去掉理化特征 P_emb（不再计算、不再拼接）
2) 训练集训练（不使用验证集）
3) 保存 best checkpoint（按 train loss 最小保存，避免用测试集选 best）
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from config import TrainConfig
from label_map import get_label_map
from dataset_fasta import ProteinFasta, collate_batch_fasta
from esm_embedder import build_mito_input, build_mito_input_concat
from model import MitoMixerClassifier
from losses import LabelSmoothingCrossEntropy, compute_balanced_class_weights
from metrics import (
    confusion_matrix_from_logits,
    accuracy_from_confmat,
    per_class_mcc_from_confmat,
    gcc_baldi_from_confmat,
)
from utils import set_seed, get_device, save_checkpoint

# ================== 用户配置区（无需命令行） ==================
TRAIN_FASTA_PATH = "data/SubMitoPred.fasta"      # 训练集（相对路径）
EMB_DIR = "embeddings/esm2-SubMitoPred"                # 预计算 embedding 目录（相对路径）

STRICT_FASTA = True
USE_CONCAT_GLOBAL = True

PRIMARY_METRIC = "GCC"
# ===============================================================


def _seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


class EmbeddingStore:
    """
    Each file:
      embeddings/.../{md5(seq)}.pt -> torch.Tensor [L, D] on CPU
    """
    def __init__(self, emb_dir: str):
        self.emb_dir = Path(emb_dir)

    def load(self, seq: str) -> torch.Tensor:
        h = _seq_hash(seq)
        p = self.emb_dir / f"{h}.pt"
        if not p.exists():
            raise FileNotFoundError(
                f"Embedding not found: {p}\n"
                f"Please run precompute_esm_embeddings.py first, and ensure EMB_DIR is correct."
            )
        return torch.load(p, map_location="cpu")  # [L, D]


def forward_model(model, x_tokens: torch.Tensor) -> torch.Tensor:
    try:
        return model(x_tokens)
    except TypeError:
        return model(x_tokens, None)


def rebuild_classifier_no_physchem(
    model: nn.Module, in_dim: int, cls_hidden: int, cls_dropout: float, num_classes: int
):
    """
    如果 model.classifier 的输入维度不是 in_dim（比如还带着 physchem_out=64），就强制重建。
    注意：这里仍使用 BatchNorm1d，因此训练 DataLoader 用 drop_last=True 避免 batch=1 报错。
    """
    if not hasattr(model, "classifier"):
        return

    first_linear = None
    for m in model.classifier.modules():
        if isinstance(m, nn.Linear):
            first_linear = m
            break
    if first_linear is None:
        return

    old_in = int(first_linear.in_features)
    if old_in == int(in_dim):
        return

    model.classifier = nn.Sequential(
        nn.Linear(in_dim, cls_hidden),
        nn.BatchNorm1d(cls_hidden),
        nn.GELU(),
        nn.Dropout(cls_dropout),
        nn.Linear(cls_hidden, num_classes),
    )
    print(f"[Patch] Rebuilt classifier input dim: {old_in} -> {in_dim} (physchem removed)")


def main():
    cfg = TrainConfig()
    cfg.ensure_dirs()

    set_seed(cfg.seed)
    device = get_device()

    label_map = get_label_map()
    cfg.num_classes = 4
    inv_label = {v: k for k, v in label_map.items()}

    root = Path(__file__).resolve().parent
    train_fasta_path = (root / TRAIN_FASTA_PATH).resolve()
    emb_dir = (root / EMB_DIR).resolve()

    if not train_fasta_path.exists():
        raise FileNotFoundError(f"TRAIN FASTA not found: {train_fasta_path}")
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embedding dir not found: {emb_dir}")

    out_dir = (root / cfg.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # save label map
    pd.DataFrame([{"label_str": k, "label_id": v} for k, v in label_map.items()]) \
        .sort_values("label_id") \
        .to_csv(out_dir / "label_map.csv", index=False)

    # dataset
    train_ds = ProteinFasta(str(train_fasta_path), label_map=label_map, strict=STRICT_FASTA)
    train_labels = np.array([train_ds[i]["label"] for i in range(len(train_ds))], dtype=int)
    train_counts = np.bincount(train_labels, minlength=cfg.num_classes)

    print("Train FASTA:", train_fasta_path)
    print("Train sequences:", len(train_ds))
    print("Train class counts:")
    for cid in range(cfg.num_classes):
        print(f"  {cid} ({inv_label[cid]}): {int(train_counts[cid])}")

    # embedding store
    store = EmbeddingStore(str(emb_dir))
    X0 = store.load(train_ds[0]["sequence"])  # [L, D]
    # print("Example ESM embedding shape:", tuple(X0.shape))  # <- 你要的维度打印
    D = int(X0.shape[1])

    Lm = int(cfg.head_len)
    D_in = (2 * D) if USE_CONCAT_GLOBAL else D

    print(f"Loaded embeddings: base D={D}, D_in={D_in}, token length L={Lm}, concat_global={USE_CONCAT_GLOBAL}")
    print(f"Primary metric: {PRIMARY_METRIC} (Baldi GCC)")

    def make_tokens(seqs):
        xs = []
        for s in seqs:
            X = store.load(s)  # [L, D]
            if USE_CONCAT_GLOBAL:
                X_in, _ = build_mito_input_concat(X, cfg.head_len)  # [N, 2D]
            else:
                X_in, _ = build_mito_input(X, cfg.head_len)         # [N, D]
            xs.append(X_in)
        return torch.stack(xs, dim=0)  # [B, N, D_in]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch_fasta,
        drop_last=True,  # 防 BatchNorm batch=1
    )

    model = MitoMixerClassifier(
        L=Lm, D=D_in, num_classes=cfg.num_classes,
        mixer_blocks=cfg.mixer_blocks,
        token_mlp_ratio=cfg.token_mlp_ratio,
        channel_mlp_ratio=cfg.channel_mlp_ratio,
        mixer_dropout=cfg.dropout,
        cls_hidden=cfg.cls_hidden,
        cls_dropout=cfg.cls_dropout,
    ).to(device)

    rebuild_classifier_no_physchem(
        model, in_dim=D_in, cls_hidden=cfg.cls_hidden, cls_dropout=cfg.cls_dropout, num_classes=cfg.num_classes
    )

    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if getattr(cfg, "class_weight", "balanced") == "balanced":
        w = compute_balanced_class_weights(train_labels.tolist(), cfg.num_classes, device=device)
    else:
        w = None

    criterion = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing, weight=w)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_fp16 and device.type == "cuda"))

    best_train_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    ckpt_path = out_dir / "best_fulltrain.pt"

    print("\n========== Full-Train Training ==========")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_C = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
        nb = 0

        for batch in train_loader:
            seqs = batch["sequences"]
            y = batch["labels"].to(device)
            x_tokens = make_tokens(seqs).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_fp16 and device.type == "cuda")):
                logits = forward_model(model, x_tokens)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if getattr(cfg, "grad_clip", 0.0) and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(opt)
            scaler.update()

            tr_loss += loss.item()
            tr_C += confusion_matrix_from_logits(logits, y, cfg.num_classes)
            nb += 1

        tr_loss /= max(nb, 1)
        tr_acc = accuracy_from_confmat(tr_C)
        tr_gcc = gcc_baldi_from_confmat(tr_C)
        tr_mcc_each = per_class_mcc_from_confmat(tr_C)
        tr_mcc_macro = float(np.mean(tr_mcc_each))

        mcc_each_str = " ".join([f"{v:+.4f}" for v in tr_mcc_each.tolist()])
        print(
            f"[Epoch {epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc:.4f} gcc={tr_gcc:.4f} "
            f"mcc_macro={tr_mcc_macro:.4f} mcc_each=[{mcc_each_str}]"
        )

        if tr_loss < best_train_loss - 1e-6:
            best_train_loss = float(tr_loss)
            best_epoch = int(epoch)
            bad_epochs = 0

            save_checkpoint(str(ckpt_path), {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "best_train_loss": float(best_train_loss),
                "label_map": label_map,
                "config": cfg.__dict__,
                "D": int(D),
                "D_in": int(D_in),
                "Lm": int(Lm),
                "concat_global": bool(USE_CONCAT_GLOBAL),
                "train_fasta_path": str(train_fasta_path),
                "emb_dir": str(emb_dir),
            })
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"  -> Early stop on TRAIN loss. Best train loss={best_train_loss:.4f} (epoch={best_epoch})")
                break

    print(f"\n[Train Done] Saved best checkpoint: {ckpt_path}")
    print(f"best_epoch={best_epoch}, best_train_loss={best_train_loss:.4f}")


if __name__ == "__main__":
    main()
