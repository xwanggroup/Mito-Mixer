# esm_embedder.py
"""
ESM2Embedder (Transformers backend) with:
- Plan A: disable Xet/CAS + longer timeouts
- Local freezing (save_pretrained) to ./models/...
- Auto-fix for partial/corrupted downloads:
    - if consistency check fails, retry with force_download=True
"""

# ===================== Plan A: network robustness =====================
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"          # disable Xet/CAS route
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # avoid hf_transfer path
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"  # seconds
os.environ["HF_HUB_READ_TIMEOUT"] = "600"      # seconds
# Optional:
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# =====================================================================

import hashlib
from pathlib import Path
from typing import Optional, Tuple
import shutil
import torch


def _safe_rmtree(p: Path):
    try:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass


def _cleanup_incomplete_safetensors(local_model_dir: Path):
    """
    If a previous download was interrupted, we may have partial files.
    Common locations:
      - local_model_dir/model.safetensors (if already partially saved)
      - local_model_dir/models--facebook--... (HF cache layout)
      - temp *.incomplete files under cache
    We won't try to be too clever: we just remove any existing model.safetensors
    in the freeze directory to allow force_download to re-fetch cleanly.
    """
    bad_files = [
        local_model_dir / "model.safetensors",
        local_model_dir / "pytorch_model.bin",
        local_model_dir / "model.safetensors.index.json",
    ]
    for f in bad_files:
        if f.exists():
            try:
                f.unlink()
            except Exception:
                pass

    # Also remove possible partially created "snapshots" in local_model_dir
    # (only if they exist and look like HF cache)
    # This is safe because we will re-download.
    for child in local_model_dir.glob("models--*"):
        _safe_rmtree(child)
    for child in local_model_dir.glob("downloads"):
        _safe_rmtree(child)
    for child in local_model_dir.glob("tmp"):
        _safe_rmtree(child)


class ESM2Embedder:
    """
    Transformers-only embedder.

    Behavior:
    - If local_model_dir already contains a frozen HF model (config.json exists):
        load tokenizer/model from local_model_dir with local_files_only=True
    - Else:
        download from HF (Plan A env applied)
        if partial/corrupted files detected -> cleanup + force_download=True retry
        then save_pretrained() to local_model_dir
        then reload offline from local_model_dir
    """

    def __init__(
        self,
        model_name: str,
        backend: str = "transformers",
        cache_dir: str = "cache/embeddings",
        layer: int = -1,
        max_len: int = 4096,
        device: Optional[torch.device] = None,
        use_fp16: bool = True,
        # ---- local model control ----
        local_model_dir: str = "models/esm2",
        ensure_local: bool = True,
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.layer = layer
        self.max_len = max_len
        self.device = device or torch.device("cpu")
        self.use_fp16 = use_fp16 and (self.device.type == "cuda")

        self.local_model_dir = Path(local_model_dir)
        self.local_model_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_local = ensure_local
        self.local_files_only_flag = local_files_only

        if backend != "transformers":
            raise ValueError("This version supports only backend='transformers'.")

        from transformers import AutoTokenizer, AutoModel

        # Case 1: local frozen model exists
        if (self.local_model_dir / "config.json").exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.local_model_dir),
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                str(self.local_model_dir),
                local_files_only=True
            )

        else:
            # Case 2: no local model -> download once
            if self.local_files_only_flag:
                raise FileNotFoundError(
                    f"local_files_only=True but local model not found at: {self.local_model_dir}"
                )
            if not self.ensure_local:
                raise FileNotFoundError(
                    f"Local model not found at {self.local_model_dir} and ensure_local=False."
                )

            print(f"[ESM2] Downloading model {model_name} and freezing to:")
            print(f"       {self.local_model_dir}")

            # First attempt (normal)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=str(self.local_model_dir),
                    local_files_only=False,
                )
                self.model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=str(self.local_model_dir),
                    local_files_only=False,
                )
            except OSError as e:
                # Handle partial downloads / consistency check failures
                msg = str(e).lower()
                if "consistency check failed" in msg or "file should be of size" in msg:
                    print("[ESM2] Detected incomplete/corrupted download. Cleaning and retrying with force_download=True ...")
                    _cleanup_incomplete_safetensors(self.local_model_dir)

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=str(self.local_model_dir),
                        local_files_only=False,
                        force_download=True,
                    )
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=str(self.local_model_dir),
                        local_files_only=False,
                        force_download=True,
                    )
                else:
                    raise

            # Freeze to local dir (so later runs can be offline)
            self.tokenizer.save_pretrained(str(self.local_model_dir))
            self.model.save_pretrained(str(self.local_model_dir))

            # Reload offline to ensure integrity
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.local_model_dir),
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                str(self.local_model_dir),
                local_files_only=True
            )

        self.model.to(self.device)
        self.model.eval()

    # ------------------- residue embedding caching -------------------
    def _seq_hash(self, seq: str) -> str:
        return hashlib.md5(seq.encode("utf-8")).hexdigest()

    def _cache_path(self, seq: str) -> Path:
        h = self._seq_hash(seq)
        return self.cache_dir / f"{self.model_name.replace('/', '_')}__{h}.pt"

    @torch.no_grad()
    def embed_residue(self, seq: str) -> torch.Tensor:
        seq = seq.strip().upper()
        if len(seq) == 0:
            raise ValueError("Empty sequence.")
        if len(seq) > self.max_len:
            raise ValueError(
                f"Sequence length {len(seq)} exceeds max_len {self.max_len}. No truncation is applied."
            )

        cp = self._cache_path(seq)
        if cp.exists():
            return torch.load(cp, map_location="cpu")

        inputs = self.tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        autocast = torch.cuda.amp.autocast if self.use_fp16 else torch.cpu.amp.autocast
        with autocast(enabled=self.use_fp16):
            out = self.model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[self.layer]            # [1, T, D]
            X = hs[0, 1:-1, :].detach().float().cpu()     # [L, D]

        torch.save(X, cp)
        return X


# =================== Mito-Mixer input builders ===================

def build_mito_input(X: torch.Tensor, head_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X_input = X_head + broadcast(V_global) -> [head_len, D]
    """
    L, D = X.shape
    V_global = X.mean(dim=0)  # [D]

    X_head = X[:min(L, head_len), :]
    if L < head_len:
        pad = torch.zeros(head_len - L, D, dtype=X.dtype)
        X_head = torch.cat([X_head, pad], dim=0)
    else:
        X_head = X_head[:head_len, :]

    X_input = X_head + V_global.unsqueeze(0).expand(head_len, D)
    return X_input, V_global


from typing import Tuple
import torch

def build_mito_input_concat(X: torch.Tensor, head_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Feature-dim concat version (matches: [X_head ; V_global * 1^T] in feature/channel axis)

    Input:
      X: [L, D]
    Output:
      X_input: [head_len, 2D]     (i.e., N x 2D)
      V_global: [D]
    """
    L, D = X.shape
    V_global = X.mean(dim=0)  # [D]

    # head slice + pad to N=head_len
    X_head = X[:min(L, head_len), :]
    if L < head_len:
        pad = torch.zeros(head_len - L, D, dtype=X.dtype)
        X_head = torch.cat([X_head, pad], dim=0)
    else:
        X_head = X_head[:head_len, :]

    # replicate global to N positions: [N, D]
    V_rep = V_global.unsqueeze(0).expand(head_len, D)

    # concat on feature/channel dim -> [N, 2D]
    X_input = torch.cat([X_head, V_rep], dim=1)
    return X_input, V_global

