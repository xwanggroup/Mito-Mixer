# model.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MitoMixerBlock(nn.Module):
    """
    X: [B, L, D]
    Token-mixing: MLP over L by transposing to [B, D, L]
    Channel-mixing: MLP over D (shared across tokens)
    """
    def __init__(self, L: int, D: int, token_mlp_ratio: float, channel_mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(D)
        self.norm2 = nn.LayerNorm(D)

        token_hidden = max(1, int(L * token_mlp_ratio))
        channel_hidden = max(1, int(D * channel_mlp_ratio))

        self.token_mlp = MLP(in_dim=L, hidden_dim=token_hidden, out_dim=L, dropout=dropout)
        self.channel_mlp = MLP(in_dim=D, hidden_dim=channel_hidden, out_dim=D, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        y = self.norm1(x)        # [B, L, D]
        y = y.transpose(1, 2)    # [B, D, L]
        y = self.token_mlp(y)    # [B, D, L]
        y = y.transpose(1, 2)    # [B, L, D]
        x = x + y

        # Channel mixing
        z = self.norm2(x)        # [B, L, D]
        z = self.channel_mlp(z)  # [B, L, D]
        x = x + z
        return x

class PhysChemMLP(nn.Module):
    def __init__(self, in_dim: int = 7, hidden: int = 64, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, p):
        return self.net(p)

class MitoMixerClassifier(nn.Module):
    def __init__(
        self,
        L: int,
        D: int,
        num_classes: int,
        mixer_blocks: int = 4,
        token_mlp_ratio: float = 0.5,
        channel_mlp_ratio: float = 2.0,
        mixer_dropout: float = 0.1,
        physchem_dim: int = 7,
        physchem_hidden: int = 64,
        physchem_out: int = 64,
        cls_hidden: int = 512,
        cls_dropout: float = 0.4,
    ):
        super().__init__()
        self.mixer = nn.ModuleList([
            MitoMixerBlock(L=L, D=D, token_mlp_ratio=token_mlp_ratio, channel_mlp_ratio=channel_mlp_ratio, dropout=mixer_dropout)
            for _ in range(mixer_blocks)
        ])

        # self.physchem = PhysChemMLP(in_dim=physchem_dim, hidden=physchem_hidden, out_dim=physchem_out, dropout=mixer_dropout)

        # cls_in = D + physchem_out
        cls_in = D
        self.classifier = nn.Sequential(
            nn.Linear(cls_in, cls_hidden),
            nn.BatchNorm1d(cls_hidden),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, num_classes),
        )

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.mixer:
            x_tokens = blk(x_tokens)

        z_global = x_tokens.mean(dim=1)  # [B, D]
        # p_emb = self.physchem(p7)        # [B, physchem_out]
        # feat = torch.cat([z_global, p_emb], dim=-1)
        logits = self.classifier(z_global)
        return logits
