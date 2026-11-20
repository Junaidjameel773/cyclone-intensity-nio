r"""
model.py - the network, built up in four small pieces.

Read it top to bottom; each class is one box in your architecture diagram.

    images (B,4,1,128,128)
            |
        [1] CNN  -> a 8x8 grid of 256-dim features per frame
            |            \
        [2] LSTM          \  (all 4x8x8 = 256 patch tokens)
            |  summary h   \
        [3] Cross-attention: "given what the LSTM remembers, which patches
            |                 of which frames matter?"  -> context vector
            |                                            + attention map
    tab -> [4] MLP ------> concat -> regression head -> wind speed (kt)
"""

import torch
import torch.nn as nn
import config


# ------------------------------------------------------------------ [1] CNN
class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, halving the image each time."""
    def __init__(self, cin, cout, stride=2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.f(x)


class SpatialEncoder(nn.Module):
    """128x128 -> 8x8 feature map with `width` channels."""
    def __init__(self, width=256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32),          # 64
            ConvBlock(32, 64),         # 32
            ConvBlock(64, 128),        # 16
            ConvBlock(128, width),     # 8
        )
        self.out_dim = width

    def forward(self, x):              # (N,1,128,128)
        return self.net(x)             # (N,C,8,8)


# ------------------------------------------------------- [3] cross-attention
class CrossAttention(nn.Module):
    """
    Query  = what the LSTM remembers about the storm's evolution.
    Keys   = every patch of every frame.
    Output = a weighted summary + the weights themselves (your attention map).
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, tokens):                 # (B,1,D), (B,N,D)
        out, weights = self.attn(query, tokens, tokens,
                                 need_weights=True, average_attn_weights=True)
        return self.norm(out + query).squeeze(1), weights.squeeze(1)   # (B,D),(B,N)


# ------------------------------------------------------------- full model
class MultiModalCycloneModel(nn.Module):
    def __init__(self, n_tab=len(config.TAB_COLS), width=256, hidden=256,
                 heads=4, dropout=0.2, use_images=True, use_tab=True):
        super().__init__()
        # Ablation switches. The architecture and parameter count stay
        # identical; a disabled branch simply contributes zeros, so any
        # difference in score is due to the INFORMATION, not model size.
        self.use_images, self.use_tab = use_images, use_tab
        self.cnn = SpatialEncoder(width)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # [2] temporal module
        self.lstm = nn.LSTM(width, hidden, num_layers=1, batch_first=True)

        # tokens and query must live in the same space for attention
        self.token_proj = nn.Linear(width, hidden)
        self.cross = CrossAttention(hidden, heads)

        # [4] tabular branch
        self.tab_mlp = nn.Sequential(
            nn.Linear(n_tab, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden * 2 + 64, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, 1),
        )
        # start by predicting a typical storm (~60 kt) instead of 0
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, 60.0)

    def forward(self, images, tab, return_attention=False):
        B, T = images.shape[:2]

        fmap = self.cnn(images.flatten(0, 1))              # (B*T,C,8,8)
        C, H, W = fmap.shape[1:]

        per_frame = self.pool(fmap).flatten(1).view(B, T, C)      # (B,T,C)
        seq, (h_n, _) = self.lstm(per_frame)
        query = h_n[-1].unsqueeze(1)                              # (B,1,hidden)

        tokens = fmap.flatten(2).transpose(1, 2).reshape(B, T * H * W, C)
        tokens = self.token_proj(tokens)                          # (B,T*64,hidden)

        context, attn = self.cross(query, tokens)                 # (B,hidden),(B,T*64)

        h = query.squeeze(1)
        tab_emb = self.tab_mlp(tab)
        if not self.use_images:
            context, h = torch.zeros_like(context), torch.zeros_like(h)
        if not self.use_tab:
            tab_emb = torch.zeros_like(tab_emb)

        fused = torch.cat([context, h, tab_emb], dim=1)
        wind = self.head(fused).squeeze(1)

        if return_attention:
            return wind, attn.view(B, T, H, W)
        return wind


# ------------------------------------------------------------------ loss
class WeightedMSE(nn.Module):
    """
    Plain MSE treats a 10 kt error on a tropical depression the same as a
    10 kt error on a Category 4. It is not the same. This multiplies the
    squared error by a weight that ramps up smoothly around Cat 3.
    """
    def __init__(self, threshold=config.CAT3_KT, high_weight=config.HIGH_WEIGHT,
                 softness=10.0):
        super().__init__()
        self.t, self.w, self.s = threshold, high_weight, softness

    def forward(self, pred, true):
        ramp = torch.sigmoid((true - self.t) / self.s)
        weight = 1.0 + (self.w - 1.0) * ramp
        return (weight * (pred - true) ** 2).mean()


if __name__ == "__main__":
    m = MultiModalCycloneModel()
    x = torch.randn(2, config.SEQ_LEN, 1, config.IMG_SIZE, config.IMG_SIZE)
    t = torch.randn(2, len(config.TAB_COLS))
    y, a = m(x, t, return_attention=True)
    print("params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    print("wind:", y.shape, "attention:", a.shape)
