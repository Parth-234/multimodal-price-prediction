class ResidualBlock(nn.Module):
    """Pre-activation residual block: LN → Linear → GELU → Dropout → Linear → + skip"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.ln2  = nn.LayerNorm(dim)
        self.fc2  = nn.Linear(dim, dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        r = x
        x = self.ln1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.ln2(x)
        x = self.fc2(x)
        return x + r


class DeepResidualMLP(nn.Module):
    """
    14-layer Deep Residual MLP
    Input → Projection(1) → ResBlock×5(10) → Bottleneck(1) → Head(1) → 13 + LN = ~14
    """
    def __init__(self, input_dim, hidden_dim=512, n_blocks=5, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        bot_dim = hidden_dim // 2
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bot_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(bot_dim),
            nn.Linear(bot_dim, 1),
        )
        nn.init.xavier_uniform_(self.proj[0].weight)
        nn.init.xavier_uniform_(self.bottleneck[1].weight)
        nn.init.xavier_uniform_(self.head[1].weight)

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.bottleneck(x)
        return self.head(x).squeeze(-1)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error — official competition metric."""
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.maximum(np.array(y_pred, dtype=np.float64), 0.01)
    num  = np.abs(y_pred - y_true)
    den  = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(num / (den + 1e-8)) * 100)


print('Model architecture defined.')
print(f'  ResidualBlock + DeepResidualMLP ready.')
print(f'  SMAPE metric function ready.')
