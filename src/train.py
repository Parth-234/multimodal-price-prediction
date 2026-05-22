def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE=='cuda'))

def make_pred_loader(X, batch_size=512):
    ds = TensorDataset(torch.from_numpy(X).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=False)

def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    preds, trues, total = [], [], 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        p = model(Xb)
        total += criterion(p, yb).item() * len(yb)
        preds.append(p.cpu().numpy())
        trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    
    pred_price = inv_target(preds).clip(min=0.01)
    true_price = inv_target(trues)
    val_smape  = smape(true_price, pred_price)
    val_loss   = total / len(loader.dataset)
    return val_loss, val_smape, preds, trues

@torch.no_grad()
def predict(model, X):
    model.eval()
    loader = make_pred_loader(X)
    preds  = []
    for (Xb,) in loader:
        preds.append(model(Xb.to(DEVICE)).cpu().numpy())
    log_preds = np.concatenate(preds)
    return inv_target(log_preds).clip(min=0.01)

print('Training loop functions defined.')

input_dim = X_train.shape[1]
model     = DeepResidualMLP(
    input_dim  = input_dim,
    hidden_dim = HIDDEN_DIM,
    n_blocks   = N_BLOCKS,
    dropout    = DROPOUT,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model ready | input_dim={input_dim} | params={n_params:,} | device={DEVICE}')

criterion    = nn.HuberLoss(delta=1.0)
total_steps  = EPOCHS * (len(X_train) // BATCH_SIZE + 1)
warmup_steps = int(total_steps * WARMUP_RATIO)


decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.requires_grad:
        (no_decay if ('bias' in name or 'ln' in name) else decay).append(p)
optimizer = torch.optim.AdamW(
    [{'params': decay, 'weight_decay': WEIGHT_DECAY},
     {'params': no_decay, 'weight_decay': 0.0}],
    lr=LR, betas=(0.9, 0.999), eps=1e-8,
)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    prog = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + np.cos(np.pi * prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)


best_smape, patience_ctr = float('inf'), 0
history = []

print(f'\nTraining for up to {EPOCHS} epochs (early stop patience={PATIENCE})\n')
print(f'{"Epoch":>6} {"Train Loss":>12} {"Val Loss":>10} {"SMAPE %":>10}')
print('-' * 45)

for epoch in range(1, EPOCHS + 1):
    t_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    v_loss, v_smape, _, _ = evaluate(model, val_loader, criterion)

    history.append({'epoch': epoch, 'train_loss': t_loss,
                    'val_loss': v_loss, 'smape': v_smape})
    print(f'{epoch:>6}  {t_loss:>12.4f}  {v_loss:>10.4f}  {v_smape:>9.2f}%')

    if v_smape < best_smape:
        best_smape   = v_smape
        patience_ctr = 0
        torch.save({'state_dict': model.state_dict(),
                    'input_dim': input_dim,
                    'best_smape': best_smape,
                    'epoch': epoch}, MODEL_PATH)
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f'\nEarly stopping at epoch {epoch}.')
            break

print(f'\nBest validation SMAPE: {best_smape:.2f}%')
print(f'Model saved to: {MODEL_PATH}')
