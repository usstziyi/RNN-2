"""
字符级 LSTM 文本生成 (Char-LSTM Text Generation)

完整流水线:
  语料加载 -> 字符级Tokenize -> 构建词表 -> 滑动窗口数据集
  -> LSTM模型构建 -> 训练循环 -> 采样生成 -> 结果分析

数据: Tiny Shakespeare (Karpathy's char-rnn)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib.request
import os
import argparse


# ============================================================
# 0. 设备检测
# ============================================================
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================
# 1. 数据加载
# ============================================================

SAMPLE_TEXT = """\
ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she:
Be not her maid, since she is envious;
Her vestal livery is but sick and green
And none but fools do wear it; cast it off.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

HAMLET:
To be, or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles,
And by opposing end them? To die: to sleep;
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to, 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep: perchance to dream: ay, there's the rub;
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause.
"""


def load_shakespeare():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    cache_path = "/tmp/tinyshakespeare.txt"

    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read()

    try:
        print("正在下载 Tiny Shakespeare 数据集...")
        urllib.request.urlretrieve(url, cache_path)
        with open(cache_path, 'r') as f:
            text = f.read()
        print(f"下载成功! 文本长度: {len(text):,} 字符")
        return text
    except Exception as e:
        print(f"下载失败 ({e})，使用内置样本数据")
        return SAMPLE_TEXT


# ============================================================
# 2. 字符级词表构建
# ============================================================

def build_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    encoded_text = np.array([char_to_idx[ch] for ch in text], dtype=np.int64)
    print(f"唯一字符数: {vocab_size}")
    print(f"编码后序列长度: {len(encoded_text):,}")
    return vocab_size, char_to_idx, idx_to_char, encoded_text


# ============================================================
# 3. 滑动窗口数据集
# ============================================================

class CharDataset(Dataset):
    def __init__(self, encoded_text, seq_length):
        self.data = encoded_text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def create_dataloaders(encoded_text, seq_length, batch_size, train_ratio=0.9):
    split_idx = int(len(encoded_text) * train_ratio)
    train_data = encoded_text[:split_idx]
    val_data = encoded_text[split_idx:]

    train_dataset = CharDataset(train_data, seq_length)
    val_dataset = CharDataset(val_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print(f"训练样本数: {len(train_dataset):,}")
    print(f"验证样本数: {len(val_dataset):,}")
    print(f"每 epoch 训练 batch 数: {len(train_loader)}")
    return train_loader, val_loader


# ============================================================
# 4. LSTM 语言模型
# ============================================================

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers=3, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'lstm' in name and 'bias' in name:
                nn.init.zeros_(param)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.0)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        lstm_out, hidden = self.lstm(emb, hidden)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)


# ============================================================
# 5. 训练与评估
# ============================================================

def train_epoch(model, loader, optimizer, criterion, vocab_size, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, vocab_size):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        total_loss += loss.item()
    return total_loss / len(loader)


def train(model, train_loader, val_loader, vocab_size, num_epochs=25, lr=0.002):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    train_losses, val_losses = [], []
    train_ppls, val_ppls = [], []

    print(f"开始训练 {num_epochs} 个 epoch...\n")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, vocab_size)
        val_loss = evaluate(model, val_loader, criterion, vocab_size)

        scheduler.step()

        train_ppl = np.exp(train_loss)
        val_ppl = np.exp(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ppls.append(train_ppl)
        val_ppls.append(val_ppl)

        if epoch % 5 == 0 or epoch == 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:2d}/{num_epochs} | lr={current_lr:.4f} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val PPL: {val_ppl:.2f}")

    print(f"\n训练完成! 最终 Val Perplexity: {val_ppls[-1]:.2f}")
    return train_losses, val_losses, train_ppls, val_ppls


# ============================================================
# 6. 采样策略
# ============================================================

def sample_greedy(logits):
    probs = F.softmax(logits, dim=-1)
    return torch.argmax(probs, dim=-1)


def sample_temperature(logits, temperature=0.8):
    logits = logits / max(temperature, 1e-9)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def sample_top_k(logits, k=10, temperature=0.8):
    logits = logits / max(temperature, 1e-9)
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)
    logits = torch.where(logits < min_top_k, torch.tensor(float('-inf'), device=device), logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ============================================================
# 7. 文本生成
# ============================================================

def generate(model, seed_text, char_to_idx, idx_to_char, length=200,
             strategy='temperature', temperature=0.8, top_k=10):
    model.eval()

    if strategy == 'greedy':
        sample_fn = lambda logits: sample_greedy(logits)
    elif strategy == 'top_k':
        sample_fn = lambda logits: sample_top_k(logits, k=top_k, temperature=temperature)
    else:
        sample_fn = lambda logits: sample_temperature(logits, temperature=temperature)

    input_ids = torch.tensor(
        [[char_to_idx[ch] for ch in seed_text if ch in char_to_idx]],
        dtype=torch.long, device=device
    )

    generated_chars = list(seed_text)
    hidden = None

    with torch.no_grad():
        if input_ids.size(1) > 1:
            logits, hidden = model(input_ids[:, :-1])
            last_input = input_ids[:, -1:]
        else:
            last_input = input_ids

        for _ in range(length):
            logits, hidden = model(last_input, hidden)
            next_token = sample_fn(logits[:, -1, :])

            if next_token.item() not in idx_to_char:
                break

            generated_chars.append(idx_to_char[next_token.item()])
            last_input = next_token.unsqueeze(1)

    return ''.join(generated_chars)


# ============================================================
# 8. 可视化
# ============================================================

def plot_training_curves(train_losses, val_losses, train_ppls, val_ppls, vocab_size):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss', linewidth=1.5)
    ax1.plot(val_losses, label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_ppls, label='Train PPL', linewidth=1.5)
    ax2.plot(val_ppls, label='Val PPL', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Perplexity')
    ax2.set_title('Training & Validation Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/tmp/training_curves.png", dpi=120)
    plt.show()
    print(f"随机猜测的 PPL = {vocab_size}（均匀分布）")


# ============================================================
# 9. 模型保存与加载
# ============================================================

def save_model(model, char_to_idx, idx_to_char, vocab_size, embed_dim, hidden_size, num_layers, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'embed_dim': embed_dim,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
    }, path)
    print(f"模型已保存到 {path}")


def load_model(path):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = CharLSTM(
        checkpoint['vocab_size'], checkpoint['embed_dim'],
        checkpoint['hidden_size'], checkpoint['num_layers']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"模型已从 {path} 加载")
    return model, checkpoint['char_to_idx'], checkpoint['idx_to_char']


# ============================================================
# 10. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Char-LSTM Text Generation")
    parser.add_argument("--seq_length", type=int, default=100, help="输入序列长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--embed_dim", type=int, default=128, help="嵌入维度")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=3, help="LSTM 层数")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比率")
    parser.add_argument("--num_epochs", type=int, default=25, help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.002, help="学习率")
    parser.add_argument("--generate_length", type=int, default=300, help="生成文本长度")
    parser.add_argument("--save_path", type=str, default="/tmp/char_lstm_shakespeare.pt", help="模型保存路径")
    parser.add_argument("--seed", type=str, default="ROMEO:\n", help="生成种子文本")
    args = parser.parse_args()

    # ---- 加载数据 ----
    text = load_shakespeare()
    print(f"前 200 个字符:\n{text[:200]}")

    vocab_size, char_to_idx, idx_to_char, encoded_text = build_vocab(text)

    # ---- 创建 DataLoader ----
    train_loader, val_loader = create_dataloaders(
        encoded_text, args.seq_length, args.batch_size
    )

    # ---- 构建模型 ----
    model = CharLSTM(
        vocab_size, args.embed_dim, args.hidden_size,
        args.num_layers, args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # ---- 训练 ----
    train_losses, val_losses, train_ppls, val_ppls = train(
        model, train_loader, val_loader, vocab_size,
        num_epochs=args.num_epochs, lr=args.lr
    )

    # ---- 可视化 ----
    plot_training_curves(train_losses, val_losses, train_ppls, val_ppls, vocab_size)

    # ---- 保存模型 ----
    save_model(model, char_to_idx, idx_to_char, vocab_size,
               args.embed_dim, args.hidden_size, args.num_layers, args.save_path)

    # ---- 文本生成演示 ----
    print("\n" + "=" * 60)
    print(f"【Greedy 采样】—— 总是选概率最高的")
    print("=" * 60)
    greedy_text = generate(model, args.seed, char_to_idx, idx_to_char,
                           length=args.generate_length, strategy='greedy')
    print(greedy_text)

    print("\n" + "=" * 60)
    print("【Temperature = 0.7 采样】—— 平衡确定性与随机性")
    print("=" * 60)
    temp_text = generate(model, args.seed, char_to_idx, idx_to_char,
                         length=args.generate_length, strategy='temperature', temperature=0.7)
    print(temp_text)

    print("\n" + "=" * 60)
    print("【Top-K = 10, Temperature = 0.8】—— 推荐设置")
    print("=" * 60)
    topk_text = generate(model, args.seed, char_to_idx, idx_to_char,
                         length=args.generate_length, strategy='top_k', temperature=0.8, top_k=10)
    print(topk_text)

    # ---- 验证加载 ----
    loaded_model, _, _ = load_model(args.save_path)
    with torch.no_grad():
        x = torch.randint(0, vocab_size, (1, 10)).to(device)
        out_orig, _ = model(x)
        out_loaded, _ = loaded_model(x)
        diff = (out_orig - out_loaded).abs().max().item()
        print(f"加载验证 — 输出最大差异: {diff:.10f}")


if __name__ == "__main__":
    main()
