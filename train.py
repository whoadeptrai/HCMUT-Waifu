# train.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig
import requests
import json


# -------------------------
# Load dữ liệu
# -------------------------
def load_local_data():
    with open("data/input.txt", "r", encoding="utf-8") as f:
        return f.read()

def load_remote_data():
    url = "https://huggingface.co/datasets/daily_dialog/resolve/main/dailydialog_text.txt"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text

def load_combined_data():
    local_text = load_local_data()
    try:
        remote_text = load_remote_data()
        print("✅ Đã tải thêm dữ liệu từ URL")
        return local_text + "\n" + remote_text
    except Exception as e:
        print(f"⚠️ Không lấy được dữ liệu từ URL ({e}), chỉ dùng local input.txt")
        return local_text


# -------------------------
# Dataset
# -------------------------
class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# -------------------------
# Train loop
# -------------------------
def train():
    # lấy dữ liệu từ local + remote (nếu có)
    text = load_combined_data()

    # tạo vocab
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(vocab)

    # lưu vocab ra file JSON để inference dùng lại
    vocab_info = {
        "stoi": stoi,
        "itos": itos,
        "vocab_size": vocab_size
    }
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, ensure_ascii=False, indent=2)

    # encode dữ liệu sang id
    data = [stoi[ch] for ch in text]

    # config GPT
    config = GPTConfig(vocab_size=vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)

    # chọn device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🚀 Training on:", device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    model = model.to(device)

    # dataset + dataloader (pin_memory để tăng tốc GPU)
    dataset = TextDataset(data, block_size=config.block_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # training loop
    for epoch in range(5):  # tăng số epoch nếu muốn
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"📌 Epoch {epoch+1} | Loss: {loss.item():.4f}")

        # lưu model theo từng epoch
        torch.save(model.state_dict(), f"gpt_model_epoch{epoch+1}.pt")

    # lưu model cuối
    torch.save(model.state_dict(), "gpt_model.pt")
    print("✅ Training hoàn tất, model đã lưu vào gpt_model.pt và vocab.json")


if __name__ == "__main__":
    train()
