# train.py
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import GPT, GPTConfig


# Dataset text cơ bản
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


def train():
    # đọc dữ liệu
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # tạo vocab
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(vocab)

    # encode text
    data = [stoi[ch] for ch in text]

    # config GPT
    config = GPTConfig(vocab_size=vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)

    # chọn device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    model = model.to(device)

    # dataset + dataloader
    dataset = TextDataset(data, block_size=config.block_size)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        for i, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(device), yb.to(device)

            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {i} | Loss: {loss.item():.4f}")

        # lưu checkpoint theo epoch
        ckpt_name = f"gpt_model_epoch{epoch+1}.pt"
        torch.save(model.state_dict(), ckpt_name)
        print(f"✅ Saved checkpoint: {ckpt_name}")

    # lưu final model
    torch.save(model.state_dict(), "gpt_model.pt")
    print("🎉 Training finished. Final model saved as gpt_model.pt")


if __name__ == "__main__":
    train()
