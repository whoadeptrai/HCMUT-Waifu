# inference.py
import torch
from model import GPT, GPTConfig


def load_vocab():
    # đọc lại vocab từ input.txt (giống train.py)
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, len(vocab)


def generate_text(start_text="Hello", max_new_tokens=100, ckpt_path="gpt_model.pt"):
    stoi, itos, vocab_size = load_vocab()

    # config phải khớp với lúc train
    config = GPTConfig(vocab_size=vocab_size, block_size=64, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)

    # chọn device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Generating on:", device)

    # load checkpoint (an toàn hơn với weights_only=True)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # encode start_text -> tensor
    idx = torch.tensor([[stoi[ch] for ch in start_text if ch in stoi]], dtype=torch.long).to(device)

    # generate
    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=1.0, top_k=5)

    # decode lại thành string (chuyển về CPU trước khi decode)
    result = "".join([itos[i.item()] for i in out[0].cpu()])
    return result


if __name__ == "__main__":
    text = generate_text("Hello", max_new_tokens=200)
    print(text)
