# tokenizer.py

def load_vocab():
    # đọc toàn bộ file data.txt
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}

    return stoi, itos, len(vocab)
