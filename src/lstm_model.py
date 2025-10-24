import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, start_seq, max_len=20, device="cpu"):
        self.eval()
        generated = start_seq.tolist()
        input_seq = start_seq.unsqueeze(0).to(device)
        hidden = None
        for _ in range(max_len):
            logits, hidden = self.forward(input_seq, hidden)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_token.item())
            input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)
        return generated
