
import random
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class Encoder(nn.Module):
    """Embedding -> bidirectional LSTM. Final forward/backward states are
    projected down to a single hidden_dim vector used to initialize the
    decoder."""
 
    def __init__(self, vocab_size, pad_idx, emb_dim=256, hidden_dim=512, num_layers=1, dropout=0.3):
        super().__init__()
 
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
 
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, source):
        embedded = self.dropout(self.embedding(source))              # [B, T, E]
        outputs, (hidden, cell) = self.lstm(embedded)                 # outputs: [B, T, 2H]
 
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1)))
 
        return outputs, hidden.unsqueeze(0), cell.unsqueeze(0)
 
 
class Attention(nn.Module):
    """Bahdanau (additive) attention over encoder outputs."""
 
    def __init__(self, hidden_dim):
        super().__init__()
 
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
 
    def forward(self, decoder_hidden, encoder_outputs, mask):
        seq_len = encoder_outputs.size(1)
 
        hidden_repeated = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)   # [B, T, H]
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)                                    # [B, T]
 
        scores = scores.masked_fill(mask == 0, -1e10)
 
        return F.softmax(scores, dim=1)
 
 
class Decoder(nn.Module):
    """One decoding step: embed previous token, attend over encoder outputs,
    run an LSTM cell, project to vocabulary logits."""
 
    def __init__(self, vocab_size, pad_idx, emb_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
 
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(emb_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)     # [B, 1, E]
 
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)   # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)           # [B, 1, 2H]
 
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
 
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
 
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
 
        return prediction, hidden, cell, attn_weights
 
 
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, source_pad_idx, device):
        super().__init__()
 
        self.encoder = encoder
        self.decoder = decoder
        self.source_pad_idx = source_pad_idx
        self.device = device
 
    def create_mask(self, source):
        return source != self.source_pad_idx
 
    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size, target_len = target.shape
        vocab_size = self.decoder.fc_out.out_features
 
        outputs = torch.zeros(batch_size, target_len, vocab_size, device=self.device)
 
        encoder_outputs, hidden, cell = self.encoder(source)
        mask = self.create_mask(source)
 
        input_token = target[:, 0]  # <sos>
 
        for t in range(1, target_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[:, t] = prediction
 
            top1 = prediction.argmax(1)
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = target[:, t] if teacher_force else top1
 
        return outputs
 
    @torch.no_grad()
    def greedy_decode(self, source, sos_idx, eos_idx, max_len=60):
        self.eval()
 
        encoder_outputs, hidden, cell = self.encoder(source)
        mask = self.create_mask(source)
 
        batch_size = source.size(0)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=self.device)
 
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        generated = torch.full((batch_size, max_len), eos_idx, dtype=torch.long, device=self.device)
 
        for t in range(max_len):
            prediction, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            top1 = prediction.argmax(1)
 
            generated[:, t] = top1
            finished = finished | (top1 == eos_idx)
            input_token = top1
 
            if finished.all():
                break
 
        return generated
 
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
