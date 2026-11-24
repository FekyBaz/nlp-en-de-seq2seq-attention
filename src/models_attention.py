"""
Sequence-to-Sequence with Bahdanau (Additive) Attention for Neural Machine Translation.

The Bidirectional Encoder outputs annotations for every source time step.
At each decoding step, the Attention mechanism computes alignment weights (alpha_{t,s})
over all encoder hidden states to form a dynamic context vector.
"""

import random
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalEncoder(nn.Module):
    """
    Bidirectional LSTM Encoder.
    Outputs representations for every source token by concatenating forward and backward passes.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        # Linear projections to initialize decoder hidden and cell states from concatenated encoder states
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(
        self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            src: [batch_size, src_len]
            src_lens: [batch_size] length of each sequence
        Returns:
            encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
            hidden: [batch_size, dec_hid_dim] (initial decoder hidden state)
            cell: [batch_size, dec_hid_dim] (initial decoder cell state)
        """
        # src: [batch_size, src_len] -> [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        if src_lens is not None:
            lens_cpu = src_lens.to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # hidden is [2, batch_size, enc_hid_dim] -> [batch_size, enc_hid_dim * 2]
        # hidden[-2, :, :] is forward final, hidden[-1, :, :] is backward final
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        cell_cat = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

        init_hidden = torch.tanh(self.fc_hidden(hidden_cat))
        init_cell = torch.tanh(self.fc_cell(cell_cat))

        return encoder_outputs, init_hidden, init_cell


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    Computes dynamic alignment scores e_{t,s} and attention weights alpha_{t,s}.
    """

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int = 128):
        super().__init__()
        self.W_dec = nn.Linear(dec_hid_dim, attn_dim, bias=False)
        self.W_enc = nn.Linear(enc_hid_dim * 2, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden: [batch_size, dec_hid_dim] (s_{t-1})
            encoder_outputs: [batch_size, src_len, enc_hid_dim * 2] (H)
            mask: [batch_size, src_len] boolean tensor where True indicates padding
        Returns:
            attention_weights: [batch_size, src_len] (alpha_{t,s})
        """
        src_len = encoder_outputs.shape[1]

        # dec_proj: [batch_size, 1, attn_dim] -> broadcasted to [batch_size, src_len, attn_dim]
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)
        # enc_proj: [batch_size, src_len, attn_dim]
        enc_proj = self.W_enc(encoder_outputs)

        # energy: [batch_size, src_len]
        energy = self.v(torch.tanh(dec_proj + enc_proj)).squeeze(2)

        # Mask padding tokens with large negative values so their attention weight is ~0
        if mask is not None:
            energy = energy.masked_fill(mask, -1e9)

        attention_weights = F.softmax(energy, dim=1)
        return attention_weights


class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau Attention.
    Conditioned on target embedding, previous hidden state, and dynamic context vector.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        attention: BahdanauAttention,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.attention = attention
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # LSTMCell receives [embedded_token; context_vector]
        self.lstm_cell = nn.LSTMCell(emb_dim + (enc_hid_dim * 2), dec_hid_dim)

        # Linear projection to target vocabulary from [decoder_hidden; context_vector; embedded_token]
        self.fc_out = nn.Linear(dec_hid_dim + (enc_hid_dim * 2) + emb_dim, vocab_size)

    def forward(
        self,
        input_token: torch.Tensor,
        decoder_hidden: torch.Tensor,
        decoder_cell: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.
        Args:
            input_token: [batch_size]
            decoder_hidden: [batch_size, dec_hid_dim] (s_{t-1})
            decoder_cell: [batch_size, dec_hid_dim] (c_{t-1})
            encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
            mask: [batch_size, src_len]
        Returns:
            prediction: [batch_size, vocab_size]
            new_hidden: [batch_size, dec_hid_dim]
            new_cell: [batch_size, dec_hid_dim]
            attention_weights: [batch_size, src_len]
        """
        # input_token: [batch_size] -> [batch_size, 1] -> [batch_size, emb_dim]
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1))).squeeze(1)

        # Compute dynamic attention weights: [batch_size, src_len]
        attn_weights = self.attention(decoder_hidden, encoder_outputs, mask)

        # Compute context vector: [batch_size, enc_hid_dim * 2]
        # attn_weights.unsqueeze(1): [batch_size, 1, src_len]
        # encoder_outputs: [batch_size, src_len, enc_hid_dim * 2]
        # bmm gives [batch_size, 1, enc_hid_dim * 2] -> squeeze(1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        # Feed [embedded; context] into LSTMCell
        lstm_input = torch.cat((embedded, context), dim=1)
        new_hidden, new_cell = self.lstm_cell(lstm_input, (decoder_hidden, decoder_cell))

        # Output prediction from [new_hidden; context; embedded]
        output_repr = torch.cat((new_hidden, context, embedded), dim=1)
        prediction = self.fc_out(self.dropout(output_repr))

        return prediction, new_hidden, new_cell, attn_weights


class Seq2SeqAttention(nn.Module):
    """
    Full Seq2Seq model with Bahdanau Attention.
    Supports scheduled teacher forcing and extraction of attention weights across time steps.
    """

    def __init__(
        self,
        encoder: BidirectionalEncoder,
        decoder: AttentionDecoder,
        pad_idx: int,
        device: torch.device,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create mask tensor where True indicates padding."""
        return (src == self.pad_idx)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: [batch_size, src_len]
            trg: [batch_size, trg_len]
            src_lens: [batch_size]
            teacher_forcing_ratio: float
        Returns:
            outputs: [batch_size, trg_len, trg_vocab_size]
            attentions: [batch_size, trg_len - 1, src_len]
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        src_len = src.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, trg_len - 1, src_len, device=self.device)

        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_lens)
        mask = self.create_mask(src)

        # First decoder input is <sos> token
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, cell, attn_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t, :] = prediction
            attentions[:, t - 1, :] = attn_weights

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if use_teacher_forcing else top1

        return outputs, attentions
