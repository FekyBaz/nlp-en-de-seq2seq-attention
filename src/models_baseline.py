"""
Baseline Sequence-to-Sequence (LSTM without Attention) for Neural Machine Translation.

The Encoder compresses the entire source sentence into fixed-length context
vectors (final hidden and cell states), which initialize the Decoder LSTM.
"""

import random
from typing import Tuple, Optional

import torch
import torch.nn as nn


class BaselineEncoder(nn.Module):
    """
    Encoder LSTM without attention.
    Compresses variable-length source sequences into fixed-size hidden and cell states.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(
        self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: [batch_size, src_len]
            src_lens: [batch_size] length of each sequence
        Returns:
            outputs: [batch_size, src_len, hid_dim]
            (hidden, cell): each [n_layers, batch_size, hid_dim]
        """
        # src: [batch_size, src_len] -> [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        if src_lens is not None:
            # Enforce CPU lengths for pack_padded_sequence
            lens_cpu = src_lens.to("cpu")
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_outputs, (hidden, cell) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, (hidden, cell) = self.lstm(embedded)

        return outputs, (hidden, cell)


class BaselineDecoder(nn.Module):
    """
    Decoder LSTM without attention.
    Autoregressively generates target tokens given previous token and hidden state.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def decode_step(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Single decoding step: embeds token, steps LSTM, projects to vocabulary.
        """
        # input_token: [batch_size] -> [batch_size, 1]
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        # output: [batch_size, 1, hid_dim] -> [batch_size, hid_dim]
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, vocab_size]

        return prediction, (hidden, cell)

    def forward(
        self,
        input_token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.decode_step(input_token, hidden, cell)


class Seq2SeqBaseline(nn.Module):
    """
    Full Seq2Seq baseline model combining Encoder and Decoder without attention.
    Supports scheduled teacher forcing during training.
    """

    def __init__(self, encoder: BaselineEncoder, decoder: BaselineDecoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def encode(
        self, src: torch.Tensor, src_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Encode source sequence into context vector (hidden, cell)."""
        return self.encoder(src, src_lens)

    def decode_step(
        self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Step decoder autoregressively given previous token and recurrent state."""
        return self.decoder.decode_step(input_token, hidden, cell)

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        src_lens: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Full forward pass with scheduled teacher forcing.
        Args:
            src: [batch_size, src_len]
            trg: [batch_size, trg_len]
            src_lens: [batch_size]
            teacher_forcing_ratio: probability of using true target token as next input
        Returns:
            outputs: [batch_size, trg_len, trg_vocab_size]
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        # 1. Encode source sequence into fixed-length context vector (hidden, cell)
        _, (hidden, cell) = self.encode(src, src_lens)

        # 2. First input to the decoder is always <sos> token
        input_token = trg[:, 0]

        # 3. Autoregressive decoding across time steps
        for t in range(1, trg_len):
            prediction, (hidden, cell) = self.decode_step(input_token, hidden, cell)
            outputs[:, t, :] = prediction

            # Teacher forcing decision
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if use_teacher_forcing else top1

        return outputs
