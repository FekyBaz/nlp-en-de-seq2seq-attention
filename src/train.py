"""
Training, validation, checkpointing, and optimization routines
for Baseline and Attention Seq2Seq models.
"""

import time
import math
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PAD_IDX


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    clip: float = 1.0,
    teacher_forcing_ratio: float = 0.5,
    is_attention: bool = False,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Run a single training epoch."""
    model.train()
    epoch_loss = 0.0

    for batch in dataloader:
        src = batch["src"].to(device)
        trg = batch["trg"].to(device)
        src_lens = batch["src_lens"].to(device)

        optimizer.zero_grad()

        if is_attention:
            output, _ = model(
                src, trg, src_lens=src_lens, teacher_forcing_ratio=teacher_forcing_ratio
            )
        else:
            output = model(
                src, trg, src_lens=src_lens, teacher_forcing_ratio=teacher_forcing_ratio
            )

        # output: [batch_size, trg_len, trg_vocab_size]
        # trg: [batch_size, trg_len]
        # Exclude <sos> token at index 0 from loss computation
        output_dim = output.shape[-1]
        output_flat = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output_flat, trg_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    is_attention: bool = False,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Evaluate model loss on validation or test DataLoader (teacher forcing = 0)."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"].to(device)
            trg = batch["trg"].to(device)
            src_lens = batch["src_lens"].to(device)

            if is_attention:
                output, _ = model(src, trg, src_lens=src_lens, teacher_forcing_ratio=0.0)
            else:
                output = model(src, trg, src_lens=src_lens, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_epochs: int = 10,
    clip: float = 1.0,
    teacher_forcing_ratio: float = 0.5,
    is_attention: bool = False,
    checkpoint_path: str = "best_model.pt",
    device: torch.device = torch.device("cpu"),
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
) -> Dict[str, List[float]]:
    """
    Full training pipeline across epochs with checkpoint saving and metric tracking.
    """
    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val_loss = float("inf")

    print(f"[*] Starting training ({'Attention' if is_attention else 'Baseline'}) for {n_epochs} epochs on {device}...")

    for epoch in range(1, n_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=clip,
            teacher_forcing_ratio=teacher_forcing_ratio,
            is_attention=is_attention,
            device=device,
        )

        val_loss = evaluate_loss(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            is_attention=is_attention,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        elapsed = time.time() - start_time
        val_ppl = math.exp(min(val_loss, 20.0))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        # Checkpoint if validation loss improved
        saved_str = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            saved_str = f" [Saved Best Checkpoint -> {checkpoint_path}]"

        print(
            f"Epoch {epoch:02d}/{n_epochs:02d} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}{saved_str}"
        )

    return history
