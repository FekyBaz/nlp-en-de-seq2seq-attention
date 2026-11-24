"""
End-to-end experiment pipeline:
1. Loads Multi30k dataset (train/val/test splits).
2. Builds and trains Baseline Seq2Seq model.
3. Builds and trains Seq2Seq with Bahdanau Attention model.
4. Generates loss curves and saves checkpoints.
5. Evaluates corpus BLEU-1 to 4 and ROUGE-L on held-out test set.
6. Performs length-stratified analysis (bottleneck analysis).
7. Extracts attention weights and saves 2D heatmaps.
8. Generates qualitative comparison table.
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import (
    load_and_prepare_data,
    get_dataloaders,
    PAD_IDX,
    tokenize_en,
    tokenize_de,
)
from src.models_baseline import BaselineEncoder, BaselineDecoder, Seq2SeqBaseline
from src.models_attention import (
    BidirectionalEncoder,
    BahdanauAttention,
    AttentionDecoder,
    Seq2SeqAttention,
)
from src.train import train_model
from src.evaluate import (
    translate_sentence_baseline,
    translate_sentence_attention,
    evaluate_metrics,
    evaluate_by_length,
)
from src.visualize import (
    plot_loss_curves,
    plot_attention_heatmap,
    plot_length_analysis,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate NMT models.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation on existing checkpoints.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # Output directories
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    # 1. Dataset & Preprocessing
    print("\n" + "=" * 60)
    print("STEP 1: Loading Dataset and Preparing Vocabularies")
    print("=" * 60)
    train_limit = 10000
    val_limit = 1014
    test_limit = 1000

    src_vocab, trg_vocab, raw_data = load_and_prepare_data(
        train_limit=train_limit,
        val_limit=val_limit,
        test_limit=test_limit,
        min_freq=2,
        max_len=40,
    )

    print(f"[+] Source (EN) Vocab size: {len(src_vocab)}")
    print(f"[+] Target (DE) Vocab size: {len(trg_vocab)}")
    print(f"[+] Train pairs: {len(raw_data['train'][0])}")
    print(f"[+] Val pairs: {len(raw_data['validation'][0])}")
    print(f"[+] Test pairs: {len(raw_data['test'][0])}")

    batch_size = 64
    train_loader, val_loader, test_loader = get_dataloaders(
        src_vocab, trg_vocab, raw_data, batch_size=batch_size, max_len=40
    )

    # Architectural Hyperparameters for Fair Comparison
    emb_dim = 128
    hid_dim = 256  # Baseline hidden_dim = 256, Attention enc_hid=128 (bidirectional -> 256), dec_hid=256
    dropout = 0.3
    n_epochs = 10
    lr = 1e-3
    clip = 1.0

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    b_encoder = BaselineEncoder(
        vocab_size=len(src_vocab),
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        n_layers=1,
        dropout=dropout,
        pad_idx=PAD_IDX,
    )
    b_decoder = BaselineDecoder(
        vocab_size=len(trg_vocab),
        emb_dim=emb_dim,
        hid_dim=hid_dim,
        n_layers=1,
        dropout=dropout,
        pad_idx=PAD_IDX,
    )
    baseline_model = Seq2SeqBaseline(b_encoder, b_decoder, device).to(device)
    b_checkpoint = "outputs/checkpoints/baseline_best.pt"

    a_encoder = BidirectionalEncoder(
        vocab_size=len(src_vocab),
        emb_dim=emb_dim,
        enc_hid_dim=hid_dim // 2,
        dec_hid_dim=hid_dim,
        dropout=dropout,
        pad_idx=PAD_IDX,
    )
    attention = BahdanauAttention(
        enc_hid_dim=hid_dim // 2,
        dec_hid_dim=hid_dim,
        attn_dim=hid_dim // 2,
    )
    a_decoder = AttentionDecoder(
        vocab_size=len(trg_vocab),
        emb_dim=emb_dim,
        enc_hid_dim=hid_dim // 2,
        dec_hid_dim=hid_dim,
        attention=attention,
        dropout=dropout,
        pad_idx=PAD_IDX,
    )
    attention_model = Seq2SeqAttention(
        encoder=a_encoder,
        decoder=a_decoder,
        pad_idx=PAD_IDX,
        device=device,
    ).to(device)
    a_checkpoint = "outputs/checkpoints/attention_best.pt"

    if not args.eval_only:
        # 2. Train Baseline Seq2Seq
        print("\n" + "=" * 60)
        print("STEP 2: Training Baseline Seq2Seq (LSTM without Attention)")
        print("=" * 60)
        b_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=lr)
        baseline_history = train_model(
            model=baseline_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=b_optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            clip=clip,
            teacher_forcing_ratio=0.5,
            is_attention=False,
            checkpoint_path=b_checkpoint,
            device=device,
        )

        # 3. Train Seq2Seq with Bahdanau Attention
        print("\n" + "=" * 60)
        print("STEP 3: Training Seq2Seq with Bahdanau Attention")
        print("=" * 60)
        a_optimizer = torch.optim.Adam(attention_model.parameters(), lr=lr)
        attention_history = train_model(
            model=attention_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=a_optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            clip=clip,
            teacher_forcing_ratio=0.5,
            is_attention=True,
            checkpoint_path=a_checkpoint,
            device=device,
        )

        # Save histories
        with open("outputs/results/history.json", "w", encoding="utf-8") as f:
            json.dump({"baseline": baseline_history, "attention": attention_history}, f, indent=2)

        # 4. Plot Loss Curves
        print("\n" + "=" * 60)
        print("STEP 4: Plotting Convergence Loss Curves")
        print("=" * 60)
        plot_loss_curves(
            baseline_history,
            attention_history,
            save_path="outputs/figures/loss_curves.png",
        )

    # Load best weights for rigorous evaluation
    baseline_model.load_state_dict(torch.load(b_checkpoint, map_location=device))
    attention_model.load_state_dict(torch.load(a_checkpoint, map_location=device))
    baseline_model.eval()
    attention_model.eval()

    # 5. Held-Out Test Evaluation (BLEU-1..4 & ROUGE-L)
    print("\n" + "=" * 60)
    print("STEP 5: Evaluating Models on Held-Out Test Set")
    print("=" * 60)
    test_en, test_de = raw_data["test"]
    eval_test_en = test_en[:500]  # Representative held-out sample for high-precision metric scoring
    eval_test_de = test_de[:500]

    baseline_translations = []
    attention_translations = []
    reference_tokenized = []

    for en_sent, de_sent in zip(eval_test_en, eval_test_de):
        base_toks = translate_sentence_baseline(en_sent, baseline_model, src_vocab, trg_vocab, device)
        attn_toks, _, _ = translate_sentence_attention(en_sent, attention_model, src_vocab, trg_vocab, device)
        baseline_translations.append(" ".join(base_toks))
        attention_translations.append(" ".join(attn_toks))
        reference_tokenized.append(" ".join(tokenize_de(de_sent)))

    baseline_metrics = evaluate_metrics(baseline_translations, reference_tokenized)
    attention_metrics = evaluate_metrics(attention_translations, reference_tokenized)

    metrics_df = pd.DataFrame([baseline_metrics, attention_metrics], index=["Baseline (LSTM)", "Seq2Seq + Attention"])
    print("\n--- Test Set Quantitative Results ---")
    print(metrics_df.to_string())

    with open("outputs/results/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"Baseline": baseline_metrics, "Attention": attention_metrics}, f, indent=2)

    # 6. Length-Stratified Bottleneck Analysis
    print("\n" + "=" * 60)
    print("STEP 6: Sentence-Length Bottleneck Analysis")
    print("=" * 60)
    length_results = evaluate_by_length(
        src_sentences=eval_test_en,
        ref_sentences=eval_test_de,
        baseline_model=baseline_model,
        attention_model=attention_model,
        src_vocab=src_vocab,
        trg_vocab=trg_vocab,
        device=device,
    )
    with open("outputs/results/length_analysis.json", "w", encoding="utf-8") as f:
        json.dump(length_results, f, indent=2)

    print(json.dumps(length_results, indent=2))
    plot_length_analysis(length_results, save_path="outputs/figures/length_analysis.png")

    # 7. Attention Visualization on Diverse Test Sentences
    print("\n" + "=" * 60)
    print("STEP 7: Generating Attention Heatmaps on Diverse Test Sentences")
    print("=" * 60)
    sample_sentences = [
        "A young boy in a blue jacket is climbing a wooden fence.",
        "A group of people are waiting at the train station.",
        "A black dog runs through the green grass with a yellow tennis ball.",
        "Two women are sitting outside a cafe while drinking coffee.",
        "A man rides a bicycle down a crowded city street.",
    ]

    for idx, sent in enumerate(sample_sentences, 1):
        pred_tokens, attn_mat, src_tokens = translate_sentence_attention(
            sent, attention_model, src_vocab, trg_vocab, device
        )
        pred_text = " ".join(pred_tokens)
        print(f"\n[Sample {idx}]")
        print(f"EN:   {sent}")
        print(f"DE:   {pred_text}")

        plot_attention_heatmap(
            src_tokens=src_tokens,
            trg_tokens=pred_tokens,
            attention_matrix=attn_mat,
            save_path=f"outputs/figures/attention_sample_{idx}.png",
            title=f"Attention Alignment: Sample {idx}",
        )

    # 8. Side-by-Side Qualitative Error Analysis Table
    print("\n" + "=" * 60)
    print("STEP 8: Building Qualitative Error Analysis Table")
    print("=" * 60)
    qualitative_indices = [3, 14, 25, 42, 67, 88, 102]
    table_rows = []

    for q_idx in qualitative_indices:
        src_raw = test_en[q_idx]
        ref_raw = test_de[q_idx]
        b_pred = " ".join(translate_sentence_baseline(src_raw, baseline_model, src_vocab, trg_vocab, device))
        a_pred = " ".join(translate_sentence_attention(src_raw, attention_model, src_vocab, trg_vocab, device)[0])

        # Infer error category
        error_note = ""
        if "<unk>" in b_pred or "<unk>" in a_pred:
            error_note = "Vocabulary limitation (<unk> token)"
        elif len(b_pred.split()) < len(ref_raw.split()) // 2:
            error_note = "Dropped content words in Baseline"
        elif any(b_pred.split().count(w) > 3 for w in b_pred.split()):
            error_note = "Repetition error in Baseline"
        else:
            error_note = "Syntactic / Word Order Alignment"

        table_rows.append({
            "Sample ID": f"Test-{q_idx}",
            "Source (English)": src_raw,
            "Ground Truth (German)": ref_raw,
            "Baseline Prediction": b_pred,
            "Attention Prediction": a_pred,
            "Error Category / Observation": error_note,
        })

    qual_df = pd.DataFrame(table_rows)
    qual_df.to_csv("outputs/results/qualitative_table.csv", index=False)
    qual_df.to_markdown("outputs/results/qualitative_table.md", index=False)
    print(qual_df.to_string())

    print("\n" + "=" * 60)
    print("[SUCCESS] All experiments, evaluations, and figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
