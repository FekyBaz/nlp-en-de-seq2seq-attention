"""
Inference, greedy translation, attention extraction, and quantitative evaluation.
Computes corpus BLEU-1 through BLEU-4, ROUGE-L, and length-stratified metrics.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from rouge_score import rouge_scorer
import sacrebleu

from .data import (
    Vocab,
    tokenize_en,
    tokenize_de,
    SOS_IDX,
    EOS_IDX,
    PAD_IDX,
)


def translate_sentence_baseline(
    sentence: str,
    model: nn.Module,
    src_vocab: Vocab,
    trg_vocab: Vocab,
    device: torch.device,
    max_len: int = 50,
) -> List[str]:
    """
    Greedy decoding translation using the Baseline Seq2Seq model.
    """
    model.eval()

    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = [str(t).lower() for t in sentence]

    src_ids = src_vocab.encode(tokens, add_sos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        _, (hidden, cell) = model.encoder(src_tensor, src_lens)

    trg_indices = [SOS_IDX]

    for _ in range(max_len):
        input_token = torch.tensor([trg_indices[-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            prediction, (hidden, cell) = model.decoder(input_token, hidden, cell)

        pred_idx = prediction.argmax(1).item()
        if pred_idx == EOS_IDX:
            break
        trg_indices.append(pred_idx)

    # Decode indices to tokens excluding <sos> and <eos>
    translated_tokens = [trg_vocab.idx2token.get(idx, "<unk>") for idx in trg_indices[1:]]
    return translated_tokens


def translate_sentence_attention(
    sentence: str,
    model: nn.Module,
    src_vocab: Vocab,
    trg_vocab: Vocab,
    device: torch.device,
    max_len: int = 50,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Greedy decoding translation using Seq2Seq with Bahdanau Attention.
    Returns:
        translated_tokens: List[str]
        attention_matrix: np.ndarray of shape [pred_len, src_len]
        source_tokens: List[str] with <sos> and <eos>
    """
    model.eval()

    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = [str(t).lower() for t in sentence]

    src_ids = src_vocab.encode(tokens, add_sos=True, add_eos=True)
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
    src_lens = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor, src_lens)
        mask = model.create_mask(src_tensor)

    trg_indices = [SOS_IDX]
    attentions_list = []

    for _ in range(max_len):
        input_token = torch.tensor([trg_indices[-1]], dtype=torch.long, device=device)
        with torch.no_grad():
            prediction, hidden, cell, attn_weights = model.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )

        pred_idx = prediction.argmax(1).item()
        attentions_list.append(attn_weights.squeeze(0).cpu().numpy())

        if pred_idx == EOS_IDX:
            break
        trg_indices.append(pred_idx)

    translated_tokens = [trg_vocab.idx2token.get(idx, "<unk>") for idx in trg_indices[1:]]
    attention_matrix = np.array(attentions_list) if attentions_list else np.zeros((0, len(src_ids)))

    # Source tokens with <sos> and <eos>
    source_tokens = ["<sos>"] + tokens + ["<eos>"]

    # Match matrix row dimension to translated tokens if EOS was reached
    if len(translated_tokens) > 0 and len(attention_matrix) > len(translated_tokens):
        attention_matrix = attention_matrix[: len(translated_tokens)]

    return translated_tokens, attention_matrix, source_tokens


def evaluate_metrics(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4 and ROUGE-L.
    """
    # SacreBLEU expects list of references as list of systems: [[ref1, ref2, ...]]
    ref_list = [references]

    bleu_1 = sacrebleu.BLEU(max_ngram_order=1).corpus_score(hypotheses, ref_list).score
    bleu_2 = sacrebleu.BLEU(max_ngram_order=2).corpus_score(hypotheses, ref_list).score
    bleu_3 = sacrebleu.BLEU(max_ngram_order=3).corpus_score(hypotheses, ref_list).score
    bleu_4 = sacrebleu.BLEU(max_ngram_order=4).corpus_score(hypotheses, ref_list).score

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hypotheses, references)
    ]
    mean_rouge_l = float(np.mean(rouge_l_scores)) * 100.0

    return {
        "BLEU-1": round(bleu_1, 2),
        "BLEU-2": round(bleu_2, 2),
        "BLEU-3": round(bleu_3, 2),
        "BLEU-4": round(bleu_4, 2),
        "ROUGE-L": round(mean_rouge_l, 2),
    }


def evaluate_by_length(
    src_sentences: List[str],
    ref_sentences: List[str],
    baseline_model: nn.Module,
    attention_model: nn.Module,
    src_vocab: Vocab,
    trg_vocab: Vocab,
    device: torch.device,
    buckets: Optional[List[Tuple[str, int, int]]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Stratifies test set by source length and computes BLEU / ROUGE-L per bucket
    to demonstrate the information bottleneck mitigation of Attention.
    """
    if buckets is None:
        buckets = [
            ("Short (3-6 words)", 3, 6),
            ("Medium (7-11 words)", 7, 11),
            ("Long (12-18+ words)", 12, 100),
        ]

    bucket_data = {b[0]: {"src": [], "ref": []} for b in buckets}

    for src, ref in zip(src_sentences, ref_sentences):
        length = len(tokenize_en(src))
        for name, min_l, max_l in buckets:
            if min_l <= length <= max_l:
                bucket_data[name]["src"].append(src)
                bucket_data[name]["ref"].append(ref)
                break

    results = {}
    for name, data in bucket_data.items():
        src_subset = data["src"]
        ref_subset = data["ref"]
        count = len(src_subset)

        if count == 0:
            continue

        baseline_hyps = []
        attn_hyps = []

        for s in src_subset:
            base_tokens = translate_sentence_baseline(s, baseline_model, src_vocab, trg_vocab, device)
            attn_tokens, _, _ = translate_sentence_attention(s, attention_model, src_vocab, trg_vocab, device)
            baseline_hyps.append(" ".join(base_tokens))
            attn_hyps.append(" ".join(attn_tokens))

        cleaned_refs = [" ".join(tokenize_de(r)) for r in ref_subset]

        base_metrics = evaluate_metrics(baseline_hyps, cleaned_refs)
        attn_metrics = evaluate_metrics(attn_hyps, cleaned_refs)

        results[name] = {
            "count": count,
            "Baseline": base_metrics,
            "Attention": attn_metrics,
        }

    return results
