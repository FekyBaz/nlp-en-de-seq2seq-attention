"""
Data preprocessing, vocabulary construction, tokenization, and dataset loading
for English-to-German Neural Machine Translation.
"""

import re
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


# Special tokens and standard indices
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def clean_text(text: str, lang: str = "en") -> str:
    """
    Normalizes text, standardizes whitespaces, and cleans punctuation while
    strictly preserving German characters: ä, ö, ü, ß (and their capital forms).
    """
    if not isinstance(text, str):
        text = str(text)

    # Unicode normalization (NFC ensures composed characters like ä remain intact)
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()

    # Separate punctuation marks with whitespace so tokenization is clean
    # Preserves characters: a-z, 0-9, German letters (ä, ö, ü, ß)
    # Replaces special symbols, quotes, hyphens, and standard punctuation with spaced tokens
    text = re.sub(r"([?.!,;:\'\"()\-–—])", r" \1 ", text)
    
    # Filter out extraneous non-alphanumeric symbols except valid punctuation and German umlauts
    if lang == "de":
        text = re.sub(r"[^a-z0-9äöüß?.!,;:\'\"()\- ]+", " ", text)
    else:
        text = re.sub(r"[^a-z0-9?.!,;:\'\"()\- ]+", " ", text)

    # Collapse multiple whitespace characters into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_en(text: str) -> List[str]:
    """Tokenize English sentence into a list of lowercase string tokens."""
    cleaned = clean_text(text, lang="en")
    return cleaned.split() if cleaned else []


def tokenize_de(text: str) -> List[str]:
    """Tokenize German sentence into a list of lowercase string tokens."""
    cleaned = clean_text(text, lang="de")
    return cleaned.split() if cleaned else []


class Vocab:
    """
    Vocabulary class that maps tokens to indices and vice versa,
    supporting min_freq thresholds and special tokens.
    """

    def __init__(self, name: str = "vocab"):
        self.name = name
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.freqs: Counter = Counter()

        # Initialize special tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build_vocab(self, tokenized_sentences: List[List[str]], min_freq: int = 2):
        """Build vocabulary from tokenized corpus with minimum frequency cutoff."""
        for tokens in tokenized_sentences:
            self.freqs.update(tokens)

        cur_idx = len(SPECIAL_TOKENS)
        for token, count in self.freqs.most_common():
            if count >= min_freq and token not in self.token2idx:
                self.token2idx[token] = cur_idx
                self.idx2token[cur_idx] = token
                cur_idx += 1

    def encode(self, tokens: List[str], add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Convert list of tokens into indices with optional SOS and EOS."""
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        indices.extend([self.token2idx.get(t, UNK_IDX) for t in tokens])
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> List[str]:
        """Convert list of indices back into tokens."""
        tokens = []
        for idx in indices:
            token = self.idx2token.get(idx, UNK_TOKEN)
            if remove_special and token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
                continue
            tokens.append(token)
        return tokens

    def __len__(self) -> int:
        return len(self.token2idx)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.token2idx.get(item, UNK_IDX)
        elif isinstance(item, int):
            return self.idx2token.get(item, UNK_TOKEN)
        raise TypeError(f"Invalid key type: {type(item)}")


class TranslationDataset(Dataset):
    """PyTorch Dataset holding pre-tokenized and numericalized English-German pairs."""

    def __init__(
        self,
        en_sentences: List[str],
        de_sentences: List[str],
        src_vocab: Vocab,
        trg_vocab: Vocab,
        max_len: int = 50,
    ):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.samples = []

        for en_text, de_text in zip(en_sentences, de_sentences):
            en_tokens = tokenize_en(en_text)
            de_tokens = tokenize_de(de_text)

            # Filter out empty or excessively long sentences
            if 0 < len(en_tokens) <= max_len and 0 < len(de_tokens) <= max_len:
                src_ids = src_vocab.encode(en_tokens, add_sos=True, add_eos=True)
                trg_ids = trg_vocab.encode(de_tokens, add_sos=True, add_eos=True)
                self.samples.append({
                    "src_raw": en_text,
                    "trg_raw": de_text,
                    "src_tokens": en_tokens,
                    "trg_tokens": de_tokens,
                    "src_ids": torch.tensor(src_ids, dtype=torch.long),
                    "trg_ids": torch.tensor(trg_ids, dtype=torch.long),
                    "src_len": len(src_ids),
                    "trg_len": len(trg_ids),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    """
    Collate function to dynamically pad source and target sequences to batch maximum length.
    Returns:
        src_padded: Tensor of shape [batch_size, src_len]
        trg_padded: Tensor of shape [batch_size, trg_len]
        src_lens: Tensor of shape [batch_size]
        trg_lens: Tensor of shape [batch_size]
        raw_en: List[str]
        raw_de: List[str]
    """
    src_list = [item["src_ids"] for item in batch]
    trg_list = [item["trg_ids"] for item in batch]
    src_lens = torch.tensor([item["src_len"] for item in batch], dtype=torch.long)
    trg_lens = torch.tensor([item["trg_len"] for item in batch], dtype=torch.long)

    raw_en = [item["src_raw"] for item in batch]
    raw_de = [item["trg_raw"] for item in batch]

    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=PAD_IDX
    )
    trg_padded = torch.nn.utils.rnn.pad_sequence(
        trg_list, batch_first=True, padding_value=PAD_IDX
    )

    return {
        "src": src_padded,
        "trg": trg_padded,
        "src_lens": src_lens,
        "trg_lens": trg_lens,
        "raw_en": raw_en,
        "raw_de": raw_de,
    }


def load_and_prepare_data(
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    test_limit: Optional[int] = None,
    min_freq: int = 2,
    max_len: int = 40,
) -> Tuple[Vocab, Vocab, Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Loads bentrevett/multi30k dataset from HuggingFace, tokenizes, and constructs
    source (EN) and target (DE) vocabularies.
    """
    ds = load_dataset("bentrevett/multi30k")

    train_en = ds["train"]["en"][:train_limit] if train_limit else ds["train"]["en"]
    train_de = ds["train"]["de"][:train_limit] if train_limit else ds["train"]["de"]

    val_en = ds["validation"]["en"][:val_limit] if val_limit else ds["validation"]["en"]
    val_de = ds["validation"]["de"][:val_limit] if val_limit else ds["validation"]["de"]

    test_en = ds["test"]["en"][:test_limit] if test_limit else ds["test"]["en"]
    test_de = ds["test"]["de"][:test_limit] if test_limit else ds["test"]["de"]

    # Build vocabularies solely from training set to avoid data leakage
    src_tokenized_train = [tokenize_en(s) for s in train_en]
    trg_tokenized_train = [tokenize_de(s) for s in train_de]

    src_vocab = Vocab(name="en_vocab")
    trg_vocab = Vocab(name="de_vocab")

    src_vocab.build_vocab(src_tokenized_train, min_freq=min_freq)
    trg_vocab.build_vocab(trg_tokenized_train, min_freq=min_freq)

    raw_data = {
        "train": (train_en, train_de),
        "validation": (val_en, val_de),
        "test": (test_en, test_de),
    }

    return src_vocab, trg_vocab, raw_data


def get_dataloaders(
    src_vocab: Vocab,
    trg_vocab: Vocab,
    raw_data: Dict[str, Tuple[List[str], List[str]]],
    batch_size: int = 64,
    max_len: int = 40,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test PyTorch DataLoaders."""
    train_en, train_de = raw_data["train"]
    val_en, val_de = raw_data["validation"]
    test_en, test_de = raw_data["test"]

    train_ds = TranslationDataset(train_en, train_de, src_vocab, trg_vocab, max_len=max_len)
    val_ds = TranslationDataset(val_en, val_de, src_vocab, trg_vocab, max_len=max_len)
    test_ds = TranslationDataset(test_en, test_de, src_vocab, trg_vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
