"""
extract_hidden_states.py

Extracts hidden states from a specified GPT-2 layer for every token
in a corpus, returning a numpy array of shape (N, d_model).

Usage:
    python extract_hidden_states.py \
        --corpus corpus.txt \
        --layer 5 \
        --output states.npy \
        --tokens tokens.txt

Requirements:
    pip install torch transformers numpy tqdm
"""

import argparse
import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model
from tqdm import tqdm


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "gpt2"          # gpt2 | gpt2-medium | gpt2-large | gpt2-xl
DEFAULT_LAYER   = 5               # 0-indexed; GPT-2 small has 12 layers (0–11)
DEFAULT_BATCH   = 8               # sentences per forward pass
MAX_SEQ_LEN     = 1024            # GPT-2's hard context limit
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ── Corpus loading ────────────────────────────────────────────────────────────

def load_corpus(path: str) -> list[str]:
    """
    Load corpus from a plain-text file.
    Each non-empty line is treated as one sentence / document chunk.
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines):,} sentences from '{path}'")
    return lines


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str = DEFAULT_MODEL):
    """
    Load a frozen GPT-2 model and tokenizer.
    GPT-2 hidden size:
        gpt2         → 768   (12 layers)
        gpt2-medium  → 1024  (24 layers)
        gpt2-large   → 1280  (36 layers)
        gpt2-xl      → 1600  (48 layers)
    """
    print(f"Loading '{model_name}' onto {DEVICE} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT-2 has no pad token by default; use eos as pad
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(
        model_name,
        output_hidden_states=True,   # return all layer outputs
    )
    model.eval()
    model.to(DEVICE)

    # Freeze all parameters — we only need forward passes
    for param in model.parameters():
        param.requires_grad_(False)

    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded | d_model={d_model} | layers={n_layers}")
    return tokenizer, model, d_model


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_layer(
    sentences : list[str],
    tokenizer,
    model,
    layer     : int,
    batch_size: int = DEFAULT_BATCH,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract hidden states from `layer` for every token in `sentences`.

    Args:
        sentences:  List of raw text strings.
        tokenizer:  HuggingFace GPT-2 tokenizer.
        model:      Frozen GPT-2Model with output_hidden_states=True.
        layer:      Which layer to extract (0 = embedding layer,
                    1-12 = transformer layers for GPT-2 small).
        batch_size: Number of sentences per forward pass.

    Returns:
        states : np.ndarray of shape (N_tokens, d_model), float32
        tokens : list of N_tokens decoded token strings (for inspection)

    Notes:
        - Special tokens (BOS / EOS / PAD) are excluded.
        - Sequences longer than MAX_SEQ_LEN are truncated.
        - hidden_states[0] is the embedding layer (before any attention).
          hidden_states[l] for l >= 1 is after transformer block l.
          To match "layer 5 after 5 transformer blocks" pass layer=5.
    """
    all_states : list[np.ndarray] = []
    all_tokens : list[str]        = []

    n_batches = (len(sentences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size),
                      total=n_batches, desc=f"Extracting layer {layer}"):

            batch_sents = sentences[i : i + batch_size]

            # ── Tokenise ──────────────────────────────────────────────────
            encoding = tokenizer(
                batch_sents,
                return_tensors      = "pt",
                padding             = True,
                truncation          = True,
                max_length          = MAX_SEQ_LEN,
                return_attention_mask = True,
            )
            input_ids      = encoding["input_ids"].to(DEVICE)       # (B, L)
            attention_mask = encoding["attention_mask"].to(DEVICE)  # (B, L)

            # ── Forward pass ──────────────────────────────────────────────
            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )

            # hidden_states is a tuple of length (n_layers + 1):
            #   index 0  → embedding layer output
            #   index l  → output of transformer block l  (l = 1 … 12)
            hidden_states = outputs.hidden_states  # tuple of (B, L, d_model)
            layer_output  = hidden_states[layer]   # (B, L, d_model)

            # ── Collect real (non-pad) tokens ─────────────────────────────
            for b in range(len(batch_sents)):
                # attention_mask == 1 for real tokens, 0 for padding
                real_mask = attention_mask[b].bool()  # (L,)

                # hidden state vectors for real tokens only
                h = layer_output[b][real_mask]         # (n_real, d_model)
                all_states.append(h.cpu().float().numpy())

                # decode each real token id to a string for later inspection
                real_ids = input_ids[b][real_mask].tolist()
                decoded  = [tokenizer.decode([tid]) for tid in real_ids]
                all_tokens.extend(decoded)

    states = np.concatenate(all_states, axis=0)  # (N_total, d_model)
    assert states.shape[0] == len(all_tokens), "Shape mismatch — bug!"
    print(f"\nExtracted {states.shape[0]:,} token vectors | "
          f"shape={states.shape} | dtype={states.dtype}")
    return states, all_tokens


# ── Persistence ───────────────────────────────────────────────────────────────

def save_outputs(
    states     : np.ndarray,
    tokens     : list[str],
    states_path: str,
    tokens_path: str,
) -> None:
    """Save hidden states as .npy and tokens as a newline-delimited text file."""
    np.save(states_path, states)
    print(f"Saved states  → '{states_path}'  {states.shape}")

    with open(tokens_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
    print(f"Saved tokens  → '{tokens_path}'  ({len(tokens):,} entries)")


def load_outputs(
    states_path: str,
    tokens_path: str,
) -> tuple[np.ndarray, list[str]]:
    """Reload previously saved outputs."""
    states = np.load(states_path)
    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = f.read().splitlines()
    print(f"Loaded states={states.shape}  tokens={len(tokens):,}")
    return states, tokens


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract GPT-2 hidden states for POS-GMM pipeline"
    )
    p.add_argument("--corpus",  required=True,
                   help="Path to plain-text corpus (one sentence per line)")
    p.add_argument("--layer",   type=int, default=DEFAULT_LAYER,
                   help=f"Layer index to extract (default: {DEFAULT_LAYER})")
    p.add_argument("--model",   default=DEFAULT_MODEL,
                   help=f"GPT-2 variant (default: {DEFAULT_MODEL})")
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH,
                   help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--output",  default="states.npy",
                   help="Output path for hidden states .npy (default: states.npy)")
    p.add_argument("--tokens",  default="tokens.txt",
                   help="Output path for token strings (default: tokens.txt)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Validate layer index before loading the model
    layer_limits = {
        "gpt2": 12, "gpt2-medium": 24,
        "gpt2-large": 36, "gpt2-xl": 48,
    }
    max_layer = layer_limits.get(args.model, 12)
    if not (0 <= args.layer <= max_layer):
        raise ValueError(
            f"Layer {args.layer} out of range for '{args.model}' "
            f"(valid: 0-{max_layer})"
        )

    sentences           = load_corpus(args.corpus)
    tokenizer, model, _ = load_model(args.model)

    states, tokens = extract_layer(
        sentences = sentences,
        tokenizer = tokenizer,
        model     = model,
        layer     = args.layer,
        batch_size= args.batch,
    )

    save_outputs(states, tokens, args.output, args.tokens)


# ── Library interface (import without CLI) ────────────────────────────────────

def extract_from_text(
    texts     : list[str],
    layer     : int  = DEFAULT_LAYER,
    model_name: str  = DEFAULT_MODEL,
    batch_size: int  = DEFAULT_BATCH,
) -> tuple[np.ndarray, list[str]]:
    """
    Convenience function for use as a library.

    Example:
        from extract_hidden_states import extract_from_text

        states, tokens = extract_from_text(
            texts=["The cat sat.", "A dog ran away."],
            layer=5,
        )
        # states.shape == (N_tokens, 768)
    """
    tokenizer, model, _ = load_model(model_name)
    return extract_layer(texts, tokenizer, model, layer, batch_size)


if __name__ == "__main__":
    main()
