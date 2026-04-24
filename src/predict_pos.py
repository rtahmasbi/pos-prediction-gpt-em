"""
predict_pos.py

Predict unsupervised POS tags for new sentences using:
  1. A frozen GPT-2 model  (extracts hidden states at a given layer)
  2. A trained GMM         (assigns each hidden state to a POS cluster)

The output is a component index (0 … K-1) per token.  Component indices have
no intrinsic linguistic label — they are assigned meaning post-hoc by looking
at which ground-truth POS tag dominates each cluster (see --label-map).

Usage
-----
# Basic — print component indices
python predict_pos.py \
    --gmm gmm.npz \
    --sentences "The cat sat on the mat." "A dog ran away."

# With a label map (component_index -> POS string)
python predict_pos.py \
    --gmm gmm.npz \
    --label-map label_map.json \
    --sentences "The cat sat on the mat."

# Read sentences from a file, write results to JSON
python predict_pos.py \
    --gmm  gmm.npz \
    --label-map label_map.json \
    --input-file sentences.txt \
    --output-file predictions.json

Requirements
------------
    pip install torch transformers numpy
"""

import argparse
import json
import os
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2Model


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_MODEL  = "gpt2"
DEFAULT_LAYER  = 5
MAX_SEQ_LEN    = 1024
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ── Model loading (reuse pattern from extract_hidden_states.py) ───────────────

def load_model(model_name: str = DEFAULT_MODEL):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2Model.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    model.to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)

    return tokenizer, model


# ── GMM loading ───────────────────────────────────────────────────────────────

def load_gmm(path: str) -> dict:
    """Load the trained GMM parameters saved by gmm_em.py."""
    data = np.load(path)
    gmm  = {
        "pi"    : data["pi"].astype(np.float64),      # (K,)
        "mu"    : data["mu"].astype(np.float64),      # (K, d)
        "sigma2": data["sigma2"].astype(np.float64),  # (K,)
    }
    K, d = gmm["mu"].shape
    print(f"Loaded GMM  K={K}  d={d}  from '{path}'")
    return gmm


# ── Label map ─────────────────────────────────────────────────────────────────

def load_label_map(path: str) -> dict[int, str]:
    """
    Load a JSON file mapping component index (int) → POS string.

    Example label_map.json:
        {"0": "DET", "1": "NOUN", "2": "VERB", "3": "ADJ", ...}

    This map is built post-hoc by inspecting which ground-truth POS tag
    dominates each cluster (many-to-one mapping).  It is optional — if not
    provided, raw component indices are returned.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    print(f"Loaded label map ({len(label_map)} entries) from '{path}'")
    return label_map


# ── Hidden state extraction for a single batch of sentences ──────────────────

def extract_hidden_states(
    sentences : list[str],
    tokenizer,
    model,
    layer     : int,
) -> tuple[list[np.ndarray], list[list[str]]]:
    """
    Run GPT-2 on `sentences` and return hidden states + token strings.

    Returns
    -------
    states_per_sent : list of (T_i, d) float64 arrays, one per sentence
    tokens_per_sent : list of T_i token strings,       one per sentence

    Token strings are the raw subword pieces produced by the GPT-2 tokenizer
    (e.g. ' cat', 'Ġcat').  GPT-2 uses byte-pair encoding so multi-character
    tokens are common.
    """
    encoding = tokenizer(
        sentences,
        return_tensors       = "pt",
        padding              = True,
        truncation           = True,
        max_length           = MAX_SEQ_LEN,
        return_attention_mask= True,
    )
    input_ids      = encoding["input_ids"].to(DEVICE)       # (B, L)
    attention_mask = encoding["attention_mask"].to(DEVICE)  # (B, L)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # hidden_states[layer] has shape (B, L, d)
    layer_output = outputs.hidden_states[layer]             # (B, L, d)

    states_per_sent : list[np.ndarray] = []
    tokens_per_sent : list[list[str]]  = []

    for b in range(len(sentences)):
        real_mask = attention_mask[b].bool()                # (L,)

        # hidden states for real (non-pad) tokens
        h = layer_output[b][real_mask].cpu().float().numpy()  # (T_i, d)
        states_per_sent.append(h.astype(np.float64))

        # decode each real token id to its subword string
        real_ids = input_ids[b][real_mask].tolist()
        toks     = [tokenizer.decode([tid]) for tid in real_ids]
        tokens_per_sent.append(toks)

    return states_per_sent, tokens_per_sent


# ── GMM inference ─────────────────────────────────────────────────────────────

LOG_2PI = np.log(2.0 * np.pi)


def log_gaussian_isotropic(
    states : np.ndarray,  # (T, d)
    mu     : np.ndarray,  # (K, d)
    sigma2 : np.ndarray,  # (K,)
) -> np.ndarray:          # (T, K)
    """
    Log-density log N(h_t | mu_k, sigma_k^2 I) for all t, k.
    Identical to the function in gmm_em.py — reproduced here so this
    file is self-contained and does not depend on gmm_em.py at runtime.
    """
    T, d = states.shape

    h_sq  = (states ** 2).sum(axis=1, keepdims=True)       # (T, 1)
    cross = -2.0 * (states @ mu.T)                         # (T, K)
    mu_sq = (mu ** 2).sum(axis=1, keepdims=True).T         # (1, K)

    sq_dist  = h_sq + cross + mu_sq                        # (T, K)
    log_norm = (
        - 0.5 * d * LOG_2PI
        - 0.5 * d * np.log(sigma2)[None, :]
        - 0.5 * sq_dist / sigma2[None, :]
    )
    return log_norm                                         # (T, K)


def gmm_predict(
    states : np.ndarray,  # (T, d)
    gmm    : dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each hidden state to a GMM component.

    Returns
    -------
    tags  : (T,) int    hard assignment  z_t = argmax_k gamma_t(k)
    gamma : (T, K) float soft responsibilities  gamma_t(k) = p(z_t=k | h_t)
    """
    pi     = gmm["pi"]      # (K,)
    mu     = gmm["mu"]      # (K, d)
    sigma2 = gmm["sigma2"]  # (K,)

    log_pi    = np.log(pi + 1e-300)                         # (K,)
    log_gauss = log_gaussian_isotropic(states, mu, sigma2)  # (T, K)
    log_joint = log_pi[None, :] + log_gauss                 # (T, K)

    # logsumexp normalisation
    log_max   = log_joint.max(axis=1, keepdims=True)        # (T, 1)
    log_sum   = log_max[:, 0] + np.log(
                    np.exp(log_joint - log_max).sum(axis=1)
                )                                           # (T,)
    log_gamma = log_joint - log_sum[:, None]                # (T, K)
    gamma     = np.exp(log_gamma)                           # (T, K)
    tags      = gamma.argmax(axis=1)                        # (T,)

    return tags, gamma


# ── Result formatting ─────────────────────────────────────────────────────────

def format_sentence(
    tokens   : list[str],
    tags     : np.ndarray,           # (T,)
    gamma    : np.ndarray,           # (T, K)
    label_map: dict[int, str] | None,
    top_k    : int = 3,
) -> list[dict]:
    """
    Build a list of per-token result dicts for one sentence.

    Each dict contains:
        token      : subword string
        component  : hard-assigned component index
        label      : POS label string (if label_map provided, else None)
        confidence : probability of the winning component  max_k gamma_t(k)
        top_k      : list of {component, label, prob} for the top_k components
    """
    K       = gamma.shape[1]
    results = []

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        g = gamma[i]                                       # (K,)

        # top-k components by responsibility
        top_idx  = g.argsort()[::-1][:top_k]
        top_list = [
            {
                "component": int(idx),
                "label"    : label_map.get(int(idx)) if label_map else None,
                "prob"     : float(g[idx]),
            }
            for idx in top_idx
        ]

        results.append({
            "token"     : tok,
            "component" : int(tag),
            "label"     : label_map.get(int(tag)) if label_map else None,
            "confidence": float(g[tag]),
            "top_k"     : top_list,
        })

    return results


def print_sentence(
    sentence : str,
    results  : list[dict],
    use_label: bool,
) -> None:
    """Pretty-print one sentence with per-token POS assignments."""
    col_w = max(len(r["token"]) for r in results) + 2
    tag_w = 12

    print(f"\n  Sentence: {sentence!r}")
    print(f"  {'Token':<{col_w}} {'Tag':<{tag_w}} {'Conf':>6}")
    print(f"  {'─'*col_w} {'─'*tag_w} {'─'*6}")

    for r in results:
        tag_str = (r["label"] if use_label and r["label"]
                   else f"C{r['component']}")
        print(f"  {r['token']:<{col_w}} {tag_str:<{tag_w}} "
              f"{r['confidence']:>6.3f}")


# ── High-level predict API ────────────────────────────────────────────────────

def predict(
    sentences  : list[str],
    gmm_path   : str,
    model_name : str              = DEFAULT_MODEL,
    layer      : int              = DEFAULT_LAYER,
    label_map  : dict | None      = None,
    batch_size : int              = 16,
    verbose    : bool             = True,
) -> list[list[dict]]:
    """
    Full pipeline: text → hidden states → GMM → POS tags.

    Args:
        sentences  : List of raw text strings.
        gmm_path   : Path to the .npz file produced by gmm_em.py.
        model_name : GPT-2 variant used during training.
        layer      : Layer used during training (must match).
        label_map  : Optional {component_index: POS_string} dict.
        batch_size : Sentences per GPT-2 forward pass.
        verbose    : Print formatted output to stdout.

    Returns:
        List of per-sentence result lists (each element is a list of
        per-token dicts as produced by format_sentence()).
    """
    tokenizer, model = load_model(model_name)
    gmm              = load_gmm(gmm_path)

    all_results : list[list[dict]] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        states_batch, tokens_batch = extract_hidden_states(
            batch, tokenizer, model, layer
        )

        for sent, states, tokens in zip(batch, states_batch, tokens_batch):
            tags, gamma = gmm_predict(states, gmm)

            results = format_sentence(
                tokens, tags, gamma, label_map, top_k=3
            )
            all_results.append(results)

            if verbose:
                print_sentence(sent, results, use_label=label_map is not None)

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict unsupervised POS tags using GMM + GPT-2"
    )

    # Input
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sentences", nargs="+",
                     help="One or more sentences (quoted strings)")
    src.add_argument("--input-file",
                     help="Plain-text file, one sentence per line")

    # Model and GMM
    p.add_argument("--gmm",        required=True,
                   help="Path to trained GMM .npz file")
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help=f"GPT-2 variant (default: {DEFAULT_MODEL})")
    p.add_argument("--layer",      type=int, default=DEFAULT_LAYER,
                   help=f"Layer to extract (default: {DEFAULT_LAYER})")
    p.add_argument("--label-map",
                   help="JSON file mapping component index → POS string")

    # Output
    p.add_argument("--output-file",
                   help="Write predictions to this JSON file")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Sentences per GPT-2 forward pass (default: 16)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load sentences
    if args.sentences:
        sentences = args.sentences
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            sentences = [l.strip() for l in f if l.strip()]
    print(f"Predicting POS for {len(sentences)} sentence(s) ...")

    # Load optional label map
    label_map = load_label_map(args.label_map) if args.label_map else None

    # Run prediction
    all_results = predict(
        sentences  = sentences,
        gmm_path   = args.gmm,
        model_name = args.model,
        layer      = args.layer,
        label_map  = label_map,
        batch_size = args.batch_size,
        verbose    = True,
    )

    # Optionally save to JSON
    if args.output_file:
        output = [
            {
                "sentence": sent,
                "tokens"  : results,
            }
            for sent, results in zip(sentences, all_results)
        ]
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved predictions → '{args.output_file}'")


if __name__ == "__main__":
    main()