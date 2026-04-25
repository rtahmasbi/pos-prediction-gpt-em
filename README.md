# pos-prediction-gpt-em

```sh
git clone https://github.com/rtahmasbi/pos-prediction-gpt-em.git
cd pos-prediction-gpt-em/

```

# get_texts
python get_texts.py


# extract_hidden_states
```sh

nohup python extract_hidden_states.py \
    --corpus /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/climbmix.txt.gz \
    --layer 5 \
    --output /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/states.npy \
    --tokens /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/tokens.txt &
```

```py
from src.extract_hidden_states import extract_from_text

states, tokens = extract_from_text(
    texts=["The cat sat.", "A dog ran away."],
    layer=5,
)
# states.shape == (N_tokens, 768)
# tokens == ['The', ' cat', ' sat', '.', 'A', ' dog', ' ran', ' away', '.']
```


```py
from extract_hidden_states import load_outputs

states, tokens = load_outputs("states.npy", "tokens.txt")

```



| Representation | What it captures |
|---|---|
| e(x_t) | Token identity only — no context |
| h_t^(1–2) | Local surface patterns — character-level, morphology |
| h_t^(3–6) | Syntactic structure — POS, dependencies (peak layer for POS probing) |
| h_t^(7–10) | Semantic roles, coreference |
| h_t^(11–12) | Task-specific, long-range semantic context |



# train
```sh
# Run EM on extracted hidden states
python gmm_em.py \
    --states /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/states.npy \
    --tokens /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/tokens.txt \
    --k 15 \
    --n-init 5 \
    --output /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/gmm.npz


```

As a library

```py
# load and inspect
from gmm_em import load_gmm, decode
result = load_gmm("/media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/gmm.npz")
tags   = decode(result["gamma"])   # (N,) hard tag per token
```


## Predict
```sh
python predict_pos.py \
    --gmm /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/gmm.npz \
    --sentences "The cat sat on the mat." "A dog ran away."

Sentence: 'The cat sat on the mat.'
  Token          Tag          Conf
  ─────────────  ────────────  ──────
   The           C3           0.981
   cat           C1           0.874
   sat           C7           0.923
  ...


# With a label map
python predict_pos.py \
    --gmm /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/gmm.npz \
    --label-map label_map.json \
    --sentences "The cat sat on the mat."

label_map.json:
{"0": "DET", "1": "NOUN", "2": "VERB", "3": "ADJ"}


# from a file
python predict_pos.py \
    --gmm /media/HD2/RASOOL/OUTPUTS/pos-pred-gpt-em/gmm.npz \
    --label-map label_map.json \
    --input-file sentences.txt \
    --output-file predictions.json

```


As a library
```py
from predict_pos import predict

results = predict(
    sentences=["The cat sat on the mat."],
    gmm_path="gmm.npz",
    label_map={3: "DET", 1: "NOUN", 7: "VERB"},
)
# results[0] is a list of per-token dicts:
# [{'token': ' The', 'component': 3, 'label': 'DET', 'confidence': 0.981, ...}, ...]
```
