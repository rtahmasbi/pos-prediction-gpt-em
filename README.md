# pos_prediction_gpt_em




```py
from extract_hidden_states import extract_from_text

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