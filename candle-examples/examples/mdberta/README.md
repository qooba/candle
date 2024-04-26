# candle-deberta

Bert is a general large language model. In this example it can be used for two
different tasks:

- Compute sentence embeddings for a prompt.
- Compute similarities between a set of sentences.

## Zero shot classification

Bert is used to compute the sentence embeddings for a prompt. The model weights
are downloaded from the hub on the first run.

```bash

cargo run --example mdberta --release -- --model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --prompt "Chciałbym zrobić przelew krajowy"

```
