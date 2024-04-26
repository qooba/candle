#!/bin/bash

cargo run --example mdberta --release -- --model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --premise "Dziecko bawi się na placu zabaw." --hypotesis  "Na dworze znajduje się dziecko."
#cargo run --example mdberta --release -- --model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --premise "Kobieta czyta książkę w cichej kawiarni." --hypotesis  "Kobieta się uczy."

#cargo run --example mdberta --release -- --model-id MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --premise "Kot śpi na parapecie." --hypotesis  "Wszystkie koty są teraz na zewnątrz."