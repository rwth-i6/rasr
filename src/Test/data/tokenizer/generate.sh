#!/usr/bin/env bash
# Generates the tokenizer test files from corpus.txt. Expected values in src/Test/Tokenizer_*.cc depend on these files.
#
# Tools used:
#   subword-nmt 0.3.8 (pip), create-py-vocab.py from https://github.com/rwth-i6/subword-nmt (5015a45),
#   spm_train from sentencepiece 0.2.0, python package tokenizers 0.23.2
#
# Usage: I6_SUBWORD_NMT=<path to rwth-i6/subword-nmt checkout> ./generate.sh
set -euo pipefail
cd "$(dirname "$0")"

# subword-nmt BPE: merge operations, plain vocabulary with counts and RETURNN python dict vocabulary
subword-nmt learn-bpe --symbols 30 < corpus.txt > bpe.codes
subword-nmt apply-bpe --codes bpe.codes < corpus.txt | subword-nmt get-vocab > bpe.vocab
python3 "${I6_SUBWORD_NMT}/create-py-vocab.py" --txt corpus.txt --bpe bpe.codes --unk "<unk>" --out bpe.vocab.py

# SentencePiece unigram model
spm_train --input=corpus.txt --model_prefix=sentencepiece --vocab_size=32 --character_coverage=1.0 --normalization_rule_name=identity --minloglevel=2
rm sentencepiece.vocab

# HuggingFace byte-level BPE as used by GPT-style LLMs, with BOS added by the post-processor
python3 - <<'PY'
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=300,
    special_tokens=["<s>", "</s>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tokenizer.train(["corpus.txt"], trainer)
tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", tokenizer.token_to_id("<s>"))])
tokenizer.save("tokenizer.json", pretty=True)
PY
