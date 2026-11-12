# Sample data provenance

The built-in arithmetic challenge is generated deterministically by `ptm/demo.py`
and contains no external datasets, model outputs or personal information. It is
released with the project under Apache-2.0.

The tiny CPU integration uses four manually written prompts, a local word-level
tokenizer and randomly initialized GPT-2 architecture. No pretrained weights
or downloaded corpus is used. Saved traces and numerical fixtures are generated
by the included training script. Their purpose is integration testing, not an
evaluation benchmark. Dependencies retain their respective upstream licenses.
