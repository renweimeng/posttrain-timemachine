# Actual L2 CPU reference fixture

`native.json`, `native-base.json` and the numeric capture are produced by the actual
pinned TRL 0.24.0 trainer using `examples/quickstart-trl/train_tiny.py --deep
--capture-update --loss dapo`. The model is randomly initialized, locally authored
GPT-2 with a local tokenizer and tiny local prompts. It is not a pretrained model
or an independent accuracy benchmark. Native loss and after-step weights are
recorded directly; PTM does not generate the expected reference weights.

The original native loss is -8.940696716308594e-8; four actual completion token
observations constitute the first optimizer batch. A separate complete pre-pointer
snapshot is retained because patched replay refers to this exact batch snapshot.
`safetensors` files contain only tiny local numeric tensors, never Python pickle.

`synthetic.json` is separately labeled synthetic and exercises a 5000-token canvas;
it must not be mistaken for training output. The automated live test regenerates
actual captures for all four loss types plus native AdamW and SGD.
