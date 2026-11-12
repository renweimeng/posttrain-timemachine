"""Regenerate golden outputs by executing the actual pinned trainer; never call PTM's kernel."""
import importlib.util
from pathlib import Path

from ptm.storage import atomic_write, canonical

root = Path(__file__).resolve().parents[1]
source = root/'examples/quickstart-trl/train_tiny.py'
spec = importlib.util.spec_from_file_location('ptm_tiny', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for scale in ['group','none','batch']:
    output = root/'output'/f'trl-{scale}'
    fixture = module.train(output, root/'.ptm', scale=scale, steps=1 if scale=='batch' else 3)
    atomic_write(root/'fixtures/trl'/f'{scale}.json', canonical(fixture))
    print(f'Captured actual trainer output: {scale}, {len(fixture["batches"])} batches')
