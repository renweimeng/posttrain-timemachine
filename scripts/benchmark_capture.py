import contextlib
import importlib.util
import io
import json
import statistics
from pathlib import Path

from ptm.storage import atomic_write, canonical

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ptm_tiny_benchmark',root/'examples/quickstart-trl/train_tiny.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results=[]
for i in range(5):
    pair={}
    # Alternate ordering to reduce systematic warm-cache bias.
    for enabled in ([False,True] if i%2==0 else [True,False]):
        with contextlib.redirect_stdout(io.StringIO()):
            result=module.train(root/'tmp/capture-benchmark'/f'{i}-{enabled}',
                root/'tmp/capture-benchmark/store',steps=40,capture_on=enabled,sample_probability=.1)
        pair['capture_on_seconds' if enabled else 'baseline_seconds']=(
            result['provenance']['seconds'] if enabled else result['seconds'])
    results.append(pair)
baseline=statistics.median(r['baseline_seconds'] for r in results)
captured=statistics.median(r['capture_on_seconds'] for r in results)
paired_overheads=[(r['capture_on_seconds']/r['baseline_seconds']-1)*100 for r in results]
overhead=statistics.median(paired_overheads)
report={'model':'tiny random GPT2 n_embd=32 n_layer=1 CPU, one Torch thread',
    'trl':'0.24.0','steps':40,'seed':42,'group_size':4,'sampling_probability':.1,
    'raw_pairs':results,'median_baseline_seconds':baseline,'median_capture_seconds':captured,
    'paired_overhead_percent':paired_overheads,'median_overhead_percent':overhead,
    'ratio_of_medians_overhead_percent':(captured/baseline-1)*100,
    'target_percent':3,'target_met_in_this_toy_experiment':overhead<=3,
    'limits':['Small CPU model only; not evidence of production GPU capture overhead.',
              'Timing covers trainer.train and capture writer drain, excludes post-training Parquet compaction.',
              'Alternating run order; five paired repetitions; environment in benchmark.json.']}
atomic_write(root/'output/capture-benchmark.json',canonical(report))
print(json.dumps(report,indent=2))
