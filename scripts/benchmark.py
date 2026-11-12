"""Reproducible metadata-query and DSL microbenchmarks; no trainer or token throughput claim."""
import argparse
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ptm.demo import create_demo
from ptm.query import Repository
from ptm.replay import evaluate, load_spec
from ptm.schemas import Completion
from ptm.storage import Store, atomic_write, canonical


def timed(fn, repeats=20):
    raw = []
    for _ in range(repeats):
        start=time.perf_counter()
        fn()
        raw.append((time.perf_counter()-start)*1000)
    return {'raw_ms':raw,'median_ms':statistics.median(raw),'p95_ms':float(np.percentile(raw,95))}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--completions',type=int,default=1_000_000)
    parser.add_argument('--output',type=Path,default=Path('output/benchmark.json'))
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    data=root/'tmp/benchmark'
    data.mkdir(parents=True,exist_ok=True)
    parquet=data/'metadata.parquet'
    n=args.completions
    schema=pa.schema([('completion_id',pa.string()),('group_id',pa.string()),('sequence_index',pa.int32()),
                     ('reward_total',pa.float64()),('text_hash',pa.string())])
    raw_bytes=0
    started=time.perf_counter()
    with pq.ParquetWriter(parquet,schema,compression='zstd') as writer:
        for offset in range(0,n,20_000):
            rows=[{'completion_id':f'cmp-{i:09d}','group_id':f'group-{i//4:09d}',
                   'sequence_index':i%4,'reward_total':(i%4)/3,'text_hash':f'{i:064x}'}
                  for i in range(offset,min(offset+20_000,n))]
            table=pa.Table.from_pylist(rows,schema=schema)
            raw_bytes+=table.nbytes
            writer.write_table(table)
    generation_seconds=time.perf_counter()-started
    with duckdb.connect() as con:
        def lookup():
            rows=con.execute('SELECT completion_id,reward_total FROM read_parquet(?) WHERE group_id=? ORDER BY sequence_index',
                             [str(parquet),f'group-{n//8:09d}']).fetchall()
            assert len(rows)==4
        cold=timed(lookup,1)
        warm=timed(lookup,30)
        overview=timed(lambda:con.execute('SELECT count(*),avg(reward_total) FROM read_parquet(?)',[str(parquet)]).fetchall(),30)
    # Full production repository, separately measured on a canonical fixture.
    store=Store(data/'canonical')
    create_demo(store)
    repo=Repository(store)
    repository=timed(lambda:repo.groups('demo-format-truth',limit=50),20)
    cs=[Completion(completion_id=f'c-{i}',group_id=f'g-{i//4}',sequence_index=i%4,reward_total=0,
        inputs={'correctness_score':float(i%2),'format_score':float(i%3)/2}) for i in range(100_000)]
    spec=load_spec(root/'examples/format-over-truth/patch.yaml')
    scoring=timed(lambda:evaluate(cs,spec),5)
    startup=[]
    for _ in range(3):
        start=time.perf_counter()
        subprocess.run([str(root/'.venv/bin/python'),'-m','ptm','demo','--no-serve','--root',str(data/'startup')],
                       check=True,stdout=subprocess.DEVNULL)
        startup.append(time.perf_counter()-start)
    memory=subprocess.run(['sysctl','-n','hw.memsize'],capture_output=True,text=True).stdout.strip() if platform.system()=='Darwin' else 'unknown'
    result={'environment':{'platform':platform.platform(),'machine':platform.machine(),'python':platform.python_version(),
        'cpu':platform.processor(),'ram_bytes':memory,'storage':'local workspace filesystem; SSD model not queried',
        'duckdb':duckdb.__version__,'numpy':np.__version__,'pyarrow':pa.__version__,'trainer_concurrent':False},
        'metadata':{'completion_count':n,'group_count':n//4,'token_count':0,'capture_fraction':1.0,
            'uncompressed_arrow_bytes':raw_bytes,'parquet_bytes':parquet.stat().st_size,
            'compression_ratio':raw_bytes/parquet.stat().st_size,'generation_seconds':generation_seconds,
            'synthetic':True,'index':'Parquet row-group statistics; no persistent DuckDB index'},
        'million_metadata_group_lookup_cold':cold,'million_metadata_group_lookup_warm':warm,
        'million_metadata_overview_warm':overview,'production_repository_384_completions':repository,
        'dsl_100k_completions':scoring,'demo_startup_seconds':startup,
        'limits':['Million-row benchmark is a metadata microbenchmark, not full API/React first-screen P95.',
                  'DSL timing includes vectorized scoring and constructing per-completion component dicts, excludes import/artifact persistence.',
                  'No tokens, LLM-as-Judge, GPU throughput, or production capture overhead measured here.',
                  'Token canvas scrolling is measured separately in output/token-canvas-performance.json; not included in this microbenchmark.']}
    atomic_write(args.output,canonical(result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
