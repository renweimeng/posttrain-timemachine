"""Create a fresh, isolated browser-test store from synthetic and actual captured fixtures."""
import shutil
from pathlib import Path

from ptm.demo import create_demo
from ptm.importer import read_snapshot
from ptm.storage import Store

project = Path(__file__).resolve().parents[1]
root = project/'tmp/e2e-data'
if root.exists():
    shutil.rmtree(root)
store = Store(root)
create_demo(store)
store.commit(read_snapshot(project/'fixtures/trl/captured-events.jsonl'))
store.commit(read_snapshot(project/'fixtures/l2/synthetic.json'))

store.commit(read_snapshot(project/'fixtures/l2/native-base.json'))
store.commit(read_snapshot(project/'fixtures/l2/native.json'))
shutil.copytree(project/'fixtures/l2/update-captures',root/'update-captures',dirs_exist_ok=True)
