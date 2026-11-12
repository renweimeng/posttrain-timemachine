import threading
import time
from pathlib import Path

from ptm.demo import create_demo
from ptm.jobs import Jobs
from ptm.replay import AnalysisRequest, load_spec
from ptm.storage import Store, atomic_write, canonical


def test_interrupted_job_recovery_and_queued_cancellation(tmp_path):
    store = Store(tmp_path)
    create_demo(store)
    atomic_write(store.path('jobs','ana_interrupted.json'),canonical({'analysis_id':'ana_interrupted','status':'running'}))
    jobs = Jobs(store)
    assert jobs.get('ana_interrupted')['error']['code'] == 'PTM_JOB_INTERRUPTED'
    blocked, release = threading.Event(), threading.Event()

    def occupy():
        blocked.set()
        release.wait(5)

    jobs.pool.submit(occupy)
    assert blocked.wait(1)
    request = AnalysisRequest(run_id='demo-format-truth',reward_spec=load_spec(
        Path(__file__).resolve().parents[1]/'examples/format-over-truth/patch.yaml'))
    queued = jobs.submit(request)
    assert jobs.cancel(queued['analysis_id'])['status'] == 'cancellation_requested'
    release.set()
    for _ in range(100):
        if jobs.get(queued['analysis_id'])['status'] == 'cancelled':
            break
        time.sleep(.01)
    assert jobs.get(queued['analysis_id'])['status'] == 'cancelled'
    assert not store.path('analyses',queued['analysis_id']).exists()
    jobs.close()
