import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, analysis_identity, load_analysis, run_analysis
from ptm.storage import Store, atomic_write, canonical


class Jobs:
    def __init__(self, store: Store) -> None:
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ptm-analysis")
        self.lock = threading.RLock()
        self.cancelled: set[str] = set()
        parent = store.path("jobs")
        if parent.exists():
            for path in parent.glob("*.json"):
                if not path.resolve().is_relative_to(store.root):
                    continue
                job = json.loads(path.read_bytes())
                if job["status"] in {"queued", "running"}:
                    self.write(job["analysis_id"], "failed", error={**PTMError(
                        "PTM_JOB_INTERRUPTED", "Worker restarted; resubmit to retry").as_dict(), "retryable": True})

    def write(self, aid: str, status: str, **extra: Any) -> dict[str, Any]:
        path = self.store.path("jobs", aid + ".json")
        prior = json.loads(path.read_bytes()) if path.exists() else {}
        if status == "queued":
            prior.pop("error", None)
        data = {**prior, "analysis_id": aid, "status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra}
        atomic_write(path, canonical(data))
        return data

    def get(self, aid: str) -> dict[str, Any]:
        if self.store.path("analyses", aid).exists():
            return load_analysis(self.store, aid)
        path = self.store.path("jobs", aid + ".json")
        if not path.exists():
            raise PTMError("PTM_NOT_FOUND", "Analysis job not found", status=404)
        result: dict[str, Any] = json.loads(path.read_bytes())
        return result

    def submit(self, request: AnalysisRequest) -> dict[str, Any]:
        aid, _ = analysis_identity(self.store, request)
        # Resolve the snapshot now. A later import must not silently change a queued job's inputs.
        request = request.model_copy(update={"snapshot_id": self.store.manifest(
            request.run_id, request.snapshot_id)["snapshot_id"]})
        with self.lock:
            try:
                existing = self.get(aid)
                if existing["status"] in {"queued", "running", "succeeded"}:
                    return {"analysis_id": aid, "status": existing["status"]}
            except PTMError as exc:
                if exc.code != "PTM_NOT_FOUND":
                    raise
            self.cancelled.discard(aid)
            result = self.write(aid, "queued", phase="waiting", progress=0, request=request.model_dump(mode="json"))
            self.pool.submit(self.work, aid, request)
            return result

    def work(self, aid: str, request: AnalysisRequest) -> None:
        try:
            with self.lock:
                if aid in self.cancelled:
                    self.write(aid, "cancelled")
                    return
                self.write(aid, "running", phase="validate-score-statistics", progress=0.2)
            run_analysis(self.store, request, cancelled=lambda: aid in self.cancelled)
            self.write(aid, "succeeded", progress=1)
        except PTMError as exc:
            self.write(aid, "cancelled" if exc.code == "PTM_CANCELLED" else "failed", error=exc.as_dict())
        except Exception:
            self.write(aid, "failed", error=PTMError("PTM_JOB_ERROR", "Analysis failed; inspect local logs").as_dict())
            import logging
            logging.getLogger(__name__).exception("Analysis failed: %s", aid)

    def cancel(self, aid: str) -> dict[str, Any]:
        with self.lock:
            existing = self.get(aid)
            if existing["status"] in {"queued", "running"}:
                self.cancelled.add(aid)
                return {"analysis_id": aid, "status": "cancellation_requested"}
            return {"analysis_id": aid, "status": existing["status"]}

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=False)
