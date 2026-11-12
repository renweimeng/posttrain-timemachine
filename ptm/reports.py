import hashlib
import html
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from ptm import __version__
from ptm.errors import PTMError
from ptm.replay import AnalysisRequest, RewardSpec, compute_diff, load_analysis
from ptm.schemas import Snapshot
from ptm.signals import replay_objective
from ptm.storage import Store, canonical

ALLOWED_FILES = {"analysis.json", "input.json", "request.json", "report.html", "report.md", "manifest.json"}
OPTIONAL_FILES = {"objective-replays.json"}


def objective_replays(snapshot: Snapshot, spec: RewardSpec) -> list[dict[str, Any]]:
    results = []
    for objective in snapshot.objectives:
        row: dict[str, Any] = {"objective_id": objective.objective_id}
        for name, reward_spec in [("before", None), ("after", spec)]:
            try:
                row[name] = replay_objective(snapshot, objective, reward_spec)
            except PTMError as exc:
                row[name] = {"evidence": "unavailable", "error": exc.as_dict()}
        results.append(row)
    return results


def report_files(store: Store, analysis_id: str) -> dict[str, bytes]:
    result = load_analysis(store, analysis_id)
    snapshot = store.load(result["run_id"], result["snapshot_id"])
    selected = {row["group_id"] for row in result["groups"]}
    snapshot.groups = [g for g in snapshot.groups if g.group_id in selected]
    snapshot.completions = [c for c in snapshot.completions if c.group_id in selected]
    completion_ids = {c.completion_id for c in snapshot.completions}
    snapshot.tokens = [t for t in snapshot.tokens if t.completion_id in completion_ids]
    token_ids = {t.token_blob_id for t in snapshot.tokens}
    snapshot.objectives = [o for o in snapshot.objectives if set(o.token_blob_ids) <= token_ids]
    snapshot.trajectories = [t for t in snapshot.trajectories if t.completion_id in completion_ids]
    batches = {g.generation_batch_id for g in snapshot.groups}
    snapshot.batches = [b for b in snapshot.batches if b.generation_batch_id in batches]
    snapshot.optimizer_steps = []
    # Export an allowlist of algorithm parameters, never arbitrary trainer configuration/secrets.
    snapshot.run.config = {
        k: v
        for k, v in snapshot.run.config.items()
        if k
        in {
            "reward_batch_scope",
            "num_generations",
            "correctness_weight",
            "format_weight",
            "scale_rewards",
            "capture_errors",
            "world_size",
            "importance_sampling_level",
            "top_entropy_quantile",
            "delta",
            "use_vllm",
            "vllm_importance_sampling_correction",
        }
    }
    objective_results = objective_replays(snapshot, RewardSpec.model_validate(result["reward_spec"]))
    provenance = {
        "ptm_version": __version__,
        "schema_version": snapshot.run.schema_version,
        "input_snapshot_hash": result["snapshot_id"],
        "result_hash": result["result_hash"],
        "spec_hash": hashlib.sha256(canonical(result["reward_spec"])).hexdigest(),
        "run": snapshot.run.model_dump(mode="json"),
        "recipe": result["recipe"],
        "sample_filter": result["group_filter"],
        "coverage": result["coverage"],
        "alignment": "explicit generation_step only",
        "export_scope": "selected captured groups; allowlisted config; optimizer state excluded",
        "evidence": result["evidence"],
        "limits": result["limits"],
        "token_observations_included": len(snapshot.tokens),
        "complete_objectives_included": len(snapshot.objectives),
    }
    text = (
        f"# PostTrain Time Machine investigation\n\nMaterial: **{snapshot.run.material}**\n\n"
        "Fixed saved completions only. **Retrain validation pending.**\n\n"
        "Observed completion → Recomputed reward → Recomputed advantage → Future outcome unavailable.\n\n"
        f"## Summary\n\n```json\n{json.dumps(result['summary'], indent=2)}\n```\n\n"
        "## Provenance and known missing inputs\n\n"
        f"```json\n{json.dumps(provenance, indent=2, ensure_ascii=False)}\n```\n\n"
        + (
            f"Captured Token observations: {len(snapshot.tokens)}. Complete objective batches included: {len(snapshot.objectives)}.\n\n"
            "Token arrays and masks are available only when included in input.json. Gradients and model/optimizer weights are excluded.\n\n"
            f"## Objective evidence\n\n```json\n{json.dumps(objective_results, indent=2)}\n```\n\n"
        )
        + "Reproduce the calculations: `ptm verify report.zip --recompute`. "
        "This reproduces the fixed-sample analysis, not the stochastic training run.\n"
    )
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(c[k]))}</td>"
            for k in [
                "completion_id",
                "reward_before",
                "reward_after",
                "rank_before",
                "rank_after",
                "advantage_before",
                "advantage_after",
            ]
        )
        + "</tr>"
        for c in result["changes"]
    )
    page = (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
        "<title>PTM investigation</title><style>body{background:#0b1120;color:#eaf4ff;font:15px system-ui;"
        "max-width:1200px;margin:40px auto;padding:24px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "table{border-collapse:collapse;font:12px monospace}td,th{padding:10px;border:1px solid #33445e}"
        "a{color:#4ce8d6}</style><h1>PostTrain Time Machine</h1><pre>"
        + html.escape(text)
        + "</pre><h2>Completion evidence / before → after</h2><table><caption>Recomputed on saved samples</caption>"
        "<thead><tr><th>Completion</th><th>Reward before</th><th>Reward after</th><th>Rank before</th>"
        "<th>Rank after</th><th>Advantage before</th><th>Advantage after</th></tr></thead><tbody>"
        + rows
        + "</tbody></table></html>"
    )
    request = {
        "run_id": result["run_id"],
        "snapshot_id": None,
        "reward_spec": result["reward_spec"],
        "algorithm_recipe_id": result["recipe"]["recipe_id"],
        "group_filter": result["group_filter"],
    }
    files = {
        "analysis.json": canonical(result),
        "input.json": canonical(snapshot.ordered()),
        "request.json": canonical(request),
        "report.md": text.encode(),
        "report.html": page.encode(),
    }
    if snapshot.objectives:
        files["objective-replays.json"] = canonical(objective_results)
    manifest = {
        "format_version": 1,
        "analysis_id": analysis_id,
        "provenance": provenance,
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    files["manifest.json"] = canonical(manifest)
    return files


def build_bundle(store: Store, analysis_id: str) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(report_files(store, analysis_id).items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return stream.getvalue()


def verify_bundle(path: Path, recompute: bool = False) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if (
                not ALLOWED_FILES <= set(names)
                or set(names) - ALLOWED_FILES - OPTIONAL_FILES
                or len(names) != len(set(names))
            ):
                raise PTMError("PTM_UNSAFE_ARCHIVE", "Unexpected, duplicate or missing archive members")
            if sum(i.file_size for i in archive.infolist()) > 128 * 1024 * 1024:
                raise PTMError("PTM_IMPORT_LIMIT", "Uncompressed bundle exceeds 128 MiB")
            files = {name: archive.read(name) for name in names}
        manifest = json.loads(files["manifest.json"])
        if set(manifest["files"]) != set(files) - {"manifest.json"}:
            raise PTMError("PTM_INTEGRITY_ERROR", "Manifest file set mismatch")
        for name, expected in manifest["files"].items():
            if hashlib.sha256(files[name]).hexdigest() != expected:
                raise PTMError("PTM_INTEGRITY_ERROR", f"Checksum mismatch: {name}")
        if recompute:
            snapshot = Snapshot.model_validate_json(files["input.json"])
            request = AnalysisRequest.model_validate_json(files["request.json"])
            result = json.loads(files["analysis.json"])
            calculated = compute_diff(snapshot, request)
            for key in ["summary", "changes", "groups", "evidence", "limits"]:
                if canonical(calculated[key]) != canonical(result[key]):
                    raise PTMError("PTM_RECOMPUTE_MISMATCH", f"Recomputed {key} differs")
            if "objective-replays.json" in files:
                expected = objective_replays(snapshot, request.reward_spec)
                if canonical(expected) != files["objective-replays.json"]:
                    raise PTMError("PTM_RECOMPUTE_MISMATCH", "Recomputed objective evidence differs")
        return {
            "verified": True,
            "analysis_id": manifest["analysis_id"],
            "files": len(files),
            "recomputed": recompute,
            "limit": "Checksums establish integrity, not trusted authorship.",
        }
    except PTMError:
        raise
    except (ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise PTMError("PTM_REPORT_INVALID", str(exc)) from None
