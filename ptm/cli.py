import json
from pathlib import Path

import typer

from ptm.demo import create_demo
from ptm.errors import PTMError
from ptm.storage import Store, doctor

app = typer.Typer(no_args_is_help=True, help="PostTrain Time Machine — local GRPO evidence workspace")
analyze_app = typer.Typer(no_args_is_help=True)
contract_app = typer.Typer(no_args_is_help=True)
app.add_typer(analyze_app, name="analyze")
app.add_typer(contract_app, name="contract")
report_app = typer.Typer(no_args_is_help=True)
app.add_typer(report_app, name="report")


def emit(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


@analyze_app.command("reward-diff")
def analyze_diff(
    run: str = typer.Option(...),
    spec: Path = typer.Option(...),
    root: Path = Path(".ptm"),
    group_id: list[str] | None = None,
) -> None:
    from ptm.replay import AnalysisRequest, GroupFilter, load_spec, run_analysis

    store = Store(root)
    recipe = store.load(run).run.recipe_id
    result = run_analysis(
        store,
        AnalysisRequest(
            run_id=run,
            reward_spec=load_spec(spec),
            algorithm_recipe_id=recipe,
            group_filter=GroupFilter(group_ids=group_id),
        ),
    )
    emit({k: result[k] for k in ("analysis_id", "result_hash", "summary", "coverage", "limits")})


@contract_app.command("test")
def contract_test(
    directory: Path,
    reward_spec: Path = typer.Option(...),
    json_output: Path = Path("contract-results.json"),
    junit_output: Path = Path("contract-junit.xml"),
) -> None:
    from ptm.contracts import junit, load_contract, test_contract
    from ptm.replay import load_spec
    from ptm.storage import atomic_write, canonical

    spec = load_spec(reward_spec)
    files = sorted(directory.glob("*.yaml")) if directory.is_dir() else [directory]
    if not files:
        raise PTMError("PTM_EMPTY_SELECTION", "No contract YAML files found")
    results = [test_contract(load_contract(path), spec) for path in files]
    atomic_write(json_output, canonical(results))
    atomic_write(junit_output, junit(results))
    emit(results)
    if not all(r["passed"] for r in results):
        raise typer.Exit(1)


@contract_app.command("export-pytest")
def contract_export(
    directory: Path, reward_spec: Path = typer.Option(...), output: Path = typer.Option(..., "-o", "--output")
) -> None:
    from ptm.contracts import export_pytest, load_contract
    from ptm.replay import load_spec

    files = sorted(directory.glob("*.yaml")) if directory.is_dir() else [directory]
    if not files:
        raise PTMError("PTM_EMPTY_SELECTION", "No contract YAML files found")
    export_pytest([load_contract(path) for path in files], load_spec(reward_spec), output)
    emit({"output": str(output)})


@report_app.command("build")
def report_build(
    analysis: str = typer.Option(...), output: Path = typer.Option(...), root: Path = Path(".ptm")
) -> None:
    from ptm.reports import build_bundle, report_files
    from ptm.storage import atomic_write

    store = Store(root)
    if output.suffix == ".zip":
        atomic_write(output, build_bundle(store, analysis))
    elif output.suffix in {".html", ".md"}:
        atomic_write(output, report_files(store, analysis)["report" + output.suffix])
    else:
        raise PTMError("PTM_REPORT_FORMAT", "Report output must end in .zip, .html or .md")
    emit({"output": str(output)})


@app.command("verify")
def verify_command(bundle: Path, recompute: bool = False) -> None:
    from ptm.reports import verify_bundle

    emit(verify_bundle(bundle, recompute))


@app.command()
def demo(root: Path = Path(".ptm"), no_serve: bool = False, port: int = 8765) -> None:
    """Create an explicitly synthetic, GPU-free challenge and open the workspace."""
    store = Store(root)
    run_id = create_demo(store)
    emit({"run_id": run_id, "material": "synthetic example", "root": str(store.root)})
    if not no_serve:
        serve(root, port=port, open_browser=True)


@app.command("doctor")
def doctor_command(run: str = typer.Argument("demo-format-truth"), root: Path = Path(".ptm")) -> None:
    """Verify snapshot checksums, schema, coverage, and available capture capabilities."""
    result = doctor(Store(root).load(run))
    emit(result)
    if result["incomplete_groups"] or result["invalid_rewards"]:
        raise typer.Exit(1)


@app.command("import")
def import_command(
    source: Path,
    root: Path = Path(".ptm"),
    format: str = typer.Option("ptm-jsonl", "--format"),
    text_capture: str = "full",
) -> None:
    """Validate canonical events and atomically publish an immutable snapshot."""
    from ptm.importer import read_snapshot

    if format not in {"ptm-jsonl", "trl-jsonl", "verl-jsonl"}:
        raise PTMError("PTM_IMPORT_FORMAT", "Only normalized PTM/TRL typed events are supported")
    if text_capture not in {"full", "off", "hash_only"}:
        raise PTMError("PTM_IMPORT_INVALID", "text_capture must be full, off or hash_only")
    if format == "verl-jsonl":
        from ptm.interop import read_verl

        snapshot = read_verl(source, text_capture)  # type: ignore[arg-type]
    else:
        snapshot = read_snapshot(source, text_capture)  # type: ignore[arg-type]
    emit(Store(root).commit(snapshot))


@app.command("merge")
def merge_command(manifest: Path, root: Path = Path(".ptm")) -> None:
    """Verify and assemble explicitly identified rank spools into one global snapshot."""
    from ptm.interop import merge_rank_manifest

    emit(Store(root).commit(merge_rank_manifest(manifest)))


@app.command("replay-update")
def update_command(capture: str, root: Path = Path(".ptm"), spec: Path | None = None) -> None:
    """Execute one real bounded optimizer step from a verified numeric local capture."""
    from ptm.replay import load_spec
    from ptm.updates import replay_update

    emit(replay_update(Store(root), capture, load_spec(spec) if spec else None))


@app.command("preflight")
def preflight_command() -> None:
    """Report actual CPU/CUDA reference capability without promising untested hardware."""
    import importlib.util

    result: dict[str, object] = {
        "local_workspace": True,
        "reference_installed": importlib.util.find_spec("torch") is not None,
    }
    if result["reference_installed"]:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "verified_deep_capture_device": "cpu",
            }
        )
    emit(result)


@app.command()
def serve(
    root: Path = typer.Argument(Path(".ptm")),
    port: int = 8765,
    open_browser: bool = typer.Option(False, "--open"),
) -> None:
    """Serve the local workspace (loopback only)."""
    import webbrowser

    import uvicorn

    from ptm.api import create_app

    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}")
    uvicorn.run(create_app(root), host="127.0.0.1", port=port)


def main() -> None:
    try:
        app()
    except PTMError as exc:
        emit(exc.as_dict())
        raise SystemExit(1) from None
