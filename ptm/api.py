import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import yaml
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import Field

from ptm.algorithms import RECIPES, get_recipe
from ptm.arena import challenge, challenges, evaluate_challenge, load_challenge
from ptm.comparison import CompareRequest, compare_groups
from ptm.contracts import RewardContract, export_pytest, load_contract, save_contract
from ptm.diagnostics import detect
from ptm.errors import PTMError
from ptm.http_limits import BodyLimitMiddleware
from ptm.importer import check_depth
from ptm.jobs import Jobs
from ptm.query import Repository
from ptm.replay import AnalysisRequest, RewardSpec
from ptm.reports import build_bundle
from ptm.schemas import Model
from ptm.signals import save_objective_analysis
from ptm.storage import Store, doctor
from ptm.updates import list_captures, reference_environment, replay_update
from ptm.workspace import (
    ImportRequest,
    analysis_history,
    contract_history,
    contract_results_export,
    export_run,
    import_content,
    record_contract_test,
    recorded_reward_spec,
)


class ObjectiveRequest(Model):
    run_id: str
    reward_spec: RewardSpec | None = None


class ContractContent(Model):
    content: str = Field(max_length=4 * 1024 * 1024)


def create_app(root: Path | str = ".ptm") -> FastAPI:
    store = Store(root)
    repository = Repository(store)
    jobs = Jobs(store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        jobs.close()

    app = FastAPI(title="PostTrain Time Machine", version="0.2.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)
    app.state.store = store
    app.state.jobs = jobs

    @app.middleware("http")
    async def local_origin(request: Request, call_next: Any) -> Response:
        host = request.url.hostname
        if host not in {"127.0.0.1", "localhost", "[::1]", "::1", "testserver"}:
            return JSONResponse(
                PTMError("PTM_HOST_FORBIDDEN", "Local workspace host required", status=403).as_dict(), 403
            )
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            fetch_site = request.headers.get("sec-fetch-site")
            if (
                origin and origin != f"{request.url.scheme}://{request.url.netloc}"
            ) or fetch_site == "cross-site":
                return JSONResponse(
                    PTMError("PTM_ORIGIN_FORBIDDEN", "Same-origin writes required", status=403).as_dict(), 403
                )
            length = request.headers.get("content-length")
            if length and (not length.isdecimal() or int(length) > 8 * 1024 * 1024):
                return JSONResponse(PTMError("PTM_IMPORT_LIMIT", "Request body exceeds 8 MiB").as_dict(), 413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; frame-ancestors 'none'; object-src 'none'"
        )
        return response

    @app.exception_handler(PTMError)
    async def error_handler(request: Request, exc: PTMError) -> JSONResponse:
        return JSONResponse(exc.as_dict(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            PTMError(
                "PTM_VALIDATION_ERROR",
                str(exc),
                missing_fields=[".".join(map(str, e["loc"])) for e in exc.errors()],
            ).as_dict(),
            status_code=422,
        )

    @app.get("/api/v1/system/capabilities")
    def system_capabilities() -> dict[str, Any]:
        return reference_environment()

    @app.get("/api/v1/runs")
    def runs() -> dict[str, Any]:
        return {"items": store.runs()}

    @app.get("/api/v1/arena")
    def arena_list() -> dict[str, Any]:
        return {"items": challenges()}

    @app.get("/api/v1/arena/{challenge_id}")
    def arena_detail(challenge_id: str) -> dict[str, Any]:
        return challenge(challenge_id)

    @app.post("/api/v1/arena/{challenge_id}/evaluate")
    def arena_evaluate(challenge_id: str, spec: RewardSpec) -> dict[str, Any]:
        return evaluate_challenge(challenge_id, spec)

    @app.post("/api/v1/arena/{challenge_id}/load", status_code=201)
    def arena_load(challenge_id: str) -> dict[str, Any]:
        return load_challenge(store, challenge_id)

    @app.post("/api/v1/imports", status_code=201)
    def import_run(request: ImportRequest) -> dict[str, Any]:
        return import_content(store, request)

    @app.get("/api/v1/runs/{run_id}/export")
    def run_export(run_id: str) -> Response:
        content = export_run(store, run_id)
        return Response(
            content,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.jsonl"'},
        )

    @app.get("/api/v1/runs/{run_id}/reward-spec")
    def saved_reward(run_id: str) -> dict[str, Any]:
        return recorded_reward_spec(store, run_id)

    @app.get("/api/v1/analyses")
    def analyses(run_id: str | None = None) -> dict[str, Any]:
        return {"items": analysis_history(store, run_id)}

    @app.get("/api/v1/reports")
    def reports(run_id: str | None = None) -> dict[str, Any]:
        return {"items": [r for r in analysis_history(store, run_id) if r["status"] == "succeeded"]}

    @app.get("/api/v1/runs/{run_id}/summary")
    def summary(run_id: str) -> dict[str, Any]:
        for run in store.runs():
            if run["run_id"] == run_id:
                try:
                    get_recipe(
                        run["recipe_id"],
                        framework=run["framework"],
                        framework_version=run["framework_version"],
                    )
                    compatible = True
                except PTMError:
                    compatible = False
                health = doctor(store.load(run_id))
                return {**run, "compatibility": {"reward_advantage": compatible, **health["capabilities"]}}
        raise PTMError("PTM_NOT_FOUND", "Run not found", status=404)

    @app.get("/api/v1/runs/{run_id}/doctor")
    def inspect_run(run_id: str) -> dict[str, Any]:
        return doctor(store.load(run_id))

    @app.get("/api/v1/runs/{run_id}/metrics")
    def metrics(run_id: str) -> dict[str, Any]:
        return {"items": repository.metrics(run_id), "alignment": "generation_step", "smoothing": "none"}

    @app.get("/api/v1/runs/{run_id}/groups")
    def groups(
        run_id: str,
        cursor: str = "",
        limit: int = Query(50, ge=1, le=200),
        step_min: int = Query(0, ge=0),
        step_max: int = Query(2**63 - 1, ge=0),
        search: str = Query("", max_length=200),
        collapse: bool = False,
    ) -> dict[str, Any]:
        return repository.groups(
            run_id,
            after=cursor,
            limit=limit,
            step_min=step_min,
            step_max=step_max,
            search=search,
            collapse=collapse,
        )

    @app.get("/api/v1/groups/{group_id}")
    def group(group_id: str, run_id: str | None = None) -> dict[str, Any]:
        return repository.find_group(group_id, run_id)

    @app.post("/api/v1/comparisons/groups")
    def compare(request: CompareRequest) -> dict[str, Any]:
        return compare_groups(repository, request)

    @app.get("/api/v1/completions/{completion_id}/tokens")
    def tokens(
        completion_id: str, run_id: str | None = None, token_blob_id: str | None = None
    ) -> dict[str, Any]:
        return repository.tokens(completion_id, run_id, token_blob_id)

    @app.get("/api/v1/runs/{run_id}/objectives")
    def objectives(run_id: str) -> dict[str, Any]:
        return {"items": repository.objectives(run_id)}

    @app.get("/api/v1/update-captures")
    def update_captures(run_id: str | None = None) -> dict[str, Any]:
        return {"items": list_captures(store, run_id)}

    @app.post("/api/v1/update-captures/{capture_id}/replay")
    def update_replay(capture_id: str, request: ObjectiveRequest) -> dict[str, Any]:
        captures = list_captures(store, request.run_id)
        if not any(c["capture_id"] == capture_id for c in captures):
            raise PTMError("PTM_NOT_FOUND", "Update capture does not belong to selected run", status=404)
        return replay_update(store, capture_id, request.reward_spec)

    @app.post("/api/v1/objectives/{objective_id}/replay")
    def objective_replay(objective_id: str, request: ObjectiveRequest) -> dict[str, Any]:
        return save_objective_analysis(store, request.run_id, objective_id, request.reward_spec)

    @app.get("/api/v1/completions/{completion_id}/trajectory")
    def trajectory(completion_id: str, run_id: str) -> dict[str, Any]:
        return {"items": repository.trajectories(run_id, completion_id)}

    @app.get("/api/v1/runs/{run_id}/anomalies")
    def anomalies(run_id: str) -> dict[str, Any]:
        return {"items": detect(store.load(run_id, verify=False, include_advanced=False))}

    @app.get("/api/v1/recipes")
    def recipes() -> dict[str, Any]:
        return {"items": [r.model_dump() for r in RECIPES.values()]}

    @app.get("/api/v1/schemas/reward-spec")
    def spec_schema() -> dict[str, Any]:
        return RewardSpec.model_json_schema()

    @app.post("/api/v1/reward-spec/validate")
    def validate_spec(spec: RewardSpec) -> dict[str, Any]:
        return {"valid": True, "spec": spec.model_dump()}

    @app.post("/api/v1/analyses/reward-diff", status_code=202)
    def reward_diff(request: AnalysisRequest) -> dict[str, Any]:
        return jobs.submit(request)

    @app.get("/api/v1/analyses/{analysis_id}")
    def analysis(analysis_id: str) -> dict[str, Any]:
        return jobs.get(analysis_id)

    @app.post("/api/v1/analyses/{analysis_id}/cancel")
    def cancel(analysis_id: str) -> dict[str, Any]:
        return jobs.cancel(analysis_id)

    @app.post("/api/v1/analyses/{analysis_id}/retry", status_code=202)
    def retry(analysis_id: str) -> dict[str, Any]:
        path = store.path("jobs", analysis_id + ".json")
        if not path.exists():
            raise PTMError("PTM_NOT_FOUND", "Saved job request not found", status=404)
        request = json.loads(path.read_bytes()).get("request")
        if request is None:
            raise PTMError("PTM_JOB_NOT_RETRYABLE", "This legacy job has no saved request")
        return jobs.submit(AnalysisRequest.model_validate(request))

    @app.get("/api/v1/analyses/{analysis_id}/events")
    async def events(analysis_id: str) -> StreamingResponse:
        jobs.get(analysis_id)

        async def stream() -> AsyncIterator[str]:
            previous = ""
            while True:
                job = jobs.get(analysis_id)
                data = {
                    k: job[k] for k in ["analysis_id", "status", "phase", "progress", "error"] if k in job
                }
                serialized = json.dumps(data)
                if serialized != previous:
                    yield f"event: progress\ndata: {serialized}\n\n"
                    previous = serialized
                if job["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                await asyncio.sleep(0.15)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/v1/analyses/{analysis_id}/changes")
    def changes(
        analysis_id: str,
        cursor: str = "",
        limit: int = Query(100, ge=1, le=200),
        affected: bool = False,
        sign_flips: bool = False,
    ) -> dict[str, Any]:
        result = jobs.get(analysis_id)
        if result["status"] != "succeeded":
            raise PTMError("PTM_JOB_PENDING", "Analysis has no committed results", status=409)
        rows = sorted(
            [
                c
                for c in result["changes"]
                if c["completion_id"] > cursor
                and (not affected or c["affected"])
                and (not sign_flips or c["sign_flip"])
            ],
            key=lambda c: c["completion_id"],
        )
        return {
            "items": rows[:limit],
            "next_cursor": rows[limit - 1]["completion_id"] if len(rows) > limit else None,
        }

    @app.get("/api/v1/contracts")
    def contracts() -> dict[str, Any]:
        parent = store.path("contracts")
        return {"items": [load_contract(p).model_dump(mode="json") for p in sorted(parent.glob("*.yaml"))]}

    @app.post("/api/v1/contracts/preview")
    def preview_contract(request: ContractContent) -> dict[str, Any]:
        try:
            payload = yaml.safe_load(request.content)
            check_depth(payload)
            contract = RewardContract.model_validate(payload)
            contract.origin.reviewed = False
            return contract.model_dump(mode="json")
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            raise PTMError("PTM_CONTRACT_INVALID", str(exc)) from None

    @app.get("/api/v1/contracts/{contract_id}/export")
    def contract_export(contract_id: str) -> Response:
        contract = load_contract(store.path("contracts", contract_id + ".yaml"))
        return Response(yaml.safe_dump(contract.model_dump(mode="json")), media_type="application/yaml",
                        headers={"Content-Disposition": f'attachment; filename="{contract_id}.yaml"'})

    @app.post("/api/v1/contracts", status_code=201)
    def create_contract(contract: RewardContract) -> dict[str, Any]:
        save_contract(store, contract)
        return contract.model_dump(mode="json")

    @app.post("/api/v1/contracts/{contract_id}/test")
    def contract_test(contract_id: str, spec: RewardSpec) -> dict[str, Any]:
        return record_contract_test(
            store, load_contract(store.path("contracts", contract_id + ".yaml")), spec
        )

    @app.get("/api/v1/contract-results")
    def results(contract_id: str | None = None) -> dict[str, Any]:
        return {"items": contract_history(store, contract_id)}

    @app.get("/api/v1/contract-results/export")
    def results_export(format: str = "json") -> Response:
        return Response(
            contract_results_export(store, format),
            media_type=("application/json" if format == "json" else "application/xml"),
        )

    @app.post("/api/v1/contracts/{contract_id}/export-pytest")
    def contract_pytest(contract_id: str, spec: RewardSpec) -> Response:
        contract = load_contract(store.path("contracts", contract_id + ".yaml"))
        output = store.path("exports", contract_id + ".py")
        export_pytest([contract], spec, output)
        return Response(
            output.read_bytes(),
            media_type="text/x-python",
            headers={"Content-Disposition": f'attachment; filename="test_{contract_id}.py"'},
        )

    @app.get("/api/v1/reports/{analysis_id}/export")
    def export(analysis_id: str) -> Response:
        return Response(
            build_bundle(store, analysis_id),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{analysis_id}.zip"'},
        )

    static = Path(__file__).parent / "web"
    if (static / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/{page_path:path}", include_in_schema=False)
    def frontend(page_path: str) -> Response:
        if page_path.startswith("api/"):
            raise PTMError("PTM_NOT_FOUND", "API route not found", status=404)
        index = static / "index.html"
        if not index.exists():
            return Response(
                "Frontend assets missing. Run npm --prefix web ci && npm --prefix web run build",
                status_code=503,
            )
        return FileResponse(index)

    return app
