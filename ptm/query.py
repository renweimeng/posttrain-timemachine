import json
from typing import Any

import duckdb

from ptm.algorithms import advantages, get_recipe
from ptm.errors import PTMError
from ptm.storage import Store


class Repository:
    def __init__(self, store: Store) -> None:
        self.store = store

    def groups(
        self, run_id: str, *, after: str = "", limit: int = 50,
        step_min: int = 0, step_max: int = 2**63-1, search: str = "", collapse: bool = False,
    ) -> dict[str, Any]:
        path = self.store.snapshot_path(run_id)
        with duckdb.connect() as con:
            rows = con.execute("""
                SELECT g.payload, b.generation_step, count(c.completion_id) AS recorded,
                       avg(c.reward_total) AS reward_mean, var_pop(c.reward_total) AS variance,
                       count(c.completion_id) FILTER (WHERE c.reward_status != 'valid') AS invalid
                FROM read_parquet(?) g
                JOIN read_parquet(?) b USING (generation_batch_id)
                LEFT JOIN read_parquet(?) c USING (group_id)
                WHERE g.group_id > ? AND b.generation_step BETWEEN ? AND ?
                  AND (contains(lower(g.prompt_id), lower(?)) OR contains(lower(g.payload), lower(?)))
                GROUP BY g.group_id, g.payload, b.generation_step
                HAVING (NOT ? OR var_pop(c.reward_total) = 0)
                ORDER BY g.group_id LIMIT ?
            """, [str(path/"groups.parquet"), str(path/"batches.parquet"), str(path/"completions.parquet"),
                  after, step_min, step_max, search, search, collapse, min(max(limit, 1), 200)+1]).fetchall()
        items = [{**json.loads(row[0]), "generation_step": row[1], "recorded": row[2],
                  "reward_mean": row[3], "variance": row[4], "variance_ddof": 0,
                  "invalid_rewards": row[5]} for row in rows]
        size = min(max(limit, 1), 200)
        return {"items": items[:size], "next_cursor": items[size-1]["group_id"] if len(items) > size else None,
                "snapshot_id": self.store.manifest(run_id)["snapshot_id"]}

    def group(self, run_id: str, group_id: str) -> dict[str, Any]:
        path = self.store.snapshot_path(run_id)
        with duckdb.connect() as con:
            row = con.execute("SELECT payload FROM read_parquet(?) WHERE group_id = ?",
                              [str(path/"groups.parquet"), group_id]).fetchone()
            if row is None:
                raise PTMError("PTM_NOT_FOUND", "Prompt group not found", status=404)
            group = json.loads(row[0])
            cs = con.execute("SELECT payload FROM read_parquet(?) WHERE group_id = ? ORDER BY sequence_index",
                             [str(path/"completions.parquet"), group_id]).fetchall()
            batch = con.execute("SELECT payload FROM read_parquet(?) WHERE generation_batch_id = ?",
                                [str(path/"batches.parquet"), group["generation_batch_id"]]).fetchone()
        completions = [json.loads(c[0]) for c in cs]
        available = len(completions) == group["expected_group_size"] and all(
            c["reward_status"] == "valid" for c in completions)
        run = next(r for r in self.store.runs() if r["run_id"] == run_id)
        recipe = run["recipe_id"]
        unavailable = None
        values: list[float] = []
        if available:
            try:
                r = get_recipe(recipe, framework=run["framework"], framework_version=run["framework_version"])
                if r.scale == "batch":
                    raise PTMError("PTM_BATCH_SCOPE_UNAVAILABLE", "Single-group inspector lacks batch denominator")
                values = advantages([[c["reward_total"] for c in completions]], r)[0]
            except PTMError as exc:
                available = False
                unavailable = exc.as_dict()
        if not available and unavailable is None:
            unavailable = PTMError("PTM_INCOMPLETE_GROUP", "Incomplete group or invalid reward").as_dict()
        return {**group, "run_id": run_id, "batch": json.loads(batch[0]) if batch else None,
                "completions": [{**c, "advantage": values[i] if available else None}
                                for i, c in enumerate(completions)],
                "advantage_available": available, "unavailable": unavailable,
                "recipe_id": recipe, "snapshot_id": self.store.manifest(run_id)["snapshot_id"]}

    def find_group(self, group_id: str, run_id: str | None) -> dict[str, Any]:
        if run_id:
            return self.group(run_id, group_id)
        found = []
        for run in self.store.runs():
            try:
                found.append(self.group(run["run_id"], group_id))
            except PTMError as exc:
                if exc.code != "PTM_NOT_FOUND":
                    raise
        if len(found) != 1:
            raise PTMError("PTM_AMBIGUOUS_ID" if found else "PTM_NOT_FOUND",
                           "Supply run_id for this group", status=409 if found else 404)
        return found[0]

    def metrics(self, run_id: str) -> list[dict[str, Any]]:
        path = self.store.snapshot_path(run_id)
        with duckdb.connect() as con:
            rows = con.execute("SELECT payload FROM read_parquet(?) ORDER BY generation_step, name",
                               [str(path/"metrics.parquet")]).fetchall()
        return [json.loads(r[0]) for r in rows]

    def tokens(self, completion_id: str, run_id: str | None = None, blob_id: str | None = None) -> dict[str, Any]:
        from ptm.schemas import TokenBlob
        from ptm.signals import token_signals
        found=[]
        for run in ([{'run_id':run_id}] if run_id else self.store.runs()):
            path=self.store.snapshot_path(run['run_id'])/'tokens.parquet'
            if not path.exists():
                continue
            with duckdb.connect() as con:
                rows=con.execute('SELECT payload FROM read_parquet(?) WHERE completion_id=? ORDER BY optimizer_step DESC,token_blob_id',
                                 [str(path),completion_id]).fetchall()
            if rows:
                found.append((run['run_id'],[TokenBlob.model_validate_json(r[0]) for r in rows]))
        if not found:
            raise PTMError('PTM_CAPTURE_LEVEL_TOO_LOW','No captured Token observation for this completion',
                           missing_fields=['token_ids','loss_mask','policy_logprobs'])
        if len(found)>1:
            raise PTMError('PTM_AMBIGUOUS_ID','Specify run_id for this completion',status=409)
        rid, observations=found[0]
        selected=next((t for t in observations if t.token_blob_id==blob_id),None) if blob_id else observations[0]
        if selected is None:
            raise PTMError('PTM_NOT_FOUND','Token observation not found',status=404)
        matching = [o for o in self.objectives(rid) if selected.token_blob_id in o['token_blob_ids']]
        objective = matching[0] if len(matching) == 1 else None
        epsilon_low = objective['epsilon_low'] if objective else 0.2
        epsilon_high = objective['epsilon_high'] if objective else 0.2
        clipping = {'epsilon_low': epsilon_low, 'epsilon_high': epsilon_high,
                    'source': 'captured objective' if objective else 'display calculator default; objective unavailable or ambiguous',
                    'objective_id': objective['objective_id'] if objective else None}
        return {'clipping': clipping, 'run_id':rid,'observations':[{'token_blob_id':t.token_blob_id,'optimizer_step':t.optimizer_step,
                'current_policy_version':t.current_policy_version,'token_count':len(t.token_ids)} for t in observations],
                'selection':'explicit token_blob_id' if blob_id else 'latest recorded optimizer step',
                'selected':selected.model_dump(),'signals':token_signals(selected,epsilon_low=epsilon_low,epsilon_high=epsilon_high)}

    def objectives(self, run_id: str) -> list[dict[str, Any]]:
        path=self.store.snapshot_path(run_id)/'objectives.parquet'
        if not path.exists():
            return []
        with duckdb.connect() as con:
            return [json.loads(r[0]) for r in con.execute(
                'SELECT payload FROM read_parquet(?) ORDER BY optimizer_step,objective_id',[str(path)]).fetchall()]

    def trajectories(self, run_id: str, completion_id: str) -> list[dict[str, Any]]:
        path=self.store.snapshot_path(run_id)/'trajectories.parquet'
        if not path.exists():
            return []
        with duckdb.connect() as con:
            return [json.loads(r[0]) for r in con.execute(
                'SELECT payload FROM read_parquet(?) WHERE completion_id=?',[str(path),completion_id]).fetchall()]
