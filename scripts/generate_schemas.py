"""Generate distributable Pydantic/Arrow and OpenAPI schemas without starting the service."""
import json
from pathlib import Path

from ptm.api import create_app
from ptm.schemas import Snapshot
from ptm.storage import TABLES

root = Path(__file__).resolve().parents[1]
out = root / "schemas"
out.mkdir(exist_ok=True)
(out/"snapshot.schema.json").write_text(json.dumps(Snapshot.model_json_schema(), indent=2)+"\n")
app = create_app(root/"tmp/schema-store")
(out/"openapi.json").write_text(json.dumps(app.openapi(), indent=2)+"\n")
app.state.jobs.close()
(out/"arrow-types.json").write_text(json.dumps({name: {k: str(v) for k, v in columns.items()}
    for name, columns in TABLES.items()}, indent=2)+"\n")
