import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal, Union
from xml.etree.ElementTree import Element, SubElement, tostring

import yaml
from pydantic import Field, model_validator

from ptm.errors import PTMError
from ptm.importer import check_depth
from ptm.replay import RewardSpec, evaluate
from ptm.schemas import Completion, Identifier, Model
from ptm.storage import Store, atomic_write


class Order(Model):
    type: Literal["reward_order"]
    higher: str
    lower: str


class Margin(Model):
    type: Literal["min_margin"]
    higher: str
    lower: str
    margin: float = Field(ge=0)


class Invariance(Model):
    type: Literal["invariance"]
    original: str
    variant: str
    tolerance: float = Field(ge=0)


Assertion = Annotated[Union[Order, Margin, Invariance], Field(discriminator="type")]


class Origin(Model):
    run_id: str
    analysis_id: str | None = None
    reviewed: bool = False
    note: str = "User-provided expected property; not an automatic truth judgement"


class RewardContract(Model):
    contract_version: Literal[1] = 1
    id: Identifier
    fixtures: list[Completion] = Field(min_length=2, max_length=1000)
    assertions: list[Assertion] = Field(min_length=1, max_length=1000)
    origin: Origin

    @model_validator(mode="after")
    def references(self) -> "RewardContract":
        ids = [c.completion_id for c in self.fixtures]
        if len(set(ids)) != len(ids):
            raise ValueError("contract fixture IDs must be unique")
        for assertion in self.assertions:
            values = assertion.model_dump()
            for key in ("higher", "lower", "original", "variant"):
                if key in values and values[key] not in ids:
                    raise ValueError(f"assertion fixture {values[key]} is missing")
        return self


def test_contract(contract: RewardContract, spec: RewardSpec) -> dict[str, Any]:
    if not contract.origin.reviewed:
        raise PTMError("PTM_CONTRACT_UNREVIEWED", "Explicit human review is required before executing assertions")
    totals, components = evaluate(contract.fixtures, spec)
    scores = {c.completion_id: v for c, v in zip(contract.fixtures, totals, strict=True)}
    assertions = []
    for assertion in contract.assertions:
        if isinstance(assertion, Invariance):
            delta = abs(scores[assertion.original]-scores[assertion.variant])
            passed = delta <= assertion.tolerance
        else:
            delta = scores[assertion.higher]-scores[assertion.lower]
            passed = delta > 0 if isinstance(assertion, Order) else (
                delta >= assertion.margin or math.isclose(delta, assertion.margin, rel_tol=1e-12, abs_tol=1e-12))
        assertions.append({**assertion.model_dump(), "passed": passed, "actual_delta": delta})
    return {"contract_id": contract.id, "passed": all(a["passed"] for a in assertions),
            "assertions": assertions, "scores": scores,
            "components": {c.completion_id: v for c, v in zip(contract.fixtures, components, strict=True)},
            "origin": contract.origin.model_dump(), "spec": spec.name}


# Avoid pytest collecting this public runner when imported in a test module.
test_contract.__test__ = False  # type: ignore[attr-defined]


def save_contract(store: Store, contract: RewardContract) -> Path:
    if not contract.origin.reviewed:
        raise PTMError("PTM_CONTRACT_UNREVIEWED", "Confirm the expected invariant before saving")
    path = store.path("contracts", contract.id + ".yaml")
    if path.exists() and load_contract(path) != contract:
        raise PTMError("PTM_CONTRACT_CONFLICT", "Use a new ID to revise a saved contract", status=409)
    atomic_write(path, yaml.safe_dump(contract.model_dump(mode="json"), sort_keys=True).encode())
    return path


def load_contract(path: Path) -> RewardContract:
    if path.stat().st_size > 8 * 1024 * 1024 or path.is_symlink():
        raise PTMError("PTM_IMPORT_LIMIT", "Contract is too large or a symlink")
    data = yaml.safe_load(path.read_text())
    check_depth(data)
    return RewardContract.model_validate(data)


def junit(results: list[dict[str, Any]]) -> bytes:
    suite = Element("testsuite", name="PTM reward contracts", tests=str(len(results)),
                    failures=str(sum(not r["passed"] for r in results)))
    for result in results:
        case = SubElement(suite, "testcase", name=result["contract_id"], classname="ptm.contracts")
        if not result["passed"]:
            SubElement(case, "failure", message="Reward invariant failed").text = json.dumps(result)
    return tostring(suite, encoding="utf-8", xml_declaration=True)


def export_pytest(contracts: list[RewardContract], spec: RewardSpec, output: Path) -> None:
    payload = json.dumps({"contracts": [c.model_dump(mode="json") for c in contracts],
                          "spec": spec.model_dump(mode="json")}, ensure_ascii=False)
    # repr yields a Python string literal; fixture text never becomes Python source.
    script = f'''"""Portable PTM regression fixtures. Run with pytest; requires posttrain-timemachine."""
import json
import pytest
from ptm.contracts import RewardContract, test_contract as run_contract
from ptm.replay import RewardSpec

DATA = json.loads({payload!r})

@pytest.mark.parametrize("data", DATA["contracts"], ids=[c["id"] for c in DATA["contracts"]])
def test_reward_invariant(data):
    result = run_contract(RewardContract.model_validate(data), RewardSpec.model_validate(DATA["spec"]))
    assert result["passed"], json.dumps(result, ensure_ascii=False, indent=2)
'''
    atomic_write(output, script.encode())


def mutate(fixture: Completion, kind: str) -> Completion:
    """Produce review candidates only. No mutation invents correctness/tokenizer facts."""
    value = fixture.model_copy(deep=True)
    value.completion_id = fixture.completion_id + "-variant"
    text = fixture.text or ""
    if kind == "irrelevant_padding":
        value.text = text + "\nAdditional unrelated commentary."
    elif kind == "equivalent_format":
        value.text = "  " + text + "\n"
    elif kind == "repeated_answer":
        value.text = text + "\n" + text
    elif kind == "correctness_change":
        value.text = text + "\nAlternative final answer: -999999"
    else:
        raise PTMError("PTM_MUTATION_INVALID", "Unknown built-in mutation")
    value.inputs = {k: None for k in value.inputs}
    value.grade = None
    value.reward_total = None
    value.reward_status = "missing"
    value.reward_components = {}
    value.completion_token_count = None
    value.text_hash = None
    return value
