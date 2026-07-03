"""Tests for the RAG eval judges: strict schemas, malformed-judgment
coercion and retry, and per-case failure containment in the runner."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from evals import judges
from evals.judges import (
    CitationAccuracyJudgment,
    CompletenessJudgment,
    FaithfulnessJudgment,
    _call_judge,
    _coerce_stringified_lists,
    _strict_schema,
)

GOOD_GRADES = {
    "grades": [{"chunk_id": "c1", "relevant": True, "reasoning": "on topic"}],
    "score": 1.0,
}

# The judge failure observed in production: the list arrives as a JSON string
# with a doubled closing brace, so it is not decodable, only rejectable.
MALFORMED_GRADES = {"grades": '[{"chunk_id": "c1"}}]', "score": 1.0}


def _tool_use_msg(payload):
    return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=payload)])


def _walk(node, check):
    if isinstance(node, dict):
        check(node)
        for value in node.values():
            _walk(value, check)
    elif isinstance(node, list):
        for value in node:
            _walk(value, check)


def test_strict_schema_objects_are_closed_and_fully_required():
    for cls in (FaithfulnessJudgment, CitationAccuracyJudgment, CompletenessJudgment):
        schema = _strict_schema(cls)

        def check(node):
            for banned in ("minimum", "maximum", "default"):
                assert banned not in node
            if node.get("type") == "object" and node.get("properties"):
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])

        _walk(schema, check)


def test_score_bounds_still_enforced_client_side():
    with pytest.raises(ValidationError):
        CitationAccuracyJudgment.model_validate({"grades": [], "score": 1.5})


def test_coerce_decodes_stringified_list():
    data = {"grades": json.dumps(GOOD_GRADES["grades"]) + "\n", "score": 1.0}
    fixed = _coerce_stringified_lists(data, CitationAccuracyJudgment)
    judgment = CitationAccuracyJudgment.model_validate(fixed)
    assert judgment.grades[0].chunk_id == "c1"


def test_coerce_passes_well_formed_input_through():
    fixed = _coerce_stringified_lists(dict(GOOD_GRADES), CitationAccuracyJudgment)
    assert fixed["grades"] == GOOD_GRADES["grades"]


def test_coerce_leaves_undecodable_strings_to_fail_validation():
    fixed = _coerce_stringified_lists(dict(MALFORMED_GRADES), CitationAccuracyJudgment)
    assert fixed["grades"] == MALFORMED_GRADES["grades"]
    with pytest.raises(ValidationError):
        CitationAccuracyJudgment.model_validate(fixed)


def test_call_judge_sends_strict_tool_and_returns_judgment():
    client = MagicMock()
    client.messages.create.return_value = _tool_use_msg(dict(GOOD_GRADES))
    with patch.object(judges, "_client", return_value=client):
        result = _call_judge("model", "sys", "user", CitationAccuracyJudgment)

    assert result.score == 1.0
    assert client.messages.create.call_count == 1
    tool = client.messages.create.call_args.kwargs["tools"][0]
    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False


def test_call_judge_retries_once_on_malformed_judgment():
    client = MagicMock()
    client.messages.create.side_effect = [
        _tool_use_msg(dict(MALFORMED_GRADES)),
        _tool_use_msg(dict(GOOD_GRADES)),
    ]
    with patch.object(judges, "_client", return_value=client):
        result = _call_judge("model", "sys", "user", CitationAccuracyJudgment)

    assert result.grades[0].chunk_id == "c1"
    assert client.messages.create.call_count == 2


def test_call_judge_raises_after_two_malformed_attempts():
    client = MagicMock()
    client.messages.create.side_effect = [
        _tool_use_msg(dict(MALFORMED_GRADES)),
        _tool_use_msg(dict(MALFORMED_GRADES)),
    ]
    with patch.object(judges, "_client", return_value=client):
        with pytest.raises(ValidationError):
            _call_judge("model", "sys", "user", CitationAccuracyJudgment)
    assert client.messages.create.call_count == 2


def test_call_judge_raises_when_no_tool_use_block():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text")]
    )
    with patch.object(judges, "_client", return_value=client):
        with pytest.raises(RuntimeError):
            _call_judge("model", "sys", "user", CitationAccuracyJudgment)
    assert client.messages.create.call_count == 2


def test_run_case_records_judging_failure_as_case_error():
    from evals import run as eval_run

    case = {
        "id": "case-1",
        "fixture_pdf": "Document.pdf",
        "question": "q",
        "reference_answer": "a",
        "expected_refusal": False,
        "reference_keywords": ["a"],
        "category": "factual",
    }
    chunk = SimpleNamespace(id="c1", text="chunk text")
    response = SimpleNamespace(answer="an answer", refusal_reason=None, citations=[])

    with (
        patch.object(eval_run, "embed_query", return_value=[0.0]),
        patch.object(eval_run, "retrieve", return_value=[chunk]),
        patch.object(eval_run, "llm_ask", return_value=(response, {})),
        patch.object(
            eval_run.judges, "judge_faithfulness", side_effect=RuntimeError("boom")
        ),
    ):
        result = eval_run._run_case(
            case, {"Document.pdf": "job-1"}, db=None, judge_model="model"
        )

    assert result["case_id"] == "case-1"
    assert "judging failed" in result["error"]
