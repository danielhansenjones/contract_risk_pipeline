import time
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from worker.processors.classifier import run, _CONFIDENCE_THRESHOLD
from worker.processors.clause_labels import CLAUSE_LABELS
from shared.models import Chunk, Job, JobStage, JobStatus
from tests.conftest import make_classifier_pipeline


def test_high_confidence_assigns_top_label(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(
        job.id, "The licensee shall indemnify and hold harmless the provider.", index=0
    )

    classifier = make_classifier_pipeline(label="indemnification", score=0.92)
    run(job, db_session, classifier)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "indemnification"
    assert chunk.confidence == pytest.approx(0.92)


def test_below_threshold_assigns_general(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Some generic clause text that is hard to classify.", index=0)

    classifier = make_classifier_pipeline(label="warranty", score=0.35)
    run(job, db_session, classifier)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "general"


def test_exactly_at_threshold_assigns_label(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Termination clause text.", index=0)

    classifier = make_classifier_pipeline(
        label="termination", score=_CONFIDENCE_THRESHOLD
    )
    run(job, db_session, classifier)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "termination"


def test_all_chunks_are_classified(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    for i in range(5):
        make_chunk(job.id, f"Clause text number {i}.", index=i)

    run(job, db_session, make_classifier_pipeline())

    chunks = db_session.query(Chunk).filter(Chunk.job_id == job.id).all()
    assert len(chunks) == 5
    assert all(c.clause_type is not None for c in chunks)
    assert all(c.confidence is not None for c in chunks)


def test_batch_boundary_nine_chunks_requires_two_batches(
    make_job, make_chunk, db_session
):
    job = make_job(stage=JobStage.CLASSIFICATION)
    for i in range(9):
        make_chunk(job.id, f"Clause {i} text.", index=i)

    call_count = {"n": 0}
    base = make_classifier_pipeline()

    def counting_pipeline(texts, candidate_labels=None, batch_size=None, multi_label=False):
        call_count["n"] += 1
        return base(texts, candidate_labels=candidate_labels, batch_size=batch_size)

    run(job, db_session, counting_pipeline)

    assert call_count["n"] == 2


def test_single_chunk_pipeline_result_coerced_correctly(
    make_job, make_chunk, db_session
):
    """HuggingFace pipeline returns a bare dict instead of a list for single-item inputs."""
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Only one clause in this document.", index=0)

    def single_dict_pipeline(texts, candidate_labels=None, batch_size=None, multi_label=False):
        labels = candidate_labels or []
        return {
            "labels": ["confidentiality"]
            + [l for l in labels if l != "confidentiality"],
            "scores": [0.88] + [0.01] * (len(labels) - 1),
        }

    run(job, db_session, single_dict_pipeline)

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.clause_type == "confidentiality"


def test_stage_advances_to_scoring(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Payment terms clause.", index=0)

    run(job, db_session, make_classifier_pipeline())

    assert job.stage == JobStage.SCORING


def test_empty_chunk_list_still_advances_stage(make_job, db_session):
    """Jobs with zero chunks are unusual but possible if text extraction yields nothing."""
    job = make_job(stage=JobStage.CLASSIFICATION)
    run(job, db_session, make_classifier_pipeline())
    assert job.stage == JobStage.SCORING


def test_span_extraction_populates_chunk_fields(make_job, make_chunk, db_session):
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(
        job.id, "Either party may terminate this agreement for convenience.", index=0
    )

    extractor = MagicMock()
    extractor.extract.return_value = {
        "Termination For Convenience": {
            "text": "terminate this agreement for convenience",
            "score": 0.91,
        }
    }

    run(
        job,
        db_session,
        make_classifier_pipeline(label="termination", score=0.95),
        span_extractor=extractor,
    )

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert chunk.extracted_span == "terminate this agreement for convenience"
    assert chunk.extracted_span_category == "Termination For Convenience"


def test_span_extraction_timeout_falls_back_to_tier1(make_job, make_chunk, db_session):
    # Label must be one with CUAD coverage (category_mapping.json), otherwise
    # tier-2 never fires and the test passes without exercising the timeout.
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Either party may terminate this agreement.", index=0)

    extractor = MagicMock()
    extractor.extract.side_effect = lambda text, categories: (time.sleep(1), {})[1]

    with patch("worker.processors.classifier.settings") as mock_settings:
        mock_settings.span_extractor_tier1_confidence_threshold = 0.7
        mock_settings.span_extractor_timeout_s = 0.05

        run(
            job,
            db_session,
            make_classifier_pipeline(label="termination", score=0.95),
            span_extractor=extractor,
        )

    chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
    assert extractor.extract.called
    assert chunk.clause_type == "termination"
    assert chunk.extracted_span is None


def test_span_timeout_does_not_starve_later_chunks(make_job, make_chunk, db_session):
    """A hung extraction must not leave later chunks queued behind it."""
    job = make_job(stage=JobStage.CLASSIFICATION)
    make_chunk(job.id, "Either party may terminate this agreement.", index=0)
    make_chunk(job.id, "This agreement terminates upon written notice.", index=1)

    calls = {"n": 0}

    def extract(text, categories):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(0.5)
            return {}
        return {
            "Termination For Convenience": {"text": "may terminate", "score": 0.9}
        }

    extractor = MagicMock()
    extractor.extract.side_effect = extract

    with patch("worker.processors.classifier.settings") as mock_settings:
        mock_settings.span_extractor_tier1_confidence_threshold = 0.7
        mock_settings.span_extractor_timeout_s = 0.05

        run(
            job,
            db_session,
            make_classifier_pipeline(label="termination", score=0.95),
            span_extractor=extractor,
        )

    chunks = (
        db_session.query(Chunk)
        .filter(Chunk.job_id == job.id)
        .order_by(Chunk.index)
        .all()
    )
    assert chunks[0].extracted_span is None
    assert chunks[1].extracted_span == "may terminate"


def test_all_clause_labels_can_be_assigned(make_job, make_chunk, db_session):
    for label in CLAUSE_LABELS:
        job = make_job(stage=JobStage.CLASSIFICATION)
        make_chunk(job.id, f"A clause about {label}.", index=0)

        run(job, db_session, make_classifier_pipeline(label=label, score=0.9))

        chunk = db_session.query(Chunk).filter(Chunk.job_id == job.id).first()
        assert chunk.clause_type == label
        db_session.rollback()
