"""
test_vectorstore.py
===================
Unit tests for VectorStoreManager.

These tests cover the components most likely to be asked about
in technical interviews: duplicate detection, ingestion correctness,
retrieval with filters, and the hallucination guard threshold.

Run with: uv run pytest tests/ -v

PEP 8 | OOP
"""

from __future__ import annotations

import pytest

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> Settings:
    """Isolated Chroma path and test-friendly retrieval thresholds.

    BaseSettings reads .env / OS env, which otherwise overrides kwargs and
    points tests at the developer's real ./data/chroma_db.
    """
    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma_db"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_dl_corpus")
    monkeypatch.setenv("RETRIEVAL_K", "5")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.25")
    return Settings()


@pytest.fixture
def store(test_settings: Settings) -> VectorStoreManager:
    return VectorStoreManager(settings=test_settings)


@pytest.fixture
def sample_chunk() -> DocumentChunk:
    """A single valid DocumentChunk for use across tests."""
    metadata = ChunkMetadata(
        topic="LSTM",
        difficulty="intermediate",
        type="concept_explanation",
        source="test_lstm.md",
        related_topics=["RNN", "vanishing_gradient"],
        is_bonus=False,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_lstm.md", "test content"),
        chunk_text=(
            "Long Short-Term Memory networks solve the vanishing gradient problem "
            "through gated mechanisms: the forget gate, input gate, and output gate. "
            "These gates control information flow through the cell state, allowing "
            "the network to maintain relevant information across long sequences."
        ),
        metadata=metadata,
    )


@pytest.fixture
def bonus_chunk() -> DocumentChunk:
    """A bonus topic chunk (GAN) for testing is_bonus filtering."""
    metadata = ChunkMetadata(
        topic="GAN",
        difficulty="advanced",
        type="architecture",
        source="test_gan.md",
        related_topics=["autoencoder", "generative_models"],
        is_bonus=True,
    )
    return DocumentChunk(
        chunk_id=VectorStoreManager.generate_chunk_id("test_gan.md", "gan content"),
        chunk_text=(
            "Generative Adversarial Networks consist of two competing neural networks: "
            "a generator that produces synthetic data and a discriminator that "
            "distinguishes real from generated samples. Training is a minimax game."
        ),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Chunk ID Generation Tests
# ---------------------------------------------------------------------------


class TestChunkIdGeneration:
    """Tests for the deterministic chunk ID generation logic."""

    def test_same_content_produces_same_id(self) -> None:
        """Identical source and text must always produce the same ID."""
        id1 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
        id2 = VectorStoreManager.generate_chunk_id("lstm.md", "same content")
        assert id1 == id2

    def test_different_content_produces_different_id(self) -> None:
        """Different text must produce different IDs."""
        id1 = VectorStoreManager.generate_chunk_id("lstm.md", "content one")
        id2 = VectorStoreManager.generate_chunk_id("lstm.md", "content two")
        assert id1 != id2

    def test_different_source_produces_different_id(self) -> None:
        """Same text from different sources must produce different IDs."""
        id1 = VectorStoreManager.generate_chunk_id("file_a.md", "same text")
        id2 = VectorStoreManager.generate_chunk_id("file_b.md", "same text")
        assert id1 != id2

    def test_id_is_16_characters(self) -> None:
        """Generated IDs must be exactly 16 hex characters."""
        chunk_id = VectorStoreManager.generate_chunk_id("source.md", "text")
        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)


# ---------------------------------------------------------------------------
# Duplicate Detection Tests
# ---------------------------------------------------------------------------


class TestDuplicateDetection:
    """
    Tests for the check_duplicate method.

    Interview talking point: these tests verify the core invariant
    of the duplicate guard — the system must never silently ingest
    the same content twice.
    """

    def test_new_chunk_is_not_duplicate(
        self, store: VectorStoreManager, sample_chunk: DocumentChunk
    ) -> None:
        """A chunk that has never been ingested must not be flagged as duplicate."""
        assert store.check_duplicate(sample_chunk.chunk_id) is False

    def test_ingested_chunk_is_duplicate(
        self, store: VectorStoreManager, sample_chunk: DocumentChunk
    ) -> None:
        """A chunk that has been ingested must be flagged as duplicate on re-check."""
        store.ingest([sample_chunk])
        assert store.check_duplicate(sample_chunk.chunk_id) is True

    def test_ingestion_skips_duplicate(
        self, store: VectorStoreManager, sample_chunk: DocumentChunk
    ) -> None:
        """Ingesting the same chunk twice must result in skipped=1 on second call."""
        first = store.ingest([sample_chunk])
        assert first.ingested == 1
        second = store.ingest([sample_chunk])
        assert second.skipped == 1
        assert second.ingested == 0


# ---------------------------------------------------------------------------
# Retrieval Tests
# ---------------------------------------------------------------------------


class TestRetrieval:
    """
    Tests for the query method.

    These cover the hallucination guard threshold and metadata filtering,
    both of which are common interview discussion topics.
    """

    def test_relevant_query_returns_results(
        self, store: VectorStoreManager, sample_chunk: DocumentChunk
    ) -> None:
        """A query semantically similar to an ingested chunk must return results."""
        store.ingest([sample_chunk])
        hits = store.query("LSTM forget gate mechanism vanishing gradient")
        assert len(hits) > 0

    def test_irrelevant_query_returns_empty(
        self,
        sample_chunk: DocumentChunk,
        monkeypatch,
        tmp_path,
    ) -> None:
        """
        A query with no semantic similarity to the corpus must return empty list.

        This tests the hallucination guard threshold. The system must return
        an empty list — not low-quality chunks — when nothing matches.
        """
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "chroma_strict"))
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_dl_corpus_strict")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.85")
        monkeypatch.setenv("RETRIEVAL_K", "5")
        store = VectorStoreManager(settings=Settings())
        store.ingest([sample_chunk])
        hits = store.query("history of the roman empire marble monuments")
        assert hits == []

    def test_topic_filter_restricts_results(
        self,
        store: VectorStoreManager,
        sample_chunk: DocumentChunk,
        bonus_chunk: DocumentChunk,
    ) -> None:
        """Results with topic_filter='LSTM' must not include GAN chunks."""
        store.ingest([sample_chunk, bonus_chunk])
        hits = store.query(
            "neural network architecture training",
            topic_filter="LSTM",
        )
        assert hits
        assert all(c.metadata.topic == "LSTM" for c in hits)

    def test_results_sorted_by_score_descending(
        self, store: VectorStoreManager, sample_chunk: DocumentChunk
    ) -> None:
        """Retrieved chunks must be sorted with highest similarity first."""
        meta2 = ChunkMetadata(
            topic="LSTM",
            difficulty="intermediate",
            type="architecture",
            source="test_lstm_extra.md",
            related_topics=["RNN"],
            is_bonus=False,
        )
        extra = DocumentChunk(
            chunk_id=VectorStoreManager.generate_chunk_id(
                "test_lstm_extra.md",
                "The output gate in an LSTM decides what parts of the cell state "
                "are exposed to the next hidden state and downstream layers.",
            ),
            chunk_text=(
                "The output gate in an LSTM decides what parts of the cell state "
                "are exposed to the next hidden state and downstream layers."
            ),
            metadata=meta2,
        )
        store.ingest([sample_chunk, extra])
        hits = store.query("LSTM output gate cell state hidden state", k=4)
        assert len(hits) >= 2
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
