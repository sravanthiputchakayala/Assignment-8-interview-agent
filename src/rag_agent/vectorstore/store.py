"""
store.py
========
ChromaDB vector store management.

Handles all interactions with the persistent ChromaDB collection:
initialisation, ingestion, duplicate detection, and retrieval.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings


def _distance_to_similarity(distance: float) -> float:
    """Chroma cosine space returns lower distance for closer vectors."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


@lru_cache(maxsize=1)
def get_default_vector_store() -> "VectorStoreManager":
    """
    Process-wide singleton so embedding models and the Chroma client load once.

    Tests should construct `VectorStoreManager(settings=...)` directly.
    """
    return VectorStoreManager()


class VectorStoreManager:
    """
    Manages the ChromaDB persistent vector store for the corpus.

    All corpus ingestion and retrieval operations pass through this class.
    It is the single point of contact between the application and ChromaDB.

    Parameters
    ----------
    settings : Settings, optional
        Application settings. Uses get_settings() singleton if not provided.

    Example
    -------
    >>> manager = VectorStoreManager()
    >>> result = manager.ingest(chunks)
    >>> print(f"Ingested: {result.ingested}, Skipped: {result.skipped}")
    >>>
    >>> chunks = manager.query("explain the vanishing gradient problem", k=4)
    >>> for chunk in chunks:
    ...     print(chunk.to_citation(), chunk.score)
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _initialise(self) -> None:
        """
        Create or connect to the persistent ChromaDB client and collection.

        Creates the chroma_db_path directory if it does not exist.
        Uses PersistentClient so data survives between application restarts.

        Called automatically during __init__. Should not be called directly.

        Raises
        ------
        RuntimeError
            If ChromaDB cannot be initialised at the configured path.
        """
        try:
            db_path = Path(self._settings.chroma_db_path)
            db_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(db_path))
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            count = self._collection.count()
            logger.info(
                "ChromaDB ready: collection={!r}, items={}",
                self._settings.chroma_collection_name,
                count,
            )
        except Exception as e:
            msg = f"Failed to initialise ChromaDB at {self._settings.chroma_db_path}"
            logger.exception(msg)
            raise RuntimeError(msg) from e

    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        """
        Generate a deterministic chunk ID from source filename and content.

        Using a content hash ensures two uploads of the same file produce
        the same IDs, making duplicate detection reliable regardless of
        filename changes.

        Parameters
        ----------
        source : str
            The source filename (e.g. 'lstm.md').
        chunk_text : str
            The full text content of the chunk.

        Returns
        -------
        str
            A 16-character hex string derived from SHA-256 of the inputs.
        """
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def check_duplicate(self, chunk_id: str) -> bool:
        """
        Check whether a chunk with this ID already exists in the collection.

        Parameters
        ----------
        chunk_id : str
            The deterministic chunk ID to check.

        Returns
        -------
        bool
            True if the chunk already exists (duplicate). False otherwise.

        Interview talking point: content-addressed deduplication is more
        robust than filename-based deduplication because it detects identical
        content even when files are renamed or re-uploaded.
        """
        res = self._collection.get(ids=[chunk_id], include=[])
        return bool(res.get("ids"))

    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        """
        Embed and store a list of DocumentChunks in ChromaDB.

        Checks each chunk for duplicates before embedding. Skips duplicates
        silently and records the count in the returned IngestionResult.

        Parameters
        ----------
        chunks : list[DocumentChunk]
            Prepared chunks with text and metadata. Use DocumentChunker
            to produce these from raw files.

        Returns
        -------
        IngestionResult
            Summary with counts of ingested, skipped, and errored chunks.

        Notes
        -----
        Embeds in batches of 100 to avoid memory issues with large corpora.
        Uses upsert (not add) so re-ingestion of modified content updates
        existing chunks rather than raising an error.

        Interview talking point: batch processing with a configurable
        batch size is a production pattern that prevents OOM errors when
        ingesting large document sets.
        """
        result = IngestionResult()
        batch_size = 100
        pending: list[DocumentChunk] = []

        def flush_batch() -> None:
            nonlocal pending
            if not pending:
                return
            texts = [c.chunk_text for c in pending]
            try:
                vectors = self._embeddings.embed_documents(texts)
            except Exception as e:
                logger.exception("Embedding batch failed")
                for c in pending:
                    result.errors.append(f"{c.metadata.source}: {e!s}")
                pending = []
                return
            ids = [c.chunk_id for c in pending]
            metadatas = [c.metadata.to_dict() for c in pending]
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )
            srcs = {c.metadata.source for c in pending}
            for s in srcs:
                if s not in result.document_ids:
                    result.document_ids.append(s)
            result.ingested += len(pending)
            pending = []

        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue
                pending.append(chunk)
                if len(pending) >= batch_size:
                    flush_batch()
            except Exception as e:
                logger.exception("Ingest failed for chunk")
                result.errors.append(f"{chunk.metadata.source}: {e!s}")

        flush_batch()
        logger.info(
            "Ingest complete: ingested={}, skipped={}, errors={}",
            result.ingested,
            result.skipped,
            len(result.errors),
        )
        return result

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Applies similarity threshold filtering — chunks below
        settings.similarity_threshold are excluded from results.

        Parameters
        ----------
        query_text : str
            The user query or rewritten query to retrieve against.
        k : int, optional
            Number of chunks to retrieve. Defaults to settings.retrieval_k.
        topic_filter : str, optional
            Restrict retrieval to a specific topic (e.g. 'LSTM').
            Maps to ChromaDB where-filter on metadata.topic.
        difficulty_filter : str, optional
            Restrict retrieval to a difficulty level.
            Maps to ChromaDB where-filter on metadata.difficulty.

        Returns
        -------
        list[RetrievedChunk]
            Chunks sorted by similarity score descending.
            Empty list if no chunks meet the similarity threshold.

        Interview talking point: returning an empty list (not hallucinating)
        when no relevant context exists is the hallucination guard. This is
        a critical production RAG pattern — the system must know what it
        does not know.
        """
        k = k or self._settings.retrieval_k
        where: dict | None = None
        if topic_filter and difficulty_filter:
            where = {
                "$and": [
                    {"topic": topic_filter},
                    {"difficulty": difficulty_filter},
                ]
            }
        elif topic_filter:
            where = {"topic": topic_filter}
        elif difficulty_filter:
            where = {"difficulty": difficulty_filter}

        query_embedding = self._embeddings.embed_query(query_text)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = raw.get("ids") or []
        docs_list = raw.get("documents") or []
        meta_list = raw.get("metadatas") or []
        dist_list = raw.get("distances") or []

        if not ids_list or not ids_list[0]:
            return []

        out: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            ids_list[0],
            docs_list[0],
            meta_list[0],
            dist_list[0],
            strict=False,
        ):
            score = _distance_to_similarity(dist)
            if score < self._settings.similarity_threshold:
                continue
            if doc is None or meta is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=doc,
                    metadata=ChunkMetadata.from_dict(meta),
                    score=score,
                )
            )

        out.sort(key=lambda c: c.score, reverse=True)
        return out

    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """
        Return a list of all unique source documents in the collection.

        Used by the UI to populate the document viewer panel.

        Returns
        -------
        list[dict]
            Each item contains: source (str), topic (str), chunk_count (int).
        """
        raw = self._collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas") or []
        by_source: dict[str, dict] = {}
        for meta in metadatas:
            if not meta:
                continue
            src = meta.get("source", "unknown")
            if src not in by_source:
                by_source[src] = {
                    "source": src,
                    "topic": meta.get("topic", ""),
                    "chunk_count": 0,
                }
            by_source[src]["chunk_count"] += 1
        return sorted(by_source.values(), key=lambda x: x["source"].lower())

    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        """
        Retrieve all chunks belonging to a specific source document.

        Used by the document viewer to display document content.

        Parameters
        ----------
        source : str
            The source filename to retrieve chunks for.

        Returns
        -------
        list[DocumentChunk]
            All chunks from this source, ordered by their position
            in the original document.
        """
        raw = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        ids = raw.get("ids") or []
        docs = raw.get("documents") or []
        metas = raw.get("metadatas") or []
        chunks: list[DocumentChunk] = []
        for cid, doc, meta in zip(ids, docs, metas, strict=False):
            if doc is None or meta is None:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    chunk_text=doc,
                    metadata=ChunkMetadata.from_dict(meta),
                )
            )
        chunks.sort(key=lambda c: c.chunk_id)
        return chunks

    def get_collection_stats(self) -> dict:
        """
        Return summary statistics about the current collection.

        Used by the UI to show corpus health at a glance.

        Returns
        -------
        dict
            Keys: total_chunks, topics (list), sources (list),
            bonus_topics_present (bool).
        """
        raw = self._collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas") or []
        topics: set[str] = set()
        sources: set[str] = set()
        bonus = False
        for meta in metadatas:
            if not meta:
                continue
            if meta.get("topic"):
                topics.add(meta["topic"])
            if meta.get("source"):
                sources.add(meta["source"])
            if str(meta.get("is_bonus", "false")).lower() == "true":
                bonus = True
        return {
            "total_chunks": len(metadatas),
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus,
        }

    def delete_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source document.

        Parameters
        ----------
        source : str
            Source filename to remove.

        Returns
        -------
        int
            Number of chunks deleted.
        """
        existing = self._collection.get(
            where={"source": source}, include=["metadatas"]
        )
        n = len(existing.get("ids") or [])
        if n:
            self._collection.delete(where={"source": source})
            logger.info("Deleted {} chunks for source {!r}", n, source)
        return n
