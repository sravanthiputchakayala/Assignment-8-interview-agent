"""
chunker.py
==========
Document loading and chunking pipeline.

Handles ingestion of raw files (PDF and Markdown) into structured
DocumentChunk objects ready for embedding and vector store storage.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.config import Settings, get_settings
from rag_agent.vectorstore.store import VectorStoreManager


class DocumentChunker:
    """
    Loads raw documents and splits them into DocumentChunk objects.

    Supports PDF and Markdown file formats. Chunking strategy uses
    recursive character splitting with configurable chunk size and
    overlap — both are interview-defensible parameters.

    Parameters
    ----------
    settings : Settings, optional
        Application settings.

    Example
    -------
    >>> chunker = DocumentChunker()
    >>> chunks = chunker.chunk_file(
    ...     Path("data/corpus/lstm.md"),
    ...     metadata_overrides={"topic": "LSTM", "difficulty": "intermediate"}
    ... )
    >>> print(f"Produced {len(chunks)} chunks")
    """

    # Default chunking parameters — justify these in your architecture diagram.
    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # -----------------------------------------------------------------------
    # Public Interface
    # -----------------------------------------------------------------------

    def chunk_file(
        self,
        file_path: Path,
        metadata_overrides: dict | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[DocumentChunk]:
        """
        Load a file and split it into DocumentChunks.

        Automatically detects file type and routes to the appropriate
        loader. Applies metadata_overrides on top of auto-detected
        metadata where provided.

        Parameters
        ----------
        file_path : Path
            Absolute or relative path to the source file.
        metadata_overrides : dict, optional
            Metadata fields to set or override. Keys must match
            ChunkMetadata field names. Commonly used to set topic
            and difficulty when the file does not encode these.
        chunk_size : int
            Maximum characters per chunk.
        chunk_overlap : int
            Characters of overlap between adjacent chunks.

        Returns
        -------
        list[DocumentChunk]
            Fully prepared chunks with deterministic IDs and metadata.

        Raises
        ------
        ValueError
            If the file type is not supported.
        FileNotFoundError
            If the file does not exist at the given path.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        base_meta = self._infer_metadata(path, metadata_overrides)

        if suffix == ".pdf":
            raw = self._chunk_pdf(path, chunk_size, chunk_overlap)
        elif suffix in (".md", ".markdown"):
            raw = self._chunk_markdown(path, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported file type {suffix!r} for {path.name}")

        chunks: list[DocumentChunk] = []
        for item in raw:
            text = item["text"].strip()
            if len(text) < 40:
                continue
            cid = VectorStoreManager.generate_chunk_id(base_meta.source, text)
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    chunk_text=text,
                    metadata=base_meta,
                )
            )
        return chunks

    def chunk_files(
        self,
        file_paths: list[Path],
        metadata_overrides: dict | None = None,
    ) -> list[DocumentChunk]:
        """
        Chunk multiple files in a single call.

        Used by the UI multi-file upload handler to process all
        uploaded files before passing to VectorStoreManager.ingest().

        Parameters
        ----------
        file_paths : list[Path]
            List of file paths to process.
        metadata_overrides : dict, optional
            Applied to all files. Per-file metadata should be handled
            by calling chunk_file() individually.

        Returns
        -------
        list[DocumentChunk]
            Combined chunks from all files, preserving source attribution
            in each chunk's metadata.
        """
        combined: list[DocumentChunk] = []
        for fp in file_paths:
            try:
                combined.extend(self.chunk_file(fp, metadata_overrides))
            except Exception as e:
                logger.error("chunk_files: failed for {}: {}", fp, e)
        return combined

    # -----------------------------------------------------------------------
    # Format-Specific Loaders
    # -----------------------------------------------------------------------

    def _chunk_pdf(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a PDF file.

        Uses PyPDFLoader for text extraction followed by
        RecursiveCharacterTextSplitter for chunking.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'page' keys before conversion
            to DocumentChunk objects.
        """
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = splitter.split_documents(docs)
        return [
            {
                "text": d.page_content,
                "page": d.metadata.get("page", 0),
            }
            for d in splits
        ]

    def _chunk_markdown(
        self,
        file_path: Path,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[dict]:
        """
        Load and chunk a Markdown file.

        Uses MarkdownHeaderTextSplitter first to respect document
        structure (headers create natural chunk boundaries), then
        RecursiveCharacterTextSplitter for oversized sections.

        Parameters
        ----------
        file_path : Path
        chunk_size : int
        chunk_overlap : int

        Returns
        -------
        list[dict]
            Raw dicts with 'text' and 'header' keys.
        """
        text = file_path.read_text(encoding="utf-8", errors="replace")
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
        )
        md_docs = md_splitter.split_text(text)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not md_docs:
            plain = splitter.create_documents([text])
            return [{"text": d.page_content, "header": ""} for d in plain]
        final = splitter.split_documents(md_docs)
        return [
            {"text": d.page_content, "header": repr(d.metadata)} for d in final
        ]

    # -----------------------------------------------------------------------
    # Metadata Inference
    # -----------------------------------------------------------------------

    def _infer_metadata(
        self,
        file_path: Path,
        overrides: dict | None = None,
    ) -> ChunkMetadata:
        """
        Infer chunk metadata from filename conventions and apply overrides.

        Filename convention (recommended to Corpus Architects):
          <topic>_<difficulty>.md or <topic>_<difficulty>.pdf
          e.g. lstm_intermediate.md, alexnet_advanced.pdf

        If the filename does not follow this convention, defaults are
        applied and the Corpus Architect must provide overrides manually.

        Parameters
        ----------
        file_path : Path
            Source file path used to infer topic and difficulty.
        overrides : dict, optional
            Explicit metadata values that take precedence over inference.

        Returns
        -------
        ChunkMetadata
            Populated metadata object.
        """
        stem = file_path.stem
        topic = "General"
        difficulty = "intermediate"
        topic_slugs = {
            "ann": "ANN",
            "cnn": "CNN",
            "rnn": "RNN",
            "lstm": "LSTM",
            "seq2seq": "Seq2Seq",
            "autoencoder": "Autoencoder",
            "gan": "GAN",
            "som": "SOM",
            "boltzmann": "BoltzmannMachine",
            "boltzmannmachine": "BoltzmannMachine",
        }
        difficulties = {"beginner", "intermediate", "advanced"}
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].lower() in difficulties:
            slug, difficulty = parts[0].lower(), parts[1].lower()
            topic = topic_slugs.get(slug, parts[0].upper())
        else:
            slug = stem.lower().replace(" ", "_")
            topic = topic_slugs.get(slug, stem.replace("_", " ").title()[:48])

        bonus_topics = {"GAN", "SOM", "BoltzmannMachine"}
        is_bonus = topic in bonus_topics

        meta = ChunkMetadata(
            topic=topic,
            difficulty=difficulty,
            type="concept_explanation",
            source=file_path.name,
            related_topics=[],
            is_bonus=is_bonus,
        )
        if overrides:
            for key, val in overrides.items():
                if hasattr(meta, key):
                    setattr(meta, key, val)
        return meta
