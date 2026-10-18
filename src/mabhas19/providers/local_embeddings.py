"""Local CPU embedding provider (sentence-transformers), for offline retrieval experiments.

Requires the optional dependency group ``embeddings`` (CPU PyTorch + sentence-transformers);
the import is lazy so the core install never needs it. The model is loaded through the normal
Hugging Face path (safetensors weights where available) into the user's HF cache; nothing is
called over a paid API.

E5 models are trained for asymmetric retrieval: queries are embedded as ``"query: " + text`` and
passages as ``"passage: " + text``. The prefixes are model input only; stored chunk text and
retrieval evidence are never changed. Embeddings are L2-normalized, so the dot product equals
cosine similarity. Inputs longer than the model's maximum sequence length are truncated by the
tokenizer (the model's documented behaviour); :meth:`token_lengths` audits this beforehand.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from mabhas19.core.manifest import sha256_file

E5_SMALL = "intfloat/multilingual-e5-small"
E5_SMALL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"  # frozen candidate
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
_PACKAGES = ("torch", "sentence-transformers", "transformers", "tokenizers", "huggingface-hub")


class SentenceTransformerEmbedder:
    """:class:`~mabhas19.providers.embeddings.Embedder` over a local sentence-transformers model."""

    def __init__(  # noqa: PLR0913
        self,
        model_id: str = E5_SMALL,
        *,
        revision: str | None = E5_SMALL_REVISION,
        query_prefix: str = QUERY_PREFIX,
        passage_prefix: str = PASSAGE_PREFIX,
        threads: int | None = None,
        batch_size: int = 32,
    ) -> None:
        import torch  # noqa: PLC0415
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        if threads:
            torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        self.model = SentenceTransformer(model_id, revision=revision, device="cpu")
        self.model.eval()
        self.model_id = model_id
        self.pinned_revision = revision
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.batch_size = batch_size
        self.dimensions = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)
        self.threads = int(torch.get_num_threads())
        self.name = f"{model_id}@{self.revision() or 'unknown'}"

    # ------------------------------------------------------------------ Embedder protocol

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vectors]

    # ------------------------------------------------------------------ audit / provenance

    def token_lengths(self, texts: Sequence[str], *, passages: bool = True) -> list[int]:
        """Token counts (special tokens included) of the exact model inputs, before truncation."""
        prefix = self.passage_prefix if passages else self.query_prefix
        enc = self.model.tokenizer([prefix + t for t in texts], add_special_tokens=True)
        return [len(ids) for ids in enc["input_ids"]]

    def snapshot_dir(self) -> Path | None:
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        try:
            return Path(
                snapshot_download(
                    self.model_id, revision=self.pinned_revision, local_files_only=True
                )
            )
        except Exception:
            return None

    def revision(self) -> str | None:
        snap = self.snapshot_dir()
        return snap.name if snap else None

    def provenance(self) -> dict[str, Any]:
        snap = self.snapshot_dir()
        weights: list[dict[str, Any]] = []
        if snap:
            for f in sorted(snap.rglob("*")):
                if f.suffix in (".safetensors", ".bin") and f.is_file():
                    weights.append(
                        {
                            "file": f.relative_to(snap).as_posix(),
                            "bytes": f.stat().st_size,
                            # computed: on Windows the HF cache holds copies, not blob symlinks
                            "sha256": sha256_file(f),
                        }
                    )
        return {
            "model_id": self.model_id,
            "revision": snap.name if snap else None,
            "weights_in_snapshot": weights,
            "device": "cpu",
            "threads": self.threads,
            "dimensions": self.dimensions,
            "max_seq_length": self.max_seq_length,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "normalized": True,
            "python": sys.version.split()[0],
            "platform": sys.platform,  # platform.platform() can hang on WMI under Windows
            "cpu_count": os.cpu_count(),
            "packages": {p: _version(p) for p in _PACKAGES},
        }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None
