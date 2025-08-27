from __future__ import annotations

"""
EmbedWithControlsStep
---------------------
Creates embeddings for chunk files using SentenceTransformers with strict, local control:
- Same tokenizer as the model (no hidden truncation surprises).
- Honors model.max_seq_length (warns if configs exceed it).
- Optional E5-style "passage: " prefix for better retrieval quality.
- Fully offline if your environment/cache has the model.

Inputs
  paths.chunks_dir
  paths.embeddings_dir
  embedding.model_name          (e.g., intfloat/multilingual-e5-base)
  embedding.batch_size
  embedding.cache_folder
  e5_prefixes.use               (bool) if True, prepend "passage: "
  e5_prefixes.passage_prefix    (str)  default: "passage: "

Outputs
  embeddings_dir/embeddings.npy    (float32, shape [n, d])
  embeddings_dir/embeddings.csv    (CSV: first column 'filename', then d dims)
  ctx.artifacts["embeddings_dir"]
"""

from pathlib import Path
from typing import List
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.pipeline.core import Step, Context
from src.pipeline.utils import iter_files, fresh_dir, ensure_dir
from src.pipeline.utils.logging import get_logger


class EmbedWithControlsStep(Step):
    name = "Embed"  # keep the same name so it plugs into your existing run.steps

    def run(self, ctx: Context) -> None:
        log = get_logger(self.name)

        # --- Paths & config
        chunk_root = Path(ctx.artifacts.get("chunks_dir") or ctx.cfg["paths"]["chunks_dir"])
        out_dir = fresh_dir(ctx.cfg["paths"]["embeddings_dir"])

        ecfg = ctx.cfg["embedding"]
        model_name = ecfg["model_name"]
        batch_size = int(ecfg.get("batch_size", 32))
        cache_folder = ecfg.get("cache_folder", "./data/models/st")
        ensure_dir(cache_folder)

        # --- Enforce offline if your env/models are pre-cached
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # --- Load model locally
        log.info("Loading embedder: %s (cache: %s)", model_name, cache_folder)
        model = SentenceTransformer(model_name, cache_folder=cache_folder)

        # --- Respect model's real token limit
        max_seq = getattr(model, "max_seq_length", None)
        if isinstance(max_seq, int) and max_seq > 0:
            log.info("Model max_seq_length = %d tokens", max_seq)
        else:
            log.info("Model max_seq_length unknown; relying on model defaults.")

        # --- Optional E5 prefixes for passages
        e5p = ctx.cfg.get("e5_prefixes", {}) or {}
        use_e5 = bool(e5p.get("use", False))
        passage_prefix = e5p.get("passage_prefix", "passage: ")

        # --- Collect chunk texts
        files: List[Path] = sorted(iter_files(chunk_root, exts=[".txt"]))
        texts: List[str] = []
        names: List[str] = []
        for p in files:
            t = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not t:
                continue
            names.append(p.name)
            texts.append((passage_prefix + t) if use_e5 else t)

        if not texts:
            log.warning("No chunk texts to embed in %s", chunk_root)
            return

        # --- Embed
        log.info("Embedding %d chunks (batch_size=%d)...", len(texts), batch_size)
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=False,  # keep raw; FAISS step will normalize for cosine/IP
        )
        emb = np.asarray(emb, dtype="float32")

        # --- Save artifacts
        np.save(out_dir / "embeddings.npy", emb)
        df = pd.DataFrame(emb)
        df.insert(0, "filename", names)
        df.to_csv(out_dir / "embeddings.csv", index=False, encoding="utf-8")

        log.info("Saved embeddings → %s", out_dir)
        ctx.artifacts["embeddings_dir"] = str(out_dir)
