from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import faiss
from src.pipeline.core import Step, Context
from src.pipeline.utils.fs import ensure_dir
from src.pipeline.utils.logging import get_logger


class BuildFaissIndexStep(Step):
    """Build and save a FAISS ANN index + metadata mapping (cosine via IP on normalized vectors)."""
    name = "BuildFaissIndex"

    def run(self, ctx: Context) -> None:
        log = get_logger(self.name)

        emb_dir = Path(ctx.artifacts.get("embeddings_dir") or ctx.cfg["paths"]["embeddings_dir"])
        out_dir = ensure_dir(ctx.cfg["paths"]["vector_dataset_dir"])

        emb_path = emb_dir / "embeddings.npy"
        csv_path = emb_dir / "embeddings.csv"
        if not emb_path.exists() or not csv_path.exists():
            log.warning("Embeddings not found in %s", emb_dir)
            return

        X = np.load(emb_path).astype("float32")
        df = pd.read_csv(csv_path)
        names = df["filename"].tolist()

        n, d = X.shape
        if n == 0:
            log.warning("No vectors to index.")
            return

        # Clean up any NaNs/Infs
        if not np.isfinite(X).all():
            log.warning("Found non-finite values in embeddings; replacing with 0.")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize for cosine via inner product
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        xb = X / norms

        # Choose index
        use_ivf = n >= 10_000
        if use_ivf:
            # if user provided nlist, use it; else pick a reasonable default ~ 4 * sqrt(n)
            cfg_nlist = int(ctx.cfg.get("faiss", {}).get("nlist", 0))
            nlist = cfg_nlist if cfg_nlist > 0 else max(1, int(4 * (n ** 0.5)))
            nlist = min(nlist, n)

            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train on a subset if huge (FAISS requires training set >= nlist)
            train_size = min(n, max(nlist, 100_000))
            sel = np.random.default_rng(42).choice(n, size=train_size, replace=False) if train_size < n else np.arange(n)
            index.train(xb[sel])
            index.add(xb)
            index_type = "IVFFlat(IP)"
        else:
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            index_type = "FlatIP"

        # Save index
        index_path = out_dir / "vector_index.faiss"
        faiss.write_index(index, str(index_path))

        # Save metadata (id -> filename) plus index info
        meta = {
            "id_to_filename": {int(i): names[i] for i in range(len(names))},
            "info": {
                "n": int(n),
                "d": int(d),
                "metric": "cosine_via_inner_product_on_L2_normalized",
                "index_type": index_type,
                "nlist": int(index.nlist) if hasattr(index, "nlist") else None,
                "normalized": True,
            },
        }
        meta_path = out_dir / "vector_metadata.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        log.info("Saved index → %s", index_path)
        log.info("Saved metadata → %s", meta_path)

        ctx.artifacts["faiss_index_path"] = str(index_path)
        ctx.artifacts["faiss_metadata_path"] = str(meta_path)
