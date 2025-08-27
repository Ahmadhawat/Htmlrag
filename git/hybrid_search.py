from __future__ import annotations

"""
HybridSearchStep
Runs FAISS (vectors) + Whoosh (BM25) and fuses results.

Config it reads:
  paths.vector_dataset_dir  -> vector_index.faiss + vector_metadata.json
  paths.lexical_index_dir   -> Whoosh index directory
  paths.hybrid_results_path -> output JSON path (optional, has default)
  embedding.model_name      -> SentenceTransformer model for query embedding
  embedding.cache_folder    -> local cache folder
  query.text                -> query (or pass artifacts["query_text"])
  query.k, query.k_vec, query.k_lex
  fusion.method             -> "minmax" or "rrf"
  fusion.alpha              -> weight for vectors if minmax
  faiss.nprobe              -> IVF search breadth (if using IVF index)
  e5_prefixes.use           -> if True, prefix query with "query: " (for E5 models)
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple

import faiss
import numpy as np
from whoosh import index as whoosh_index
from whoosh.qparser import QueryParser
from whoosh.query import Every
from sentence_transformers import SentenceTransformer

from src.pipeline.core import Step, Context
from src.pipeline.utils.logging import get_logger


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype("float32")
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def _is_ivf(idx) -> bool:
    return isinstance(idx, (faiss.IndexIVFFlat, faiss.IndexIVFPQ, faiss.IndexIVFScalarQuantizer))


def _minmax(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _rrf(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


class HybridSearchStep(Step):
    name = "HybridSearch"

    def run(self, ctx: Context) -> None:
        log = get_logger(self.name)

        # ---------- Paths & config ----------
        vroot = Path(ctx.cfg["paths"]["vector_dataset_dir"])
        lroot = Path(ctx.cfg["paths"]["lexical_index_dir"])

        index_path = vroot / "vector_index.faiss"
        meta_path = vroot / "vector_metadata.json"
        if not index_path.exists() or not meta_path.exists():
            log.error("Missing FAISS index or metadata in %s", vroot)
            return
        if not lroot.exists():
            log.error("Missing lexical index dir: %s", lroot)
            return

        qcfg = ctx.cfg.get("query", {}) or {}
        fusion = ctx.cfg.get("fusion", {}) or {}
        k = int(qcfg.get("k", 10))
        k_vec = int(qcfg.get("k_vec", 30))
        k_lex = int(qcfg.get("k_lex", 30))
        method = str(fusion.get("method", "minmax")).lower()
        alpha = float(fusion.get("alpha", 0.5))

        # Query text
        qtext = ctx.artifacts.get("query_text") or qcfg.get("text")
        if not qtext:
            log.error("No query text provided. Set artifacts['query_text'] or cfg.query.text")
            return

        # ---------- Embedder (local) ----------
        ecfg = ctx.cfg["embedding"]
        model_name = ecfg["model_name"]
        cache_folder = ecfg.get("cache_folder", "./data/models/st")
        model = SentenceTransformer(model_name, cache_folder=cache_folder)

        e5p = ctx.cfg.get("e5_prefixes", {}) or {}
        use_e5 = bool(e5p.get("use", False))
        qprefix = e5p.get("query_prefix", "query: ")

        # ---------- FAISS search ----------
        index = faiss.read_index(str(index_path))
        if _is_ivf(index):
            nprobe = int(ctx.cfg.get("faiss", {}).get("nprobe", 32))
            try:
                faiss.ParameterSpace().set_index_parameter(index, "nprobe", nprobe)
            except Exception:
                try:
                    index.nprobe = nprobe  # type: ignore[attr-defined]
                except Exception:
                    pass

        q_in = f"{qprefix}{qtext}" if use_e5 else qtext
        qv = model.encode([q_in], show_progress_bar=False)
        qv = _l2_normalize(np.asarray(qv, dtype="float32"))

        D, I = index.search(qv, k_vec)  # inner product on normalized vectors ≈ cosine similarity
        vec_scores: Dict[int, float] = {}
        for doc_id, score in zip(I[0].tolist(), D[0].tolist()):
            if doc_id >= 0:
                vec_scores[int(doc_id)] = float(score)

        # ---------- BM25 search ----------
        ix = whoosh_index.open_dir(lroot)
        with ix.searcher() as searcher:
            parser = QueryParser("text", schema=ix.schema)
            try:
                q = parser.parse(qtext)
            except Exception:
                q = Every()
            hits = searcher.search(q, limit=k_lex)
            lex_scores: Dict[int, float] = {int(h["doc_id"]): float(h.score) for h in hits}

        # ---------- Fusion ----------
        all_ids = set(vec_scores) | set(lex_scores)
        fused_pairs: List[Tuple[int, float]] = []

        if method == "rrf":
            vec_sorted = sorted(vec_scores.items(), key=lambda x: x[1], reverse=True)
            lex_sorted = sorted(lex_scores.items(), key=lambda x: x[1], reverse=True)
            vec_rank = {doc_id: r for r, (doc_id, _) in enumerate(vec_sorted, start=1)}
            lex_rank = {doc_id: r for r, (doc_id, _) in enumerate(lex_sorted, start=1)}
            for did in all_ids:
                s = _rrf(vec_rank.get(did, 10**9)) + _rrf(lex_rank.get(did, 10**9))
                fused_pairs.append((did, s))
        else:
            vnorm = _minmax(vec_scores)
            lnorm = _minmax(lex_scores)
            for did in all_ids:
                s = alpha * vnorm.get(did, 0.0) + (1.0 - alpha) * lnorm.get(did, 0.0)
                fused_pairs.append((did, s))

        fused_pairs.sort(key=lambda x: x[1], reverse=True)
        top_ids = [did for did, _ in fused_pairs[:k]]

        # ---------- Attach metadata & write JSON (supports nested or flat meta) ----------
        raw_meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        # Support either:
        # 1) flat:  {"0": "file_chunk0000.txt", ...}
        # 2) nested: {"id_to_filename": {"0": "file_chunk0000.txt", ...}, "info": {...}}
        id2name = raw_meta.get("id_to_filename", raw_meta)

        def meta_get(did: int) -> str:
            return id2name.get(str(did)) or id2name.get(did) or ""

        results = []
        fused_map = dict(fused_pairs)
        for did in top_ids:
            results.append({
                "doc_id": int(did),
                "filename": meta_get(did),
                "scores": {
                    "vector": float(vec_scores.get(did, 0.0)),
                    "lexical": float(lex_scores.get(did, 0.0)),
                    "hybrid": float(fused_map.get(did, 0.0)),
                }
            })

        out_path = Path(ctx.cfg["paths"].get("hybrid_results_path", "./data/workspace/hybrid_results.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"query": qtext, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("Hybrid top-%d written to %s", k, out_path)
        ctx.artifacts["hybrid_results_path"] = str(out_path)
