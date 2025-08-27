from __future__ import annotations

"""
BuildLexicalIndexStep
Builds a Whoosh BM25 index over your chunk texts.
- Assumes embeddings.csv (row 0..n-1) matches embeddings.npy AND your FAISS ids.
- Uses the first column "filename" to read each chunk text from chunks_dir.
- Stores doc_id == row index, so hybrid fusion is trivial.
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd
from whoosh import index as whoosh_index
from whoosh.fields import Schema, TEXT, ID, NUMERIC
from whoosh.analysis import StandardAnalyzer

from src.pipeline.core import Step, Context
from src.pipeline.utils.fs import ensure_dir
from src.pipeline.utils.logging import get_logger

# OPTIONAL: light German lemmatization (can be disabled in config)
try:
    import spacy
except Exception:
    spacy = None


def _load_spacy(model_name: str):
    if not model_name:
        return None
    if spacy is None:
        raise RuntimeError(
            "spaCy not installed but lexical.lemmatize=True. "
            "Install with: pip install spacy && python -m spacy download de_core_news_md"
        )
    return spacy.load(model_name)


def _read_chunk_text(chunks_dir: Path, fname: str) -> Optional[str]:
    p = chunks_dir / fname
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore").strip()


def _preprocess_text(text: str, nlp, min_chars: int) -> str:
    """If nlp provided: lemmatize+lowercase. Else: simple lowercase+whitespace collapse."""
    if not text:
        return ""
    if nlp is None:
        return " ".join(text.lower().split())
    doc = nlp(text)
    toks: List[str] = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        lemma = (t.lemma_ or t.text).strip().lower()
        if len(lemma) >= max(1, min_chars):
            toks.append(lemma)
    return " ".join(toks)


class BuildLexicalIndexStep(Step):
    """Build a BM25 index aligned with FAISS ids (doc_id == row index)."""
    name = "BuildLexicalIndex"

    def run(self, ctx: Context) -> None:
        log = get_logger(self.name)

        emb_dir = Path(ctx.artifacts.get("embeddings_dir") or ctx.cfg["paths"]["embeddings_dir"])
        chunks_dir = Path(ctx.artifacts.get("chunks_dir") or ctx.cfg["paths"]["chunks_dir"])
        out_dir = ensure_dir(ctx.cfg["paths"]["lexical_index_dir"])

        csv_path = emb_dir / "embeddings.csv"
        if not csv_path.exists():
            log.error("embeddings.csv not found: %s", csv_path)
            return

        # Config
        lex_cfg = ctx.cfg.get("lexical", {}) if isinstance(ctx.cfg, dict) else {}
        do_lemma = bool(lex_cfg.get("lemmatize", True))
        spacy_model = lex_cfg.get("spacy_model", "de_core_news_md")
        min_chars = int(lex_cfg.get("min_chars", 2))
        nlp = _load_spacy(spacy_model) if do_lemma else None
        if do_lemma and nlp is None:
            log.warning("lemmatize=True but spaCy not available; falling back to raw text.")

        # Create/open Whoosh index
        schema = Schema(
            doc_id=NUMERIC(stored=True, unique=True),
            filename=ID(stored=True),
            text=TEXT(stored=True, analyzer=StandardAnalyzer()),
        )
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        if not any(out_dir.iterdir()):
            whoosh_index.create_in(out_dir, schema)
        ix = whoosh_index.open_dir(out_dir)

        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            log.error("embeddings.csv must have a 'filename' column as first column.")
            return

        writer = ix.writer(limitmb=512, procs=0)
        added = 0
        for i, row in enumerate(df.itertuples(index=False)):
            fname = getattr(row, "filename")
            raw = _read_chunk_text(chunks_dir, fname)
            if not raw:
                continue
            text_proc = _preprocess_text(raw, nlp, min_chars)
            writer.add_document(doc_id=int(i), filename=str(fname), text=text_proc)
            added += 1
            if added % 2000 == 0:
                writer.commit()
                writer = ix.writer(limitmb=512, procs=0)

        writer.commit()
        log.info("Built lexical index in %s (docs indexed: %d)", out_dir, added)
        ctx.artifacts["lexical_index_dir"] = str(out_dir)
