from __future__ import annotations

"""
ChunkWithEmbedderTokenizerStep
------------------------------
Sentence-aware chunking that enforces token limits using the *same* tokenizer
as your embedding model (SentenceTransformers). This prevents silent truncation.

Reads:  txt_dir/*.txt
Writes: chunks_dir/<base>_chunk0000.txt, ...

Config it uses:
  paths.txt_dir
  paths.chunks_dir
  embedding.model_name
  embedding.cache_folder
  chunking.max_total_tokens      # total budget (tokens) including metadata
  chunking.metadata_tokens       # reserved tokens for metadata (titles/ids)
  chunking.overlap_tokens        # target overlap in tokens between chunks
  chunking.spacy_model           # e.g., "de_core_news_md" (sentence splitting)
"""

from pathlib import Path
from typing import List
from src.pipeline.core import Step, Context
from src.pipeline.utils import iter_files, fresh_dir
from src.pipeline.utils.logging import get_logger

from sentence_transformers import SentenceTransformer

# Sentence splitting
try:
    import spacy
except Exception:
    spacy = None


def _load_spacy(model_name: str):
    if not model_name:
        return None
    if spacy is None:
        raise RuntimeError(
            "spaCy not installed. Run: pip install spacy && python -m spacy download de_core_news_md"
        )
    return spacy.load(model_name)


class ChunkWithEmbedderTokenizerStep(Step):
    name = "ChunkWithEmbedderTokenizer"

    def run(self, ctx: Context) -> None:
        log = get_logger(self.name)

        # ---- Paths & config
        txt_root = Path(ctx.artifacts.get("txt_dir") or ctx.cfg["paths"]["txt_dir"])
        out_dir = fresh_dir(ctx.cfg["paths"]["chunks_dir"])

        ck = ctx.cfg["chunking"]
        max_total = int(ck["max_total_tokens"])           # e.g., 512
        meta_tokens = int(ck["metadata_tokens"])          # e.g., 60
        overlap_tokens = int(ck["overlap_tokens"])        # e.g., 80~100
        spacy_model = ck.get("spacy_model", "de_core_news_md")

        # ---- Load embedder (tokenizer only) to ensure *exact* counting
        ecfg = ctx.cfg["embedding"]
        model_name = ecfg["model_name"]
        cache_folder = ecfg.get("cache_folder", "./data/models/st")
        st_model = SentenceTransformer(model_name, cache_folder=cache_folder)

        # SentenceTransformers exposes `tokenize` (returns dict with input_ids)
        def tok_count(s: str) -> int:
            toks = st_model.tokenize([s])  # batch of 1
            ids = toks["input_ids"]
            # Depending on backend it can be list or tensor
            if hasattr(ids, "__iter__") and not hasattr(ids, "shape"):
                return len(ids[0])
            return int(ids.shape[-1])

        # Respect model's built-in max_seq_length if set
        model_max = getattr(st_model, "max_seq_length", None)
        if isinstance(model_max, int) and model_max > 0 and max_total > model_max:
            log.warning(
                "chunking.max_total_tokens (%d) > model.max_seq_length (%d). "
                "Capping to avoid truncation.", max_total, model_max
            )
            max_total = model_max

        content_budget = max(16, max_total - meta_tokens)  # ensure positive budget

        # ---- Sentence splitter (German-friendly)
        nlp = _load_spacy(spacy_model)
        produced = 0

        for f in iter_files(txt_root, exts=[".txt"]):
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue

            doc = nlp(text)
            sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
            if not sentences:
                sentences = [text]

            chunks: List[str] = []
            cur: List[str] = []
            cur_tokens = 0

            def join_and_count(parts: List[str]) -> int:
                if not parts:
                    return 0
                return tok_count(" ".join(parts))

            for sent in sentences:
                stoks = tok_count(sent)
                # If a single sentence is longer than budget, hard-split it by words
                if stoks > content_budget:
                    words = sent.split()
                    piece: List[str] = []
                    for w in words:
                        piece.append(w)
                        if tok_count(" ".join(piece)) > content_budget:
                            # backoff last token
                            if len(piece) > 1:
                                piece.pop()
                            # flush piece
                            if piece:
                                if cur:
                                    chunks.append(" ".join(cur))
                                    cur = []
                                chunks.append(" ".join(piece))
                                produced += 1
                            piece = [w]  # start new with last word
                    # add remainder
                    if piece:
                        if cur:
                            chunks.append(" ".join(cur))
                            cur = []
                        chunks.append(" ".join(piece))
                        produced += 1
                    cur_tokens = 0
                    continue

                # normal case: pack sentence into current chunk
                if cur_tokens + stoks <= content_budget:
                    cur.append(sent)
                    cur_tokens += stoks
                else:
                    # flush current chunk
                    if cur:
                        chunks.append(" ".join(cur))

                    # build token-based overlap from the tail sentences
                    overlap: List[str] = []
                    for s in reversed(cur):
                        if tok_count(" ".join(overlap + [s])) < overlap_tokens:
                            overlap.insert(0, s)
                        else:
                            break

                    cur = overlap + [sent]
                    cur_tokens = join_and_count(cur)

            if cur:
                chunks.append(" ".join(cur))

            base = f.stem
            for i, ch in enumerate(chunks):
                out_path = out_dir / f"{base}_chunk{i:04d}.txt"
                out_path.write_text(ch, encoding="utf-8")
                produced += 1

        log.info(
            "Chunked with embedder tokenizer → %s (total chunk files written: %d)",
            out_dir, produced
        )
        ctx.artifacts["chunks_dir"] = str(out_dir)
