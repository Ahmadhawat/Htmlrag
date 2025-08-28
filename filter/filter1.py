# filter_html.py
from bs4 import BeautifulSoup, NavigableString
import argparse, sys, re
from pathlib import Path

INLINE_TAGS = {"a","b","strong","i","em","span","code","kbd","samp","u","small","sup","sub","mark"}
SKIP_CONTAINERS = {"script","style","noscript","svg","nav","header","footer","aside"}
BREADCRUMB_HINTS = ("sie sind hier", "zu hauptinhalt springen")

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def only_inline_children(tag) -> bool:
    """True if a block contains only inline tags / text (no block children)."""
    for c in tag.children:
        if isinstance(c, NavigableString):
            continue
        if getattr(c, "name", "").lower() not in INLINE_TAGS:
            return False
    return True

def is_block_heading_anchor(a_tag) -> bool:
    """
    Headline like <div><a>Text</a></div> OR <div><span><a>Text</a></span></div>.
    We check if the *container's* visible text equals the <a>'s text.
    """
    if not a_tag or not a_tag.get_text(strip=True):
        return False
    cont = a_tag
    # climb one level if the immediate parent is an inline wrapper
    if cont.parent and getattr(cont.parent, "name", "").lower() in INLINE_TAGS:
        cont = cont.parent
    parent = cont.parent
    if not parent or getattr(parent, "name", "").lower() not in {"div","td","th"}:
        return False
    parent_txt = normalize_ws(parent.get_text(" ", strip=True))
    link_txt = normalize_ws(a_tag.get_text(" ", strip=True))
    if not parent_txt or not link_txt:
        return False
    # treat as heading when the container text is basically just the link text
    return parent_txt == link_txt and only_inline_children(parent)

def looks_like_breadcrumb(text: str) -> bool:
    lo = (text or "").lower()
    return any(h in lo for h in BREADCRUMB_HINTS)

def filter_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content containers to reduce noise
    for tag in soup.find_all(SKIP_CONTAINERS):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    h1 = soup.find("h1")
    h1_text = normalize_ws(h1.get_text(" ", strip=True)) if h1 else ""

    # pick traversal root
    iterable = (h1.find_all_next(True) if h1 else (soup.body or soup).find_all(True))

    lines = []

    # If no <h1>, try to grab a heading immediately before first <pre>
    if not h1:
        pre = soup.find("pre")
        if pre:
            for prev in pre.find_all_previous():
                if prev.name in ("h2","h3","h4","h5","h6","div","td","th","p","b","strong","span"):
                    t = normalize_ws(prev.get_text(" ", strip=True))
                    if t and not looks_like_breadcrumb(t):
                        lines.append(t)
                        break

    for el in iterable:
        name = (el.name or "").lower()

        # Skip obvious crumbs
        if name in {"span","div"}:
            tpeek = normalize_ws(el.get_text(" ", strip=True))
            if looks_like_breadcrumb(tpeek):
                continue

        # Headings h1..h6
        if name in ("h1","h2","h3","h4","h5","h6"):
            txt = normalize_ws(el.get_text(" ", strip=True))

        # Table cells that *act* like headings: <th>…> or <td> with leading <b>/<strong>
        elif name in ("th","td"):
            raw = normalize_ws(el.get_text(" ", strip=True))
            if not raw:
                continue
            if name == "th" or el.find(["b","strong"]):
                txt = raw
            else:
                # not a heading-like cell; ignore
                continue

        # Standalone bold/strong blocks (not inside paragraphs/lis)
        elif name in ("b","strong"):
            if el.find_parent(["p","li"]):
                continue
            txt = normalize_ws(el.get_text(" ", strip=True))

        # Block-level anchor headings (e.g., <div><a>Koordinatensysteme</a></div>)
        elif name == "a" and is_block_heading_anchor(el):
            txt = normalize_ws(el.get_text(" ", strip=True))

        # Headline-ish DIVs that contain only inline nodes (no nested blocks)
        elif name == "div" and only_inline_children(el):
            txt = normalize_ws(el.get_text(" ", strip=True))
            # keep short, meaningful lines; drop UI chrome
            if not txt or looks_like_breadcrumb(txt):
                continue
            # avoid capturing generic top-bar items
            if txt.lower() in {"logout","konto","einstellungen","placeholder","alle dateien"}:
                continue

        # Regular content
        elif name in ("p","li"):
            txt = normalize_ws(el.get_text(" ", strip=True))

        # Preserve formatted blocks
        elif name == "pre":
            txt = el.get_text()  # keep line breaks/spaces

        else:
            continue

        if txt and (not lines or lines[-1] != txt):
            lines.append(txt)

    # assemble output
    out = []
    if title:
        out.append(f"title:{title}")
    if h1_text:
        out.append(h1_text)
    if lines:
        out.append("")
        out.extend(lines)

    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Filter an HTML/HTM file to plain text.")
    ap.add_argument("input")
    ap.add_argument("-o","--output")
    args = ap.parse_args()

    html = Path(args.input).read_text(encoding="utf-8", errors="ignore")
    result = filter_html(html)

    if args.output:
        Path(args.output).write_text(result + "\n", encoding="utf-8")
    else:
        sys.stdout.write(result + "\n")

if __name__ == "__main__":
    main()




# chunk_jsonl.py
import re, json, argparse, unicodedata
from pathlib import Path

# ---------- token utils ----------
TOK = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def count_tokens(s: str) -> int:
    return len(TOK.findall(s or ""))

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:120] or "section"

# ---------- parse your filtered text ----------
def parse_filtered_text(txt: str):
    """
    Your filter usually emits:
      title: ...
      <h1 line>          (optional)
      
      <content line 1>
      <content line 2>
      ...
    We derive sections by a light heading heuristic.
    """
    lines = [ln.rstrip() for ln in txt.splitlines()]
    title, h1 = "", ""
    i = 0
    if i < len(lines) and lines[i].startswith("title:"):
        title = lines[i].split(":", 1)[1].strip()
        i += 1
    # first non-empty after title as h1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip():
        h1 = lines[i].strip()
        i += 1
    # skip blank gap
    while i < len(lines) and not lines[i].strip():
        i += 1

    content = lines[i:]
    return title, h1, content

# ---------- heading heuristic ----------
END_PUNCT = (".", "!", "?", ";")
def is_heading_line(ln: str) -> bool:
    if not ln or ln.startswith("<pre>") or ln.startswith("title:"):
        return False
    s = ln.strip()
    if not s:
        return False
    # very short / title-like
    if len(s) <= 80 and not s.endswith(END_PUNCT):
        # few tokens => likely a heading (e.g., "Koordinatensysteme")
        if count_tokens(s) <= 12:
            return True
    # UPPER or Title Case bias
    if s == s.upper() and 3 <= len(s) <= 80:
        return True
    # Bullet style like "TIPP" or "Hinweis" alone
    if s.lower() in {"tipp", "hinweis"}:
        return True
    return False

# ---------- sectionizer ----------
def split_sections(lines):
    """
    Build sections using heading-like lines.
    Also treat <pre>...</pre> as indivisible blocks attached to the current section.
    Returns: list of {heading, text, level}
    """
    sections = []
    cur = {"heading": None, "text": []}

    it = iter(range(len(lines)))
    idx = 0
    while idx < len(lines):
        ln = lines[idx]

        # capture <pre> blocks as single lines
        if ln.lstrip().lower().startswith("<pre>"):
            block = [ln]
            j = idx + 1
            while j < len(lines) and "</pre>" not in lines[j].lower():
                block.append(lines[j])
                j += 1
            if j < len(lines):
                block.append(lines[j])
                idx = j
            cur["text"].append("\n".join(block))
        elif is_heading_line(ln):
            # flush previous section
            if cur["heading"] or any(x.strip() for x in cur["text"]):
                sections.append({"heading": cur["heading"], "text": "\n".join(cur["text"]).strip(), "level": 2})
            cur = {"heading": ln.strip(), "text": [], "level": 2}
        else:
            if ln.strip():
                cur["text"].append(ln)
        idx += 1

    if cur["heading"] or any(x.strip() for x in cur["text"]):
        sections.append({"heading": cur["heading"], "text": "\n".join(cur["text"]).strip(), "level": 2})
    return sections

# ---------- chunker ----------
def chunk_with_overlap(text: str, target_tokens=700, overlap_tokens=80):
    if not text:
        return []

    # protect <pre> blocks
    parts = re.split(r"(\n?<pre>[\s\S]*?</pre>\s*)", text, flags=re.IGNORECASE)
    blocks = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            blocks.append(("pre", part))
        else:
            # split into paragraphs
            paras = [p.strip() for p in part.split("\n\n") if p.strip()]
            blocks.extend(("para", p) for p in paras)

    chunks, buf, buf_tokens = [], [], 0
    for kind, blob in blocks:
        t = count_tokens(blob)
        # if a paragraph is huge, sentence-split (avoid breaking <pre>)
        if kind == "para" and t > target_tokens:
            sentences = re.split(r"(?<=[.!?:])\s+", blob)
            for s in sentences:
                st = count_tokens(s)
                if buf_tokens + st > target_tokens and buf:
                    chunks.append("\n\n".join(buf).strip())
                    # soft overlap: tail of previous buffer
                    tail = " ".join(buf)[-overlap_tokens*6:]
                    buf, buf_tokens = [tail], count_tokens(tail)
                buf.append(s)
                buf_tokens += st
            continue

        if buf_tokens + t > target_tokens and buf:
            chunks.append("\n\n".join(buf).strip())
            tail = " ".join(buf)[-overlap_tokens*6:]
            buf, buf_tokens = [tail], count_tokens(tail)

        buf.append(blob)
        buf_tokens += t

    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks

def build_heading_path(sections, i):
    # simple roll-up using order (all same level=2 here)
    path = [sec["heading"] for sec in sections[: i + 1] if sec["heading"]]
    return " > ".join(path[-3:])  # keep last 3 for brevity

# ---------- pack to JSONL ----------
def to_jsonl(filtered_text: str, source_path: str, out_path: str, lang="de",
             target_tokens=700, overlap_tokens=80):
    title, h1, content_lines = parse_filtered_text(filtered_text)
    sections = split_sections(content_lines)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sec in enumerate(sections):
            heading = sec["heading"] or h1 or title
            hp = build_heading_path(sections, i) if sec["heading"] else (h1 or heading)
            anchor = slug(heading)
            for k, chunk in enumerate(chunk_with_overlap(sec["text"],
                                                         target_tokens=target_tokens,
                                                         overlap_tokens=overlap_tokens)):
                rec = {
                    "source_path": source_path,
                    "title": title,
                    "h1": h1,
                    "heading": heading,
                    "heading_path": hp,
                    "anchor": anchor,
                    "chunk_index": k,
                    "lang": lang,
                    "text": f"## {heading}\n\n{chunk}",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chunk filtered text into JSONL for RAG.")
    ap.add_argument("input_txt", help="Output of filter_html.py (plain text)")
    ap.add_argument("-o", "--out", default="rag.jsonl", help="Output JSONL path")
    ap.add_argument("--source", default="", help="Original file path/URL for metadata")
    ap.add_argument("--lang", default="de", help="Language metadata")
    ap.add_argument("--target-tokens", type=int, default=700)
    ap.add_argument("--overlap-tokens", type=int, default=80)
    args = ap.parse_args()

    txt = Path(args.input_txt).read_text(encoding="utf-8", errors="ignore")
    to_jsonl(
        filtered_text=txt,
        source_path=args.source or args.input_txt,
        out_path=args.out,
        lang=args.lang,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )



# 1) Produce clean text (your existing step)
python filter_html.py Help-content-schnelleinstieg_details.htm -o page.txt

# 2) Chunk into JSONL
python chunk_jsonl.py page.txt -o page.jsonl --source Help-content-schnelleinstieg_details.htm --target-tokens 700 --overlap-tokens 80
