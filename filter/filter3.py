# filter_html_md.py
from bs4 import BeautifulSoup, NavigableString
import argparse, sys, re
from pathlib import Path

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

INLINE_TAGS = {"a","b","strong","i","em","span","code","kbd","samp","u","small","sup","sub","mark"}

def only_inline_children(tag) -> bool:
    for c in tag.children:
        if isinstance(c, NavigableString):
            continue
        if getattr(c, "name", "").lower() not in INLINE_TAGS:
            return False
    return True

def is_block_heading_anchor(a_tag) -> bool:
    if not a_tag or not a_tag.get_text(strip=True):
        return False
    cont = a_tag
    if cont.parent and getattr(cont.parent, "name", "").lower() in INLINE_TAGS:
        cont = cont.parent
    parent = cont.parent
    if not parent or getattr(parent, "name", "").lower() not in {"div","td","th"}:
        return False
    parent_txt = normalize_ws(parent.get_text(" ", strip=True))
    link_txt = normalize_ws(a_tag.get_text(" ", strip=True))
    return parent_txt == link_txt and only_inline_children(parent)

def filter_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(strip=True) if soup.title else ""
    h1 = soup.find("h1")
    h1_text = normalize_ws(h1.get_text(" ", strip=True)) if h1 else ""

    iterable = (h1.find_all_next(True) if h1 else (soup.body or soup).find_all(True))
    lines = []

    for el in iterable:
        name = (el.name or "").lower()
        txt, heading_level = None, None

        if name in ("h1","h2","h3","h4","h5","h6"):
            txt = normalize_ws(el.get_text(" ", strip=True))
            heading_level = int(name[1])

        elif name in ("th","td"):
            raw = normalize_ws(el.get_text(" ", strip=True))
            if raw and (name == "th" or el.find(["b","strong"])):
                txt = raw
                heading_level = 3

        elif name in ("b","strong"):
            if el.find_parent(["p","li"]):
                continue
            txt = normalize_ws(el.get_text(" ", strip=True))
            heading_level = 3

        elif name == "a" and is_block_heading_anchor(el):
            txt = normalize_ws(el.get_text(" ", strip=True))
            heading_level = 2

        elif name == "div" and only_inline_children(el):
            txt = normalize_ws(el.get_text(" ", strip=True))
            if txt:
                heading_level = 2

        elif name in ("p","li"):
            txt = normalize_ws(el.get_text(" ", strip=True))

        elif name == "pre":
            txt = el.get_text()

        if txt:
            if heading_level:
                prefix = "#" * max(2, min(6, heading_level))
                lines.append(f"{prefix} {txt}")
            else:
                lines.append(txt)

    out = []
    if title:
        out.append(f"title:{title}")
    if h1_text:
        out.append(f"# {h1_text}")
    if lines:
        out.append("")
        out.extend(lines)

    return "\n".join(out)

def main():
    ap = argparse.ArgumentParser(description="Filter HTML file to Markdown with heading levels.")
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



# rag_pack.py
import json, re, argparse, unicodedata
from pathlib import Path

_tok = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def count_tokens(s: str) -> int:
    return len(_tok.findall(s or ""))

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:120] or "section"

def parse_filtered_markdown(txt: str):
    title, h1 = "", ""
    lines, cleaned = txt.splitlines(), []
    for ln in lines:
        if ln.startswith("title:") and not title:
            title = ln.split(":", 1)[1].strip()
        elif ln.startswith("# ") and not h1:
            h1 = ln[2:].strip()
            cleaned.append(ln)
        else:
            cleaned.append(ln)
    return title, h1, "\n".join(cleaned)

def split_sections(md_text: str):
    lines = md_text.splitlines()
    sections, cur = [], {"heading": None, "level": 0, "text": []}
    for ln in lines:
        m = re.match(r"^(#{2,6})\s+(.*)$", ln)
        if m:
            if cur["heading"] or any(t.strip() for t in cur["text"]):
                cur["text"] = "\n".join(cur["text"]).strip()
                sections.append(cur)
            cur = {"heading": m.group(2).strip(), "level": len(m.group(1)), "text": []}
        else:
            cur["text"].append(ln)
    if cur["heading"] or any(t.strip() for t in cur["text"]):
        cur["text"] = "\n".join(cur["text"]).strip()
        sections.append(cur)
    return sections

def chunk_text_with_overlap(text: str, target_tokens=700, overlap_tokens=80):
    if not text:
        return []
    parts = re.split(r"(\n?<pre>[\s\S]*?</pre>\n?)", text, flags=re.IGNORECASE)
    blocks = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            blocks.append(("pre", part))
        else:
            for para in [p for p in part.split("\n\n") if p.strip()]:
                blocks.append(("para", para.strip()))

    chunks, buf, buf_tokens = [], [], 0
    for kind, blob in blocks:
        t = count_tokens(blob)
        if t > target_tokens and kind != "pre":
            sentences = re.split(r"(?<=[.!?:])\s+", blob)
            for s in sentences:
                st = count_tokens(s)
                if buf_tokens + st > target_tokens and buf:
                    chunks.append("\n\n".join(buf).strip())
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

def build_heading_path(sections, idx):
    cur = sections[idx]
    path = [cur["heading"]] if cur["heading"] else []
    my_level, j = cur["level"], idx - 1
    while j >= 0 and my_level > 0:
        if sections[j]["level"] < my_level and sections[j]["heading"]:
            path.insert(0, sections[j]["heading"])
            my_level = sections[j]["level"]
        j -= 1
    return " > ".join(path)

def pack_to_jsonl(filtered_text: str, source_path: str, out_path: str):
    title, h1, md_text = parse_filtered_markdown(filtered_text)
    sections = split_sections(md_text)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sec in enumerate(sections):
            text = sec["text"].strip()
            if not (sec["heading"] or text):
                continue
            heading = sec["heading"] or h1 or title
            hp = build_heading_path(sections, i) or h1 or title
            anchor = slug(heading)
            for k, chunk in enumerate(chunk_text_with_overlap(text)):
                chunk_text = f"### {hp}\n\n{chunk}"
                rec = {
                    "source_path": source_path,
                    "title": title,
                    "h1": h1,
                    "heading": heading,
                    "heading_path": hp,
                    "anchor": anchor,
                    "chunk_index": k,
                    "text": chunk_text,
                    "lang": "de",
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input_md", help="Markdown from filter_html_md.py")
    ap.add_argument("-o", "--out", default="rag.jsonl")
    args = ap.parse_args()

    txt = Path(args.input_md).read_text(encoding="utf-8", errors="ignore")
    pack_to_jsonl(txt, args.input_md, args.out)


python filter_html_md.py Help-content-schnelleinstieg_details.htm -o page.md


python rag_pack.py page.md -o page.jsonl
