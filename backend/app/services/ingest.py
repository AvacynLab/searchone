import fitz  # PyMuPDF
from typing import List, Dict, Any
from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP
import requests
from bs4 import BeautifulSoup
import logging
from app.core.logging_config import configure_logging
import urllib.parse
import json
from hashlib import md5
import os
from collections import Counter
from app.services.html_parser import extract_main_content, html_to_markdown
from app.services.pdf_parser import parse_pdf
from app.services.table_parser import infer_schema, table_to_records

configure_logging()
logger = logging.getLogger(__name__)

TRUSTED_DOMAINS = [d.strip().lower() for d in (os.getenv("SEARCHONE_TRUSTED_DOMAINS") or "").split(",") if d.strip()]
WHITELIST_DOMAINS = [d.strip().lower() for d in (os.getenv("SEARCHONE_WHITELIST_DOMAINS") or "").split(",") if d.strip()]
BLACKLIST_DOMAINS = [d.strip().lower() for d in (os.getenv("SEARCHONE_BLACKLIST_DOMAINS") or "").split(",") if d.strip()]
CHUNK_HASHES: Counter = Counter()
_HAS_READABILITY = False
_HAS_TRAFILATURA = False
try:
    from readability import Document as ReadabilityDocument  # type: ignore
    _HAS_READABILITY = True
except Exception:
    ReadabilityDocument = None  # type: ignore
    _HAS_READABILITY = False
try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:
    trafilatura = None  # type: ignore
    _HAS_TRAFILATURA = False


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return "\n".join(texts)


def extract_pdf_metadata(path: str) -> Dict[str, Any]:
    """Extract simple metadata from PDF (title, author, creation date)."""
    try:
        doc = fitz.open(path)
        info = doc.metadata or {}
        return {
            "title": info.get("title"),
            "author": info.get("author"),
            "created": info.get("creationDate"),
        }
    except Exception:
        return {}


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Legacy helper returning only chunk texts (kept for compatibility)."""
    return [c["text"] for c in chunk_text_with_positions(text, chunk_size=chunk_size, overlap=overlap)]


def chunk_text_with_positions(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """Chunk text with token offsets to preserve section ordering."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        end_idx = min(i + chunk_size, len(tokens))
        chunk_tokens = tokens[i:end_idx]
        chunks.append({"text": " ".join(chunk_tokens), "start": i, "end": end_idx})
        i += max(1, chunk_size - overlap)
    return chunks


def segment_text_sections(text: str) -> List[Dict[str, Any]]:
    """Naive section segmentation using headings/paragraph breaks."""
    sections = []
    lines = text.splitlines()
    buf = []
    title = "section_1"
    idx = 1
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped[:2].isdigit():
            if buf:
                sections.append({"title": title, "text": "\n".join(buf).strip()})
                buf = []
            title = stripped
            idx += 1
        else:
            buf.append(stripped)
    if buf:
        sections.append({"title": title, "text": "\n".join(buf).strip()})
    return sections
def _clean_boilerplate(text: str) -> str:
    """Basic boilerplate trimming: drop very short/long lines and repeated whitespace."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) < 40:
            # likely nav/footer noise
            continue
        if len(stripped) > 2000:
            stripped = stripped[:2000]
        lines.append(stripped)
    return "\n".join(lines)


def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML using trafilatura/readability if available, else BeautifulSoup."""
    text = ""
    if _HAS_TRAFILATURA:
        try:
            text = trafilatura.extract(html) or ""
        except Exception:
            text = ""
    if _HAS_READABILITY:
        try:
            doc = ReadabilityDocument(html)
            text = doc.summary(html_partial=False)
        except Exception:
            text = ""
    if not text:
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(['script', 'style', 'noscript']):
            s.decompose()
        text = soup.get_text(separator='\n')
    text = _clean_boilerplate(text)
    logger.debug("Extracted text from HTML (%d chars)", len(text))
    return text


def download_url(url: str, timeout: int = 15) -> str:
    logger.info("Downloading URL %s", url)
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "SearchOneBot/1.0"})
    resp.raise_for_status()
    logger.debug("Downloaded %s (%d bytes)", url, len(resp.content))
    return resp.text


def _domain_allowed(url: str) -> bool:
    try:
        domain = urllib.parse.urlparse(url).netloc.lower()
    except Exception:
        return False
    if BLACKLIST_DOMAINS and any(domain.endswith(b) for b in BLACKLIST_DOMAINS):
        return False
    if WHITELIST_DOMAINS:
        return any(domain.endswith(w) for w in WHITELIST_DOMAINS)
    return True


def compute_source_metadata(url: str, title: str = "") -> Dict[str, Any]:
    """Compute simple metadata + reliability for a URL source."""
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    tags = []
    reliability = 0.5
    if domain in TRUSTED_DOMAINS:
        reliability = 0.9
        tags.append("trusted_domain")
    if domain.endswith(".gov") or domain.endswith(".gouv.fr"):
        reliability = max(reliability, 0.9)
        tags.append("gov")
    elif domain.endswith(".edu"):
        reliability = max(reliability, 0.85)
        tags.append("edu")
    meta = {
        "domain": domain,
        "scheme": parsed.scheme,
        "path": parsed.path,
        "title": title,
        "tags": tags,
    }
    source_type = "url"
    return {"reliability": reliability, "source_metadata": json.dumps(meta, ensure_ascii=False), "source_type": source_type}


def ingest_web_page(url: str, title: str = "") -> Dict[str, Any]:
    """Download, clean, segment, chunk and persist a single web page as a document + chunks."""
    from db import get_session, Document, Chunk
    if not _domain_allowed(url):
        raise ValueError("Domain not allowed by whitelist/blacklist")
    html = download_url(url)
    parsed = extract_main_content(html)
    doc_title = title or parsed.get("title") or url
    sections = [sec for sec in parsed.get("sections", []) if sec.get("content")]
    fallback_text = ""
    if not sections:
        fallback_text = html_to_markdown(html)
        if not fallback_text:
            fallback_text = extract_text_from_html(html)
        if not fallback_text:
            raise ValueError("No text extracted from URL")
        sections = [{"title": doc_title or "body", "content": fallback_text}]
    all_chunks = []
    for sec in sections:
        sec_chunks = chunk_text_with_positions(sec["content"])
        for c in sec_chunks:
            c["section_title"] = sec.get("title")
        all_chunks.extend(sec_chunks)
    if not all_chunks:
        raise ValueError("No text extracted from URL")
    meta_info = compute_source_metadata(url, title=doc_title)
    src_meta = json.loads(meta_info.get("source_metadata") or "{}")
    doc_links = parsed.get("links") or []
    with get_session() as s:
        doc = Document(
            title=doc_title,
            source_path=url,
            reliability=meta_info.get("reliability"),
            source_metadata=meta_info.get("source_metadata"),
            source_type=meta_info.get("source_type"),
            published_at=None,
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        doc_id = doc.id
        for idx, c in enumerate(all_chunks):
            meta_chunk = {
                "source": url,
                "source_type": meta_info.get("source_type"),
                "domain": src_meta.get("domain"),
                "start": c["start"],
                "end": c["end"],
                "reliability": meta_info.get("reliability"),
                "section_title": c.get("section_title"),
                "doc_type": "html",
                "title": src_meta.get("title") or doc_title,
                "tags": src_meta.get("tags") or [],
                "links": doc_links,
                "published_at": None,
            }
            h = md5(c["text"].encode("utf-8")).hexdigest()
            CHUNK_HASHES[h] += 1
            if CHUNK_HASHES[h] > 1:
                continue
            ch = Chunk(document_id=doc.id, chunk_index=idx, text=c["text"], meta=json.dumps(meta_chunk))
            s.add(ch)
        s.commit()
    return {"document_id": doc_id, "chunks": len(all_chunks), "source": url, "title": doc_title}


def ingest_pdf_file(path: str, title: str = "", published_at: str = "") -> Dict[str, Any]:
    """Ingest a local PDF: extract text, segment, chunk, and persist."""
    from db import get_session, Document, Chunk, TabularData
    pdf_data = parse_pdf(path)
    if not pdf_data.get("full_text"):
        raise ValueError("No text extracted from PDF")
    sections = []
    for page in pdf_data.get("pages", []):
        page_text = page.get("text", "").strip()
        if not page_text:
            continue
        sections.append({"title": f"Page {page.get('number')}", "content": page_text, "page": page.get("number")})
    if not sections:
        raise ValueError("No textual sections extracted from PDF")
    all_chunks = []
    for sec in sections:
        sec_chunks = chunk_text_with_positions(sec["content"])
        for c in sec_chunks:
            c["section_title"] = sec.get("title")
            c["page_number"] = sec.get("page")
        all_chunks.extend(sec_chunks)
    meta_info = metadata_from_file(path, title=title, published_at=published_at)
    combined_metadata = json.loads(meta_info.get("source_metadata") or "{}")
    combined_metadata["pdf_metadata"] = pdf_data.get("metadata")
    combined_metadata["detected_tables"] = pdf_data.get("tables", [])
    combined_metadata["full_text_length"] = len(pdf_data.get("full_text") or "")
    with get_session() as s:
        doc = Document(
            title=title or meta_info.get("title") or path,
            source_path=path,
            reliability=meta_info.get("reliability"),
            source_metadata=json.dumps(combined_metadata, ensure_ascii=False),
            source_type=meta_info.get("source_type"),
            published_at=meta_info.get("published_at"),
        )
        s.add(doc)
        s.commit()
        s.refresh(doc)
        for idx, c in enumerate(all_chunks):
            meta_chunk = {
                "source": path,
                "source_type": meta_info.get("source_type"),
                "start": c["start"],
                "end": c["end"],
                "reliability": meta_info.get("reliability"),
                "section_title": c.get("section_title"),
                "doc_type": "pdf" if meta_info.get("source_type") == "pdf" else "file",
                "title": meta_info.get("title") or title or path,
                "author": meta_info.get("author"),
                "published_at": meta_info.get("published_at"),
                "page_number": c.get("page_number"),
            }
            h = md5(c["text"].encode("utf-8")).hexdigest()
            CHUNK_HASHES[h] += 1
            if CHUNK_HASHES[h] > 1:
                continue
            ch = Chunk(document_id=doc.id, chunk_index=idx, text=c["text"], meta=json.dumps(meta_chunk))
            s.add(ch)
        for table in pdf_data.get("tables", []):
            if not table.get("rows") or len(table["rows"]) < 2:
                continue
            schema = infer_schema(table)
            records = table_to_records(table, schema)
            if not records:
                continue
            tabular_entry = TabularData(
                document_id=doc.id,
                table_id=table.get("id") or f"{path}:{table.get('page') or '0'}",
                table_schema=json.dumps(schema, ensure_ascii=False),
                records=json.dumps(records, ensure_ascii=False),
                table_metadata=json.dumps({"page": table.get("page"), "source": path}, ensure_ascii=False),
            )
            s.add(tabular_entry)
        s.commit()
    return {"document_id": doc.id, "chunks": len(all_chunks), "source": path, "title": title or path}


def metadata_from_file(path: str, title: str = "", published_at: str = "") -> Dict[str, Any]:
    """Generate metadata for a local PDF/source with heuristic reliability."""
    name = os.path.basename(path)
    reliability = 0.6
    is_pdf = name.lower().endswith(".pdf")
    if is_pdf:
        reliability = 0.7
    meta_extra = extract_pdf_metadata(path) if is_pdf else {}
    meta = {
        "filename": name,
        "title": title or meta_extra.get("title"),
        "author": meta_extra.get("author"),
        "created": meta_extra.get("created"),
        "hash": md5(name.encode("utf-8")).hexdigest(),
    }
    return {
        "reliability": reliability,
        "source_metadata": json.dumps(meta, ensure_ascii=False),
        "source_type": "pdf" if is_pdf else "file",
        "published_at": published_at or meta_extra.get("created"),
        "title": meta.get("title"),
        "author": meta.get("author"),
    }
