#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A. Extract + normalize the book text into sentences with char offsets.

Handles your note 4: the text/ folder may contain an EPUB, a PDF, or nothing.
  * .epub -> ebooklib + BeautifulSoup (spine order)
  * .pdf  -> PyMuPDF (fitz), falling back to pdfplumber
  * missing / unsupported -> raises MissingTextError so the book is skipped
"""

import glob
import os

from config import FRONTBACK_KEYWORDS
from .dtypes import Sentence
from .normalize import pre_normalize, normalize_fa, sent_split, strip_punct


class MissingTextError(Exception):
    """No usable book text (epub/pdf) found for this title."""


def find_text_file(text_dir):
    """Return (path, kind) where kind in {'epub','pdf'}, preferring epub.
    Raises MissingTextError if neither exists."""
    if not os.path.isdir(text_dir):
        raise MissingTextError(f"no text dir: {text_dir}")
    epubs = sorted(glob.glob(os.path.join(text_dir, "*.epub")))
    if epubs:
        return epubs[0], "epub"
    pdfs = sorted(glob.glob(os.path.join(text_dir, "*.pdf")))
    if pdfs:
        return pdfs[0], "pdf"
    have = ", ".join(sorted(os.listdir(text_dir))[:10]) or "empty"
    raise MissingTextError(f"no .epub/.pdf in {text_dir} (contents: {have})")


def _is_frontback(norm_text):
    head = norm_text[:400]
    hits = sum(1 for kw in FRONTBACK_KEYWORDS if kw in head)
    return hits >= 1 and len(norm_text) < 1500


def _iter_epub_blocks_raw(path):
    """Fallback: read epub as a plain zip and extract all HTML/XHTML items."""
    import zipfile
    from bs4 import BeautifulSoup
    with zipfile.ZipFile(path, "r") as zf:
        html_names = sorted(
            n for n in zf.namelist()
            if n.lower().endswith((".html", ".xhtml", ".htm"))
        )
        for name in html_names:
            try:
                content = zf.read(name)
            except KeyError:
                continue
            soup = BeautifulSoup(content, "lxml-xml")
            for tag in soup(["script", "style", "sup", "sub"]):
                tag.decompose()
            yield soup.get_text(separator=" ")


def _iter_epub_blocks(path):
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup
    try:
        book = epub.read_epub(path, options={"ignore_ncx": True})
    except Exception:
        # epub manifest is broken (missing zip entries, etc.) — parse raw
        yield from _iter_epub_blocks_raw(path)
        return
    for item in book.get_items_of_type(ITEM_DOCUMENT):       # spine/reading order
        try:
            content = item.get_content()
        except Exception:
            continue
        soup = BeautifulSoup(content, "lxml-xml")
        for tag in soup(["script", "style", "sup", "sub"]):
            tag.decompose()                                  # kill footnote markers
        yield soup.get_text(separator=" ")


def _iter_pdf_blocks(path):
    """Yield one text block per page. PyMuPDF first; pdfplumber fallback."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        for page in doc:
            yield page.get_text("text")
        return
    except Exception:
        pass
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            yield page.extract_text() or ""


def extract_book(text_dir, drop_frontback=True, keep_punct=True):
    """Extract + normalize the book into (sentences, book_text).

    sentences : list[Sentence] with continuous char offsets. Each carries the
                punctuated ground truth (.text) and its punctuation-free
                matching form (.norm).
    book_text : the sentences' MATCHING forms joined by single spaces. Offsets
                index into this string, and Whisper transcripts are normalized
                the same punctuation-free way, so fuzzy mapping is unaffected by
                the punctuation we keep for export.
    """
    path, kind = find_text_file(text_dir)
    blocks = _iter_epub_blocks(path) if kind == "epub" else _iter_pdf_blocks(path)

    sentences, book_chars, cursor = [], [], 0
    for raw in blocks:
        pre = pre_normalize(raw)                             # keeps punctuation
        if not pre:
            continue
        if drop_frontback and _is_frontback(pre):
            continue
        for s in sent_split(pre):
            disp = normalize_fa(s, keep_punct=keep_punct)    # per-sentence
            norm = strip_punct(disp) if keep_punct else disp
            if len(norm) < 2:
                continue
            sentences.append(Sentence(text=disp, offset=cursor, norm=norm))
            book_chars.append(norm)
            cursor += len(norm) + 1                          # +1 for join space

    if not sentences:
        raise MissingTextError(f"{os.path.basename(path)} produced 0 sentences")
    return sentences, " ".join(book_chars), (path, kind)
