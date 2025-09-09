#!/usr/bin/env python3
# path: src/embeddinggemma_rag_starter.py
"""Minimal, fully-local multilingual RAG starter using EmbeddingGemma + FAISS.

Features
- PDF/image ingest → preprocess (deskew/denoise) → OCR (PaddleOCR) → elements JSON
- Multilingual embeddings via google/embeddinggemma-300m (SentenceTransformers)
- Matryoshka truncation (768/512/256/128) to trade accuracy/size
- FAISS inner-product search with L2-normalized vectors
- CLI: build-index, search

Why these choices
- Keeps the emb/index module stable while you iterate on layout/agents downstream.
- Uses auto-prompts (query/document) from SentenceTransformers integration for best retrieval.

Requirements (install once)
  pip install "git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview" \
              sentence-transformers>=5.0.0 faiss-cpu torch paddleocr opencv-python pymupdf

Usage
  python embeddinggemma_rag_starter.py build-index input.pdf --out ./out --dim 256
  python embeddinggemma_rag_starter.py search ./out --q "भारत में वेतन पर्ची में HRA क्या है?" -k 5

Notes
- First run downloads weights to HF cache, subsequent runs are offline.
- PaddleOCR covers English, Hindi (Devanagari), Urdu/Persian/Arabic, Nepali; we use 'ml' lang.
- BBoxes are HBB [x, y, w, h]; reading order is top-left to bottom-right (stable for OCR lines).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import faiss  # type: ignore
import fitz  # PyMuPDF  # type: ignore
import numpy as np
from paddleocr import PaddleOCR  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore


# --------------------------
# Data structures
# --------------------------
@dataclass
class Element:
    """Document element with minimal fields used for RAG + JSON export."""
    page: int
    cls: str  # e.g., "Text"
    bbox: Tuple[int, int, int, int]  # x, y, w, h (HBB)
    content: str
    language: Optional[str] = None


@dataclass
class Match:
    score: float
    element: Element


# --------------------------
# Image preprocessing
# --------------------------

def _deskew_and_denoise(img: np.ndarray) -> np.ndarray:
    """Fast deskew via Hough lines + mild denoise.
    Why: Small rotation hurts OCR; a coarse fix improves recall without heavy compute.
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=150, minLineLength=150, maxLineGap=20)
    angle_deg = 0.0
    if lines is not None and len(lines) > 0:
        angles = []
        for x1, y1, x2, y2 in lines[:, 0, :]:
            ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            if -45 < ang < 45:
                angles.append(ang)
        if angles:
            angle_deg = float(np.median(angles))
    if abs(angle_deg) > 0.2:
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


# --------------------------
# PDF rasterization
# --------------------------

def _pdf_to_images(pdf_path: str, dpi: int = 220) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        images.append(img[:, :, ::-1])  # RGB→BGR for OpenCV consistency
    return images


# --------------------------
# OCR
# --------------------------
class OCRor:
    def __init__(self) -> None:
        # use_angle_cls improves rotated text detection; 'ml' = multilingual
        self._ocr = PaddleOCR(use_angle_cls=True, lang="ml", show_log=False)

    def run(self, img: np.ndarray, page_idx: int) -> List[Element]:
        proc = _deskew_and_denoise(img)
        result = self._ocr.ocr(proc, cls=True)
        elements: List[Element] = []
        if not result:
            return elements
        # PaddleOCR returns [ [ [ [x,y], ... 4 pts ], (text, conf) ], ... ]
        for line in result[0]:
            pts = np.array(line[0], dtype=np.float32)
            x, y, w, h = _quad_to_hbb(pts)
            text = line[1][0]
            if text and text.strip():
                elements.append(Element(page=page_idx, cls="Text", bbox=(x, y, w, h), content=text.strip()))
        # sort: reading order top-left to bottom-right
        elements.sort(key=lambda e: (e.page, e.bbox[1], e.bbox[0]))
        return elements


def _quad_to_hbb(quad: np.ndarray) -> Tuple[int, int, int, int]:
    xs = quad[:, 0]
    ys = quad[:, 1]
    x, y = int(xs.min()), int(ys.min())
    w, h = int(xs.max() - xs.min()), int(ys.max() - ys.min())
    return x, y, w, h


# --------------------------
# Embeddings backend (EmbeddingGemma)
# --------------------------
class Embeddings:
    """Thin wrapper over SentenceTransformers for EmbeddingGemma.

    Uses auto task prompts for queries/documents and supports Matryoshka truncation.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim not in (128, 256, 512, 768):
            raise ValueError("dim must be one of {128, 256, 512, 768}")
        self.model = SentenceTransformer("google/embeddinggemma-300m", truncate_dim=dim)
        self.dim = dim

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode_document(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode_query(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype(np.float32)


# --------------------------
# Index (FAISS)
# --------------------------
class VectorIndex:
    def __init__(self, dim: int, workdir: Path) -> None:
        self.dim = dim
        self.workdir = workdir
        self.index = faiss.IndexFlatIP(dim)
        self.meta: List[Dict] = []

    @property
    def index_path(self) -> Path:
        return self.workdir / "index.faiss"

    @property
    def meta_path(self) -> Path:
        return self.workdir / "meta.json"

    def add(self, embeddings: np.ndarray, metas: List[Dict]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        # vectors must be normalized for inner-product ≈ cosine
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.meta.extend(metas)

    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, workdir: Path) -> "VectorIndex":
        with (workdir / "meta.json").open("r", encoding="utf-8") as f:
            meta: List[Dict] = json.load(f)
        index = faiss.read_index(str(workdir / "index.faiss"))
        dim = index.d
        obj = cls(dim=dim, workdir=workdir)
        obj.index = index
        obj.meta = meta
        return obj

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        faiss.normalize_L2(query_vec)
        scores, idxs = self.index.search(query_vec.astype(np.float32), k)
        pairs = [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]
        return pairs


# --------------------------
# Chunking helpers
# --------------------------

def chunk_text(text: str, max_words: int = 220, overlap: int = 30) -> List[str]:
    """Word-window chunking to stay well within 2k token context.
    Why: Keeps multilingual scripts intact without tokenizer dependency.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: List[str] = []
    i = 0
    while i < len(words):
        chunk = words[i : i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
    return chunks


# --------------------------
# End-to-end build index
# --------------------------

def build_index(in_path: Path, out_dir: Path, dim: int = 256) -> Path:
    emb = Embeddings(dim=dim)
    index = VectorIndex(dim=dim, workdir=out_dir)
    ocr = OCRor()

    all_elements: List[Element] = []

    images: List[np.ndarray] = []
    if in_path.suffix.lower() in {".pdf"}:
        images = _pdf_to_images(str(in_path))
    else:
        img = cv2.imread(str(in_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {in_path}")
        images = [img]

    # OCR each page
    for pno, img in enumerate(images):
        elements = ocr.run(img, page_idx=pno)
        all_elements.extend(elements)

    # Export elements JSON for this document
    doc_json = {
        "document_id": in_path.name,
        "elements": [
            {
                "class": e.cls,
                "bbox": list(e.bbox),
                "content": e.content,
                "language": e.language or "unknown",
                "page": e.page,
            }
            for e in all_elements
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{in_path.stem}.elements.json").write_text(
        json.dumps(doc_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Chunk, embed, and index
    corpus: List[str] = []
    metas: List[Dict] = []
    for idx_e, e in enumerate(all_elements):
        for frag in chunk_text(e.content):
            corpus.append(frag)
            metas.append(
                {
                    "doc": in_path.name,
                    "page": e.page,
                    "bbox": e.bbox,
                    "class": e.cls,
                    "content": frag,
                    "source_element_idx": idx_e,
                }
            )

    if not corpus:
        print("[warn] No OCR text found; index is empty.")
    else:
        embs = emb.encode_documents(corpus)
        index.add(embs, metas)

    index.save()
    return out_dir


# --------------------------
# Search
# --------------------------

def search(out_dir: Path, query: str, k: int = 5) -> List[Match]:
    idx = VectorIndex.load(out_dir)
    emb = Embeddings(dim=idx.dim)
    qvec = emb.encode_query(query)
    hits = idx.search(qvec, k=k)

    matches: List[Match] = []
    for i, s in hits:
        meta = idx.meta[i]
        element = Element(
            page=int(meta["page"]),
            cls=str(meta["class"]),
            bbox=tuple(meta["bbox"]),
            content=str(meta["content"]),
        )
        matches.append(Match(score=s, element=element))
    return matches


# --------------------------
# CLI
# --------------------------

def _cmd_build_index(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_dir = Path(args.out)
    dim = int(args.dim)
    build_index(in_path, out_dir, dim=dim)
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), "dim": dim}, indent=2))


def _cmd_search(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    res = search(out_dir, query=args.q, k=int(args.k))
    payload = [
        {
            "score": round(m.score, 4),
            "page": m.element.page,
            "bbox": list(m.element.bbox),
            "class": m.element.cls,
            "content": m.element.content,
        }
        for m in res
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EmbeddingGemma + FAISS multilingual RAG starter")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-index", help="OCR + embed + index a PDF/image")
    p_build.add_argument("input", type=str, help="Path to PDF or image")
    p_build.add_argument("--out", type=str, required=True, help="Output directory for index & JSON")
    p_build.add_argument("--dim", type=int, default=256, choices=[128, 256, 512, 768], help="Embedding dimension")
    p_build.set_defaults(func=_cmd_build_index)

    p_search = sub.add_parser("search", help="Search an existing index directory")
    p_search.add_argument("out", type=str, help="Index directory")
    p_search.add_argument("--q", type=str, required=True, help="Query text (any supported language)")
    p_search.add_argument("-k", type=int, default=5, help="Top-k results")
    p_search.set_defaults(func=_cmd_search)

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
