#!/usr/bin/env python3
"""Generate embeddings for Bible chapters and verses using MongoDB mdbr-leaf-ir.

Outputs:
  - chapters.json: metadata (id, book, chapter, title, verse_count, window_count)
  - chapters_embeddings.bin: all chapter window embeddings as contiguous float32
  - verses/<id>.json: verse metadata (verse number, text)
  - verses/<id>.bin: verse embeddings as contiguous float32
"""

import json
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BSB_PATH = PROJECT_DIR / "bsb.json"
DATA_DIR = PROJECT_DIR / "public" / "data"
CHAPTERS_META = DATA_DIR / "chapters.json"
CHAPTERS_BIN = DATA_DIR / "chapters_embeddings.bin"
VERSES_DIR = DATA_DIR / "verses"

EMBED_DIM = 768  # full output after Dense layer
MAX_TOKENS = 512
WINDOW_TARGET_TOKENS = 480
WINDOW_OVERLAP_VERSES = 4


def load_bible():
    """Load bsb.json and group verses by book+chapter."""
    with open(BSB_PATH, "r") as f:
        data = json.load(f)

    chapters = {}
    for entry in data["_default"].values():
        # Skip verses with no text (omitted in some manuscripts)
        if not isinstance(entry.get("text"), str) or not entry["text"].strip():
            continue
        key = f"{entry['book']}_{entry['chapter']}"
        if key not in chapters:
            chapters[key] = {
                "book": entry["book"],
                "chapter": entry["chapter"],
                "verses": [],
            }
        chapters[key]["verses"].append(
            {"verse": entry["verse"], "text": entry["text"]}
        )

    # Sort verses numerically within each chapter
    for ch in chapters.values():
        ch["verses"].sort(key=lambda v: int(v["verse"]))

    return chapters


def count_tokens(tokenizer, text):
    """Count tokens using the model's tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def make_windows(tokenizer, verses, target_tokens=WINDOW_TARGET_TOKENS, overlap=WINDOW_OVERLAP_VERSES):
    """Create sliding windows of verses that fit within target token count."""
    windows = []
    i = 0
    while i < len(verses):
        window_verses = []
        token_count = 0
        j = i
        while j < len(verses):
            verse_text = f"{verses[j]['verse']}. {verses[j]['text']}"
            verse_tokens = count_tokens(tokenizer, verse_text)
            if token_count + verse_tokens > target_tokens and window_verses:
                break
            window_verses.append(verse_text)
            token_count += verse_tokens
            j += 1
        windows.append(" ".join(window_verses))
        # Advance by window size minus overlap
        advance = max(1, len(window_verses) - overlap)
        i += advance
        if j >= len(verses):
            break
    return windows


def truncate_and_normalize(embeddings):
    """L2-normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def main():
    print("Loading Bible data...")
    chapters = load_bible()
    print(f"Found {len(chapters)} chapters")

    print("Loading mdbr-leaf-ir model...")
    model = SentenceTransformer("MongoDB/mdbr-leaf-ir", device="cpu")
    tokenizer = model.tokenizer
    print(f"  Model modules: {[(i, type(m).__name__) for i, m in enumerate(model)]}")

    os.makedirs(VERSES_DIR, exist_ok=True)

    chapters_meta = []
    all_verse_texts = []

    # Collect all verse texts for batch encoding
    sorted_keys = sorted(chapters.keys(), key=lambda k: (chapters[k]["book"], int(chapters[k]["chapter"])))

    print("Collecting verse texts...")
    for key in sorted_keys:
        ch = chapters[key]
        for v in ch["verses"]:
            all_verse_texts.append(v["text"])

    print(f"Encoding {len(all_verse_texts)} verses...")
    verse_embeddings_raw = model.encode(all_verse_texts, show_progress_bar=True, batch_size=256)
    verse_embeddings_all = truncate_and_normalize(verse_embeddings_raw)

    # Build verse embedding lookup
    verse_embed_idx = 0
    chapter_verse_embeddings = {}
    for key in sorted_keys:
        ch = chapters[key]
        n = len(ch["verses"])
        chapter_verse_embeddings[key] = verse_embeddings_all[verse_embed_idx:verse_embed_idx + n]
        verse_embed_idx += n

    # Process chapters for windowed embeddings
    print("Computing chapter window embeddings...")
    all_chapter_embeddings = []  # will concatenate into single binary

    for idx, key in enumerate(sorted_keys):
        ch = chapters[key]
        full_text = " ".join(f"{v['verse']}. {v['text']}" for v in ch["verses"])
        token_count = count_tokens(tokenizer, full_text)

        if token_count <= MAX_TOKENS:
            windows = [full_text]
        else:
            windows = make_windows(tokenizer, ch["verses"])

        window_embeddings_raw = model.encode(windows, batch_size=64)
        window_embeddings = truncate_and_normalize(window_embeddings_raw).astype(np.float32)

        all_chapter_embeddings.append(window_embeddings)

        chapters_meta.append({
            "id": key,
            "book": ch["book"],
            "chapter": ch["chapter"],
            "title": f"{ch['book']} {ch['chapter']}",
            "verse_count": len(ch["verses"]),
            "window_count": len(windows),
        })

        # Write per-chapter verse files: JSON for text, .bin for embeddings
        v_embeds = chapter_verse_embeddings[key].astype(np.float32)
        verse_meta = []
        for vi, v in enumerate(ch["verses"]):
            verse_meta.append({
                "verse": v["verse"],
                "text": v["text"],
            })

        with open(VERSES_DIR / f"{key}.json", "w") as f:
            json.dump({"verses": verse_meta}, f, separators=(",", ":"))

        v_embeds.tofile(VERSES_DIR / f"{key}.bin")

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(sorted_keys)} chapters")

    # Write chapter metadata JSON (no embeddings — those go in .bin)
    print("Writing chapters.json...")
    with open(CHAPTERS_META, "w") as f:
        json.dump({"chapters": chapters_meta}, f, separators=(",", ":"))

    # Write all chapter embeddings as single contiguous float32 .bin
    print("Writing chapters_embeddings.bin...")
    all_embs = np.concatenate(all_chapter_embeddings, axis=0).astype(np.float32)
    all_embs.tofile(CHAPTERS_BIN)

    # Stats
    meta_size = os.path.getsize(CHAPTERS_META) / 1024
    bin_size = os.path.getsize(CHAPTERS_BIN) / (1024 * 1024)
    total_verse_json = sum(
        os.path.getsize(VERSES_DIR / f)
        for f in os.listdir(VERSES_DIR) if f.endswith(".json")
    ) / (1024 * 1024)
    total_verse_bin = sum(
        os.path.getsize(VERSES_DIR / f)
        for f in os.listdir(VERSES_DIR) if f.endswith(".bin")
    ) / (1024 * 1024)
    total_windows = sum(c["window_count"] for c in chapters_meta)
    print(f"\nDone!")
    print(f"  chapters.json:           {meta_size:.1f} KB")
    print(f"  chapters_embeddings.bin: {bin_size:.1f} MB ({total_windows} windows x {EMBED_DIM} dims)")
    print(f"  verse .json files:       {total_verse_json:.1f} MB")
    print(f"  verse .bin files:        {total_verse_bin:.1f} MB")
    print(f"  Total chapters: {len(chapters_meta)}, Total windows: {total_windows}")


if __name__ == "__main__":
    main()
