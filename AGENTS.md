# AGENTS.md — Bible RAG Technical Reference

## Project Structure

```
bible-rag/
├── bsb.json                          # Source: Berean Standard Bible (31,102 verses, 16 with nan text skipped)
├── scripts/
│   └── generate_embeddings.py        # Offline embedding generation
├── public/data/
│   ├── chapters.json                 # Chapter metadata
│   ├── chapters_embeddings.bin       # Chapter window embeddings (legacy, not used for search)
│   ├── verses_embeddings.bin         # All verse embeddings (float16, ~45 MB)
│   ├── verses_index.json             # Verse index metadata
│   ├── dense_layer.bin               # Dense projection weights 384→768 (~1.2 MB)
│   └── verses/                       # Per-chapter verse text files (1,189 .json files)
│       └── Genesis_1.json            # Verse text metadata
├── src/
│   ├── App.tsx                       # Main app with search + detail layout
│   ├── App.css                       # All styles
│   ├── types.ts                      # TypeScript interfaces
│   ├── components/
│   │   ├── SearchBar.tsx             # Search input + loading progress
│   │   ├── ChapterResults.tsx        # Ranked chapter list
│   │   ├── ChapterDetail.tsx         # Verse heatmap display (auto-scrolls to best window)
│   │   └── TechDetails.tsx           # Collapsible technical info
│   ├── hooks/
│   │   ├── useEmbeddingModel.ts      # Load mdbr-leaf-ir ONNX + dense layer, encode queries
│   │   ├── useChapterSearch.ts       # Verse-level cosine sim, chapter ranking
│   │   └── useVerseHeatmap.ts        # Load + score verse embeddings for heatmap
│   └── utils/
│       └── similarity.ts             # Cosine similarity function
├── src/__tests__/
│   └── search.test.ts                # Benchmark test suite with 10 pre-computed query embeddings
```

## Embedding Pipeline

### Symmetric Architecture
- **Model**: `MongoDB/mdbr-leaf-ir` (22M params) for both documents and queries
- **Offline** (Python): sentence-transformers generates 768-dim embeddings, stored as float16
- **In-browser**: ONNX model via `@huggingface/transformers` v3, outputs 384-dim, then a Dense projection layer (384→768, weights in `dense_layer.bin`) is applied manually in JS
- **Pooling**: Mean pooling, L2-normalized
- 16 verses with nan text are skipped during embedding generation

### Search Strategy
- **Verse-level**: cosine similarity between query and all verse embeddings
- **Chapter ranking**: `max(sliding_window_2, sliding_window_3, 0.94 * max_single_verse)` where sliding windows combine consecutive verse scores
- **UI**: debounced live search (no button), auto-scrolls to best matching window in chapter detail

### Data Formats

Verse embeddings stored as **float16 binary** for compact size (~45 MB for all verses).

**verses_index.json**:
```json
{
  "verses": [{
    "chapter_id": "Genesis_1",
    "verse": "1",
    "index": 0
  }]
}
```

**verses_embeddings.bin** (~45 MB):
Contiguous float16 array. 768 dimensions per verse. Verse at index i starts at byte offset `i * 768 * 2`.

**dense_layer.bin** (~1.2 MB):
Dense projection weights to transform 384-dim ONNX output to 768-dim embedding space.

**verses/Genesis_1.json** (text only):
```json
{
  "verses": [{"verse": "1", "text": "In the beginning..."}]
}
```

### Regenerating Embeddings
```bash
pip install sentence-transformers torch
python scripts/generate_embeddings.py
```
Outputs to `public/data/`.

## Key Code Paths

- Search flow: `App` (debounced) → `useEmbeddingModel.encode` (ONNX + dense layer) → `useChapterSearch.search` (verse cosine sim → chapter ranking)
- Drill-in: `App.handleSelect` → `useVerseHeatmap.loadAndScore` → `ChapterDetail` renders heatmap, auto-scrolls to best window
- Heatmap color: `ChapterDetail.scoreToColor` maps normalized score to transparent→yellow→orange/red
- Binary loading: `useChapterSearch` fetches `verses_embeddings.bin` as `ArrayBuffer` → float16→float32 conversion, indexed via `verses_index.json`
