# Bible RAG

Semantic search across the entire Berean Standard Bible (31,086 searchable verses, 1,189 chapters). Search by concept, topic, or theme and get back the most relevant chapters with per-verse relevance heatmaps.

## How It Works

Uses **MongoDB/mdbr-leaf-ir** (22M params) for both document and query embeddings (symmetric):
- **Offline**: Python generates verse-level and chapter-level embeddings (768-dim, float16)
- **In-browser**: ONNX model loaded via `@huggingface/transformers`, with a separate Dense projection layer (384→768) applied manually in JS. Mean pooling.
- Verses scored by cosine similarity; chapters ranked by `max(sliding_window_2, sliding_window_3, 0.94 * max_single_verse)`
- Clicking a chapter shows verse-level relevance heatmap, auto-scrolling to the best matching window

## Quick Start

### 1. Generate Embeddings

```bash
pip install sentence-transformers torch
python scripts/generate_embeddings.py
```

Outputs to `public/data/`:
- `verses_embeddings.bin` + `verses_index.json` (~45 MB) — all verse embeddings (float16)
- `verses/*.json` (1,189 files) — per-chapter verse text for heatmap drill-in
- `chapters_embeddings.bin` — chapter windowed embeddings (legacy, not used for search)
- `dense_layer.bin` (~1.2 MB) — Dense projection weights for in-browser inference

### 2. Run the Web App

```bash
npm install
npm run dev
```

### 3. Build for Production

```bash
npm run build
```

Static files output to `dist/`. Deploy to any static host.

## Usage

1. Wait for the model, dense layer, and embeddings to load (progress bar shown)
2. Start typing — debounced live search runs automatically (no button)
3. Click a chapter result to see verse-level relevance heatmap
4. Warmer colors = higher semantic relevance to your query
