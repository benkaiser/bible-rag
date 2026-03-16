import { useState, useEffect, useRef } from 'react';
import type { ChapterResult } from '../types';
import { cosineSimilarity } from '../utils/similarity';

const EMBED_DIM = 768;

interface VerseIndex {
  c: string; // chapter id
  v: string; // verse number
}

export type DataLoadStatus = 'idle' | 'downloading' | 'processing' | 'ready' | 'error';

export function useChapterSearch() {
  const [verseIndex, setVerseIndex] = useState<VerseIndex[]>([]);
  const [verseEmbeddings, setVerseEmbeddings] = useState<Float32Array | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadStatus, setLoadStatus] = useState<DataLoadStatus>('idle');
  const [loadProgress, setLoadProgress] = useState(0);
  const [loadDetail, setLoadDetail] = useState('');
  const [error, setError] = useState<string | null>(null);
  const loadedRef = useRef(false);

  // Build chapter metadata lookup
  const [chapterMeta, setChapterMeta] = useState<Map<string, { book: string; chapter: string; title: string; verse_count: number }>>(new Map());

  useEffect(() => {
    // Prevent double-load in React StrictMode
    if (loadedRef.current) return;
    loadedRef.current = true;

    (async () => {
      try {
        setLoadStatus('downloading');
        setLoadDetail('Downloading verse data...');

        const [indexResp, embResp, metaResp] = await Promise.all([
          fetch('/data/verses_index.json'),
          fetch('/data/verses_embeddings.bin'),
          fetch('/data/chapters.json'),
        ]);

        // Stream the large embeddings file with progress
        const contentLength = embResp.headers.get('Content-Length');
        const totalBytes = contentLength ? parseInt(contentLength, 10) : 45_500_000; // fallback estimate
        let receivedBytes = 0;

        const reader = embResp.body!.getReader();
        const chunks: Uint8Array[] = [];

        // Read index and meta in parallel with streaming embeddings
        const indexPromise = indexResp.json() as Promise<VerseIndex[]>;
        const metaPromise = metaResp.json() as Promise<{ chapters: { id: string; book: string; chapter: string; title: string; verse_count: number }[] }>;

        // eslint-disable-next-line no-constant-condition
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          chunks.push(value);
          receivedBytes += value.length;
          const pct = Math.min(99, Math.round((receivedBytes / totalBytes) * 100));
          const mbReceived = (receivedBytes / 1024 / 1024).toFixed(0);
          const mbTotal = (totalBytes / 1024 / 1024).toFixed(0);
          setLoadDetail(`Downloading verse embeddings (${mbReceived}/${mbTotal} MB)... ${pct}%`);
          setLoadProgress(pct);
        }

        // Combine chunks into single ArrayBuffer
        const embBuf = new Uint8Array(receivedBytes);
        let offset = 0;
        for (const chunk of chunks) {
          embBuf.set(chunk, offset);
          offset += chunk.length;
        }

        const [index, meta] = await Promise.all([indexPromise, metaPromise]);

        setLoadStatus('processing');
        setLoadDetail('Converting embeddings...');

        // Yield to UI before heavy computation
        await new Promise(r => setTimeout(r, 0));

        // Convert float16 buffer to float32 for computation
        const float16 = new Uint16Array(embBuf.buffer, embBuf.byteOffset, embBuf.byteLength / 2);
        const float32 = new Float32Array(float16.length);
        for (let i = 0; i < float16.length; i++) {
          float32[i] = float16ToFloat32(float16[i]);
        }

        const metaMap = new Map<string, { book: string; chapter: string; title: string; verse_count: number }>();
        for (const ch of meta.chapters) {
          metaMap.set(ch.id, { book: ch.book, chapter: ch.chapter, title: ch.title, verse_count: ch.verse_count });
        }

        setVerseIndex(index);
        setVerseEmbeddings(float32);
        setChapterMeta(metaMap);
        setLoadStatus('ready');
        setLoadDetail(`${index.length.toLocaleString()} verses loaded`);
        setLoadProgress(100);
        setLoading(false);
      } catch (e) {
        setError(String(e));
        setLoadStatus('error');
        setLoadDetail(String(e));
        setLoading(false);
      }
    })();
  }, []);

  function search(queryEmbedding: number[], topK = 25): ChapterResult[] {
    if (!verseEmbeddings || verseIndex.length === 0) return [];

    // Score every verse and group by chapter
    const chapterSims = new Map<string, number[]>();

    for (let i = 0; i < verseIndex.length; i++) {
      const emb = verseEmbeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM);
      const score = cosineSimilarity(queryEmbedding, emb);
      const chId = verseIndex[i].c;
      let sims = chapterSims.get(chId);
      if (!sims) {
        sims = [];
        chapterSims.set(chId, sims);
      }
      sims.push(score);
    }

    // Rank chapters by: max(sliding_window_2, sliding_window_3, 0.94 * max_single_verse)
    // The sliding windows catch multi-verse passages, while the discounted single-verse
    // max rescues cases where one standout verse dominates (e.g. Good Samaritan v33)
    // without letting common short phrases overwhelm the ranking.
    const SINGLE_VERSE_WEIGHT = 0.94;
    const chapterBestScore = new Map<string, number>();
    for (const [chId, sims] of chapterSims) {
      // Discounted single-verse max
      let best = Math.max(...sims) * SINGLE_VERSE_WEIGHT;
      // Sliding window of 2
      if (sims.length >= 2) {
        for (let s = 0; s <= sims.length - 2; s++) {
          best = Math.max(best, (sims[s] + sims[s + 1]) / 2);
        }
      }
      // Sliding window of 3
      if (sims.length >= 3) {
        for (let s = 0; s <= sims.length - 3; s++) {
          best = Math.max(best, (sims[s] + sims[s + 1] + sims[s + 2]) / 3);
        }
      }
      chapterBestScore.set(chId, best);
    }

    // Convert to results
    const results: ChapterResult[] = [];
    for (const [chId, score] of chapterBestScore) {
      const meta = chapterMeta.get(chId);
      if (!meta) continue;
      results.push({
        chapter: {
          id: chId,
          book: meta.book,
          chapter: meta.chapter,
          title: meta.title,
          verse_count: meta.verse_count,
          window_count: 0,
          embeddings: [],
        },
        score,
        preview: '',
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  /**
   * Search for the best passage matches across the entire Bible at verse granularity.
   * Uses the same heuristic as chapter search (sliding window of 2, 3, and discounted single verse).
   * Returns top passages with their chapter ID, best verse index within the chapter, and score.
   */
  function searchPassages(queryEmbedding: number[], topK = 10): { chapterId: string; title: string; score: number; bestVerseIdx: number; verseCount: number }[] {
    if (!verseEmbeddings || verseIndex.length === 0) return [];

    const SINGLE_VERSE_WEIGHT = 0.94;

    // Score every verse
    const scores = new Float32Array(verseIndex.length);
    for (let i = 0; i < verseIndex.length; i++) {
      const emb = verseEmbeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM);
      scores[i] = cosineSimilarity(queryEmbedding, emb);
    }

    // Group verse indices by chapter
    const chapterVerseRanges = new Map<string, { start: number; end: number }>();
    for (let i = 0; i < verseIndex.length; i++) {
      const chId = verseIndex[i].c;
      const range = chapterVerseRanges.get(chId);
      if (!range) {
        chapterVerseRanges.set(chId, { start: i, end: i + 1 });
      } else {
        range.end = i + 1;
      }
    }

    // For each chapter, find the best passage and track which verse is the anchor
    const results: { chapterId: string; title: string; score: number; bestVerseIdx: number; verseCount: number }[] = [];

    for (const [chId, range] of chapterVerseRanges) {
      const meta = chapterMeta.get(chId);
      if (!meta) continue;

      let best = -1;
      let bestIdx = range.start;
      const len = range.end - range.start;

      // Discounted single-verse max
      for (let i = range.start; i < range.end; i++) {
        const s = scores[i] * SINGLE_VERSE_WEIGHT;
        if (s > best) { best = s; bestIdx = i; }
      }

      // Sliding window of 2
      if (len >= 2) {
        for (let i = range.start; i < range.end - 1; i++) {
          const s = (scores[i] + scores[i + 1]) / 2;
          if (s > best) { best = s; bestIdx = i; }
        }
      }

      // Sliding window of 3
      if (len >= 3) {
        for (let i = range.start; i < range.end - 2; i++) {
          const s = (scores[i] + scores[i + 1] + scores[i + 2]) / 3;
          if (s > best) { best = s; bestIdx = i; }
        }
      }

      results.push({
        chapterId: chId,
        title: meta.title,
        score: best,
        bestVerseIdx: bestIdx - range.start, // relative to chapter start
        verseCount: len,
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  const chapters = Array.from(chapterMeta.values());

  return { chapters, verseCount: verseIndex.length, loading, loadStatus, loadProgress, loadDetail, error, search, searchPassages };
}

/** Convert a IEEE 754 float16 (stored as uint16) to float32 */
function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    // Subnormal
    let val = frac / 1024;
    val *= 2 ** -14;
    return sign ? -val : val;
  }
  if (exp === 0x1f) {
    return frac ? NaN : (sign ? -Infinity : Infinity);
  }

  const val = 2 ** (exp - 15) * (1 + frac / 1024);
  return sign ? -val : val;
}
