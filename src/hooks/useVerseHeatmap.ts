import { useState, useCallback } from 'react';
import type { VerseMeta, VersesMetaData, VerseResult } from '../types';
import { cosineSimilarity } from '../utils/similarity';

const EMBED_DIM = 768;

interface CachedVerses {
  meta: VerseMeta[];
  embeddings: Float32Array;
}

const cache = new Map<string, CachedVerses>();

export interface VerseHeatmapResult {
  verses: VerseResult[];
  /** 0-based indices of the best matching window (2 or 3 consecutive verses) */
  bestWindowStart: number;
  bestWindowEnd: number; // exclusive
}

export function useVerseHeatmap() {
  const [result, setResult] = useState<VerseHeatmapResult | null>(null);
  const [loading, setLoading] = useState(false);

  const clear = useCallback(() => setResult(null), []);

  const loadAndScore = useCallback(async (chapterId: string, queryEmbedding: number[]) => {
    setLoading(true);
    try {
      let data = cache.get(chapterId);
      if (!data) {
        const [metaResp, binResp] = await Promise.all([
          fetch(`${import.meta.env.BASE_URL}data/verses/${chapterId}.json`),
          fetch(`${import.meta.env.BASE_URL}data/verses/${chapterId}.bin`),
        ]);
        const metaData: VersesMetaData = await metaResp.json();
        const buffer = await binResp.arrayBuffer();
        data = {
          meta: metaData.verses,
          embeddings: new Float32Array(buffer),
        };
        cache.set(chapterId, data);
      }

      const scores = data.meta.map((_v, i) =>
        cosineSimilarity(
          queryEmbedding,
          data!.embeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM)
        )
      );

      const verses: VerseResult[] = data.meta.map((v, i) => ({
        verse: v,
        score: scores[i],
      }));

      // Find best sliding window (2 or 3 consecutive verses)
      let bestScore = -Infinity;
      let bestStart = 0;
      let bestEnd = 1;

      // Window of 2
      if (scores.length >= 2) {
        for (let s = 0; s <= scores.length - 2; s++) {
          const avg = (scores[s] + scores[s + 1]) / 2;
          if (avg > bestScore) {
            bestScore = avg;
            bestStart = s;
            bestEnd = s + 2;
          }
        }
      }

      // Window of 3
      if (scores.length >= 3) {
        for (let s = 0; s <= scores.length - 3; s++) {
          const avg = (scores[s] + scores[s + 1] + scores[s + 2]) / 3;
          if (avg > bestScore) {
            bestScore = avg;
            bestStart = s;
            bestEnd = s + 3;
          }
        }
      }

      setResult({ verses, bestWindowStart: bestStart, bestWindowEnd: bestEnd });
    } finally {
      setLoading(false);
    }
  }, []);

  return { result, loading, loadAndScore, clear };
}
