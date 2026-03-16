import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { cosineSimilarity } from '../utils/similarity';

const DATA_DIR = join(__dirname, '../../public/data');
const EMBED_DIM = 768;

interface VerseIndex {
  c: string;
  v: string;
}

interface Benchmark {
  name: string;
  query: string;
  expected: string[];
  maxRank: number;
}

function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    let val = frac / 1024;
    val *= 2 ** -14;
    return sign ? -val : val;
  }
  if (exp === 0x1f) return frac ? NaN : (sign ? -Infinity : Infinity);
  const val = 2 ** (exp - 15) * (1 + frac / 1024);
  return sign ? -val : val;
}

function loadVerseSearch() {
  const index: VerseIndex[] = JSON.parse(readFileSync(join(DATA_DIR, 'verses_index.json'), 'utf-8'));
  const binBuf = readFileSync(join(DATA_DIR, 'verses_embeddings.bin'));
  const uint16 = new Uint16Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 2);
  const float32 = new Float32Array(uint16.length);
  for (let i = 0; i < uint16.length; i++) {
    float32[i] = float16ToFloat32(uint16[i]);
  }
  return { index, embeddings: float32 };
}

function loadFixtures() {
  const meta: { benchmarks: Benchmark[] } = JSON.parse(
    readFileSync(join(__dirname, 'benchmark_fixtures.json'), 'utf-8')
  );
  const binBuf = readFileSync(join(__dirname, 'benchmark_fixtures.bin'));
  const allFloats = new Float32Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 4);

  return meta.benchmarks.map((b, i) => ({
    ...b,
    queryEmb: allFloats.slice(i * EMBED_DIM, (i + 1) * EMBED_DIM),
  }));
}

function searchChapters(
  index: VerseIndex[],
  embeddings: Float32Array,
  queryEmb: Float32Array,
): { id: string; score: number }[] {
  // Group verse scores by chapter
  const chapterSims = new Map<string, number[]>();

  for (let i = 0; i < index.length; i++) {
    const emb = embeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM);
    const score = cosineSimilarity(queryEmb, emb);
    const chId = index[i].c;
    let sims = chapterSims.get(chId);
    if (!sims) {
      sims = [];
      chapterSims.set(chId, sims);
    }
    sims.push(score);
  }

  // Rank by max(sliding_window_2, sliding_window_3, 0.94 * max_single_verse)
  const SINGLE_VERSE_WEIGHT = 0.94;
  const results: { id: string; score: number }[] = [];
  for (const [chId, sims] of chapterSims) {
    let best = Math.max(...sims) * SINGLE_VERSE_WEIGHT;
    if (sims.length >= 2) {
      for (let s = 0; s <= sims.length - 2; s++) {
        best = Math.max(best, (sims[s] + sims[s + 1]) / 2);
      }
    }
    if (sims.length >= 3) {
      for (let s = 0; s <= sims.length - 3; s++) {
        best = Math.max(best, (sims[s] + sims[s + 1] + sims[s + 2]) / 3);
      }
    }
    results.push({ id: chId, score: best });
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

/**
 * Benchmark: 10 well-known Bible passages.
 *
 * Query embeddings are pre-computed with the full mdbr-leaf-ir Python
 * pipeline. Search is done at verse level (matching the browser app),
 * with chapters ranked by their best-matching verse.
 */
describe('Chapter search benchmark (verse-level)', () => {
  const { index, embeddings } = loadVerseSearch();
  const benchmarks = loadFixtures();

  for (const bench of benchmarks) {
    it(`"${bench.query}" → ${bench.expected[0]} (top ${bench.maxRank})`, () => {
      const results = searchChapters(index, embeddings, bench.queryEmb);

      let bestRank = Infinity;
      let bestId = bench.expected[0];
      for (const exp of bench.expected) {
        const rank = results.findIndex(r => r.id === exp) + 1;
        if (rank > 0 && rank < bestRank) {
          bestRank = rank;
          bestId = exp;
        }
      }

      const top5 = results.slice(0, 5).map(r => `${r.id}:${r.score.toFixed(4)}`).join(', ');
      console.log(`  rank=${bestRank} (${bestId}), top5=[${top5}]`);

      expect(
        bestRank,
        `Expected one of [${bench.expected}] in top ${bench.maxRank}, ` +
        `got #${bestRank} (${bestId}). Top 5: ${results.slice(0, 5).map(r => r.id).join(', ')}`
      ).toBeLessThanOrEqual(bench.maxRank);
    });
  }
});
