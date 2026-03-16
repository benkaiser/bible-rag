import { describe, it, expect } from 'vitest';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { cosineSimilarity } from '../utils/similarity';

const DATA_DIR = join(__dirname, '../../public/data');
const EMBED_DIM = 768;

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

interface VerseIndex {
  c: string;
  v: string;
}

function loadVerseData() {
  const index: VerseIndex[] = JSON.parse(readFileSync(join(DATA_DIR, 'verses_index.json'), 'utf-8'));
  const binBuf = readFileSync(join(DATA_DIR, 'verses_embeddings.bin'));
  const uint16 = new Uint16Array(binBuf.buffer, binBuf.byteOffset, binBuf.byteLength / 2);
  const float32 = new Float32Array(uint16.length);
  for (let i = 0; i < uint16.length; i++) {
    float32[i] = float16ToFloat32(uint16[i]);
  }
  return { index, embeddings: float32 };
}

describe('Data integrity', () => {
  const { index, embeddings } = loadVerseData();

  it('has ~31000+ verses indexed', () => {
    expect(index.length).toBeGreaterThan(31000);
  });

  it('embeddings size matches index', () => {
    expect(embeddings.length).toBe(index.length * EMBED_DIM);
  });

  it('embeddings are approximately L2-normalized', () => {
    for (const i of [0, 100, 1000, 10000, 20000, 30000]) {
      if (i >= index.length) break;
      const emb = embeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM);
      let norm = 0;
      for (let j = 0; j < EMBED_DIM; j++) norm += emb[j] * emb[j];
      // float16 precision means norm won't be exactly 1.0
      expect(Math.sqrt(norm)).toBeCloseTo(1.0, 1);
    }
  });

  it('per-chapter verse files exist', () => {
    // Spot check a few
    for (const id of ['Genesis_1', 'John_3', 'Revelation_22']) {
      expect(existsSync(join(DATA_DIR, 'verses', `${id}.json`))).toBe(true);
      expect(existsSync(join(DATA_DIR, 'verses', `${id}.bin`))).toBe(true);
    }
  });

  it('chapters.json exists and has 1189 chapters', () => {
    const meta = JSON.parse(readFileSync(join(DATA_DIR, 'chapters.json'), 'utf-8'));
    expect(meta.chapters.length).toBe(1189);
  });

  it('self-similarity is highest (verse matches itself)', () => {
    const queryEmb = embeddings.subarray(0, EMBED_DIM);
    let bestIdx = -1;
    let bestScore = -Infinity;
    // Check first 1000 verses
    for (let i = 0; i < Math.min(1000, index.length); i++) {
      const emb = embeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM);
      const score = cosineSimilarity(queryEmb, emb);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    expect(bestIdx).toBe(0);
    expect(bestScore).toBeCloseTo(1.0, 1);
  });

  it('cosine similarity has meaningful spread', () => {
    const queryEmb = embeddings.subarray(0, EMBED_DIM);
    let maxScore = -Infinity;
    let minScore = Infinity;
    for (let i = 0; i < Math.min(5000, index.length); i++) {
      const emb = embeddings.subarray(i * EMBED_DIM, (i + 1) * EMBED_DIM);
      const score = cosineSimilarity(queryEmb, emb);
      if (score > maxScore) maxScore = score;
      if (score < minScore) minScore = score;
    }
    expect(maxScore).toBeGreaterThan(0.9);
    expect(minScore).toBeLessThan(0.5);
    expect(maxScore - minScore).toBeGreaterThan(0.5);
  });
});
