import { useState, useEffect, useCallback, useRef } from 'react';
import type { FeatureExtractionPipeline } from '@huggingface/transformers';

const DENSE_IN = 384;
const DENSE_OUT = 768;

/** Apply Dense linear projection: out = input @ W^T + bias */
function applyDenseLayer(input: number[], weight: Float32Array, bias: Float32Array): number[] {
  const out = new Array(DENSE_OUT);
  for (let i = 0; i < DENSE_OUT; i++) {
    let sum = bias[i];
    const rowOffset = i * DENSE_IN;
    for (let j = 0; j < DENSE_IN; j++) {
      sum += input[j] * weight[rowOffset + j];
    }
    out[i] = sum;
  }
  return out;
}

function normalize(embedding: number[]): number[] {
  let norm = 0;
  for (const v of embedding) norm += v * v;
  norm = Math.sqrt(norm);
  if (norm === 0) return embedding;
  return embedding.map(v => v / norm);
}

export function useEmbeddingModel() {
  const modelRef = useRef<FeatureExtractionPipeline | null>(null);
  const denseWeightRef = useRef<Float32Array | null>(null);
  const denseBiasRef = useRef<Float32Array | null>(null);
  const loadedRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;

    (async () => {
      try {
        // Dynamic import so the large transformers bundle doesn't block initial render
        const { pipeline } = await import('@huggingface/transformers');

        const [pipe, denseBuf] = await Promise.all([
          pipeline('feature-extraction', 'MongoDB/mdbr-leaf-ir', {
            dtype: 'q8',
          }),
          fetch('/data/dense_layer.bin').then(r => r.arrayBuffer()),
        ]);

        modelRef.current = pipe;
        const allFloats = new Float32Array(denseBuf);
        denseWeightRef.current = allFloats.slice(0, DENSE_OUT * DENSE_IN);
        denseBiasRef.current = allFloats.slice(DENSE_OUT * DENSE_IN);
        setReady(true);
        setLoading(false);
      } catch (e) {
        setError(String(e));
        setLoading(false);
      }
    })();
  }, []);

  const encode = useCallback(async (query: string): Promise<number[] | null> => {
    if (!modelRef.current || !denseWeightRef.current || !denseBiasRef.current) return null;

    // Step 1: ONNX transformer + mean pooling → 384 dims
    const output = await modelRef.current(query, { pooling: 'mean', normalize: false });
    const pooled384 = Array.from(output.data as Float32Array).slice(0, DENSE_IN);

    // Step 2: Dense projection 384 → 768
    const projected = applyDenseLayer(pooled384, denseWeightRef.current, denseBiasRef.current);

    // Step 3: L2 normalize
    return normalize(projected);
  }, [ready]);

  return { loading, error, encode };
}
