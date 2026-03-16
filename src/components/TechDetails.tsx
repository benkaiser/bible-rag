import { useState } from 'react';

interface Props {
  chapterCount: number;
  verseCount: number;
  timing?: { search: number; encode: number };
}

export function TechDetails({ chapterCount, verseCount, timing }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="tech-details">
      <button className="tech-toggle" onClick={() => setOpen(!open)}>
        {open ? '▼' : '▶'} Technical Details
      </button>
      {open && (
        <div className="tech-content">
          <table>
            <tbody>
              <tr><td>Embedding model</td><td>MongoDB/mdbr-leaf-ir (22M params)</td></tr>
              <tr><td>In-browser inference</td><td>ONNX Runtime Web + Dense projection layer</td></tr>
              <tr><td>Embedding dims</td><td>768 (Transformer 384 → Dense → 768)</td></tr>
              <tr><td>Verses indexed</td><td>{verseCount.toLocaleString()}</td></tr>
              <tr><td>Chapters</td><td>{chapterCount.toLocaleString()}</td></tr>
              <tr><td>Embedding storage</td><td>Float16 (~45 MB)</td></tr>
              <tr><td>Search strategy</td><td>Per-verse cosine similarity, chapters ranked by best sliding window avg (2–3 verses)</td></tr>
              <tr><td>Pooling</td><td>Mean pooling (CLS excluded)</td></tr>
              {timing && (
                <>
                  <tr><td>Query encode time</td><td>{timing.encode.toFixed(0)} ms</td></tr>
                  <tr><td>Search time</td><td>{timing.search.toFixed(0)} ms</td></tr>
                </>
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
