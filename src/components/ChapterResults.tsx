import type { ChapterResult } from '../types';

interface Props {
  results: ChapterResult[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  searching: boolean;
}

export function ChapterResults({ results, selectedId, onSelect, searching }: Props) {
  if (searching) {
    return <div className="results-placeholder">Searching...</div>;
  }

  if (results.length === 0) {
    return null;
  }

  return (
    <div className="chapter-results">
      <h3>Top Matching Chapters</h3>
      <ul>
        {results.map(r => (
          <li
            key={r.chapter.id}
            className={r.chapter.id === selectedId ? 'selected' : ''}
            onClick={() => onSelect(r.chapter.id)}
          >
            <div className="result-header">
              <strong>{r.chapter.title}</strong>
              <span className="score">{r.score.toFixed(4)}</span>
            </div>
            <div className="result-meta">{r.chapter.verse_count} verses</div>
          </li>
        ))}
      </ul>
    </div>
  );
}
