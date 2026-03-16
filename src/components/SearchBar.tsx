import { useEffect, useRef, useState } from 'react';
import type { DataLoadStatus } from '../hooks/useChapterSearch';

interface Props {
  onSearch: (query: string) => void;
  disabled: boolean;
  searching: boolean;
  modelLoading: boolean;
  dataLoadStatus: DataLoadStatus;
  dataLoadProgress: number;
  dataLoadDetail: string;
}

export function SearchBar({ onSearch, disabled, searching, modelLoading, dataLoadStatus, dataLoadProgress }: Props) {
  const [query, setQuery] = useState('');
  const pendingRef = useRef<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (disabled || !query.trim()) return;

    if (timerRef.current) clearTimeout(timerRef.current);

    timerRef.current = setTimeout(() => {
      if (searching) {
        pendingRef.current = query.trim();
      } else {
        onSearch(query.trim());
      }
    }, 300);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [query, disabled]);

  useEffect(() => {
    if (!searching && pendingRef.current) {
      const pending = pendingRef.current;
      pendingRef.current = null;
      onSearch(pending);
    }
  }, [searching]);

  const isLoading = modelLoading || dataLoadStatus !== 'ready';

  // Progress bar based on data download (the one we can track)
  const progressPct = dataLoadStatus === 'ready' ? 100 : dataLoadProgress;

  const statusText = isLoading
    ? 'Loading semantic search models and verse embeddings...'
    : 'Ready — search for concepts, topics, or themes';

  return (
    <div className="search-bar">
      <input
        type="text"
        value={query}
        onChange={e => setQuery(e.target.value)}
        placeholder="Search the Bible... (e.g., 'love your neighbor')"
        disabled={disabled}
      />
      <div className={`status ${isLoading ? 'loading' : 'ready'}`}>
        {isLoading && <div className="progress-bar"><div className="progress-fill" style={{ width: `${progressPct}%` }} /></div>}
        <span>{statusText}</span>
      </div>
    </div>
  );
}
