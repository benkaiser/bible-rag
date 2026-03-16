import { useState, useCallback } from 'react';
import { SearchBar } from './components/SearchBar';
import { ChapterResults } from './components/ChapterResults';
import { ChapterDetail } from './components/ChapterDetail';
import { TechDetails } from './components/TechDetails';
import { useEmbeddingModel } from './hooks/useEmbeddingModel';
import { useChapterSearch } from './hooks/useChapterSearch';
import { useVerseHeatmap } from './hooks/useVerseHeatmap';
import type { ChapterResult } from './types';
import './App.css';

function App() {
  const { loading: modelLoading, encode, error: modelError } = useEmbeddingModel();
  const { chapters, verseCount, loading: dataLoading, loadStatus, loadProgress, loadDetail, error: dataError, search } = useChapterSearch();
  const { result: verseResult, loading: versesLoading, loadAndScore, clear: clearVerses } = useVerseHeatmap();

  const [results, setResults] = useState<ChapterResult[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [queryEmbedding, setQueryEmbedding] = useState<number[] | null>(null);
  const [timing, setTiming] = useState<{ encode: number; search: number } | undefined>();

  const isLoading = modelLoading || dataLoading;

  const handleSearch = useCallback(async (query: string) => {
    setSearching(true);
    setSelectedId(null);
    clearVerses();

    const encodeStart = performance.now();
    const emb = await encode(query);
    const encodeTime = performance.now() - encodeStart;

    if (!emb) {
      setSearching(false);
      return;
    }

    setQueryEmbedding(emb);

    const searchStart = performance.now();
    const results = search(emb);
    const searchTime = performance.now() - searchStart;

    setResults(results);
    setTiming({ encode: encodeTime, search: searchTime });
    setSearching(false);
  }, [encode, search]);

  const handleSelect = useCallback(async (id: string) => {
    setSelectedId(id);
    if (queryEmbedding) {
      await loadAndScore(id, queryEmbedding);
    }
  }, [queryEmbedding, loadAndScore]);

  const selectedChapter = results.find(r => r.chapter.id === selectedId);
  const error = modelError || dataError;

  return (
    <div className="app">
      <header>
        <h1>Bible RAG</h1>
        <p className="subtitle">Semantic search across the entire Bible — find chapters by concept, topic, or theme</p>
      </header>

      {results.length === 0 && !searching && (
        <div className="search-tips">
          <h3>Search Tips</h3>
          <p>This uses <strong>semantic search</strong> — describe what you're looking for in natural language. Longer, more descriptive queries work better than short keywords.</p>
          <div className="tips-columns">
            <div className="tip-good">
              <h4>Works well</h4>
              <ul>
                <li>"jesus feeds five thousand with loaves and fish"</li>
                <li>"paul shipwrecked on the way to rome"</li>
                <li>"a man is beaten and helped by a stranger from samaria"</li>
                <li>"david kills the giant with a sling and stone"</li>
              </ul>
            </div>
            <div className="tip-bad">
              <h4>Less effective</h4>
              <ul>
                <li>"feeding" <span className="tip-why">— too vague</span></li>
                <li>"Romans 8" <span className="tip-why">— use a Bible app for references</span></li>
                <li>"love" <span className="tip-why">— too broad, appears everywhere</span></li>
                <li>"Goliath" <span className="tip-why">— keywords alone miss context</span></li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {error && <div className="error">Error: {error}</div>}

      <SearchBar
        onSearch={handleSearch}
        disabled={isLoading}
        searching={searching}
        modelLoading={modelLoading}
        dataLoadStatus={loadStatus}
        dataLoadProgress={loadProgress}
        dataLoadDetail={loadDetail}
      />

      <div className="content">
        <div className="left-panel">
          <ChapterResults
            results={results}
            selectedId={selectedId}
            onSelect={handleSelect}
            searching={searching}
          />
        </div>
        <div className="right-panel">
          <ChapterDetail
            chapterTitle={selectedChapter?.chapter.title ?? ''}
            verses={verseResult?.verses ?? null}
            bestWindowStart={verseResult?.bestWindowStart ?? 0}
            bestWindowEnd={verseResult?.bestWindowEnd ?? 0}
            loading={versesLoading}
          />
        </div>
      </div>

      <TechDetails chapterCount={chapters.length} verseCount={verseCount} timing={timing} />
    </div>
  );
}

export default App;
