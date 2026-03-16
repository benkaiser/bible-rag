import { useEffect, useRef } from 'react';
import type { VerseResult } from '../types';

interface Props {
  chapterTitle: string;
  verses: VerseResult[] | null;
  bestWindowStart: number;
  bestWindowEnd: number;
  loading: boolean;
}

function scoreToColor(score: number, minScore: number, maxScore: number): string {
  if (maxScore === minScore) return 'transparent';
  const t = Math.max(0, Math.min(1, (score - minScore) / (maxScore - minScore)));
  if (t < 0.3) return 'transparent';
  const r = 255;
  const g = Math.round(220 - t * 140);
  const b = Math.round(50 - t * 50);
  const a = 0.15 + t * 0.55;
  return `rgba(${r}, ${g}, ${b}, ${a})`;
}

export function ChapterDetail({ chapterTitle, verses, bestWindowStart, bestWindowEnd, loading }: Props) {
  const windowRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to best window when verses load
  useEffect(() => {
    if (windowRef.current && containerRef.current) {
      const container = containerRef.current;
      const target = windowRef.current;
      // Scroll so the best window is roughly 1/3 from the top
      const targetTop = target.offsetTop - container.offsetTop;
      container.scrollTop = Math.max(0, targetTop - container.clientHeight / 3);
    }
  }, [verses, bestWindowStart]);

  if (loading) {
    return <div className="chapter-detail"><p>Loading verses...</p></div>;
  }

  if (!verses) {
    return <div className="chapter-detail placeholder"><p>Click a chapter to view verses with relevance heatmap</p></div>;
  }

  const scores = verses.map(v => v.score);
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);

  return (
    <div className="chapter-detail" ref={containerRef}>
      <h3>{chapterTitle}</h3>
      <div className="verses">
        {verses.map((v, i) => {
          const inWindow = i >= bestWindowStart && i < bestWindowEnd;
          const isWindowStart = i === bestWindowStart;
          const isWindowEnd = i === bestWindowEnd - 1;

          return (
            <div
              key={v.verse.verse}
              ref={isWindowStart ? windowRef : undefined}
              className={
                'verse' +
                (inWindow ? ' best-window' : '') +
                (isWindowStart ? ' window-start' : '') +
                (isWindowEnd ? ' window-end' : '')
              }
              style={{ backgroundColor: scoreToColor(v.score, minScore, maxScore) }}
            >
              <span className="verse-num">{v.verse.verse}</span>
              <span className="verse-text">{v.verse.text}</span>
              <span className="verse-score">{v.score.toFixed(4)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
