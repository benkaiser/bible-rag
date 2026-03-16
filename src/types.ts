export interface ChapterMeta {
  id: string;
  book: string;
  chapter: string;
  title: string;
  verse_count: number;
  window_count: number;
}

export interface ChapterEntry extends ChapterMeta {
  embeddings: Float32Array[];  // window_count arrays of EMBED_DIM floats
}

export interface ChaptersMetaData {
  chapters: ChapterMeta[];
}

export interface VerseMeta {
  verse: string;
  text: string;
}

export interface VersesMetaData {
  verses: VerseMeta[];
}

export interface ChapterResult {
  chapter: ChapterEntry;
  score: number;
  preview: string;
}

export interface VerseResult {
  verse: VerseMeta;
  score: number;
}
