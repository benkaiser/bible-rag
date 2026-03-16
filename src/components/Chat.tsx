import { useState, useRef, useCallback, useEffect } from 'react';

// Display message types
interface TextMessage {
  type: 'text';
  role: 'user' | 'assistant';
  content: string;
}

interface ThinkingMessage {
  type: 'thinking';
  content: string;
}

interface ToolCallMessage {
  type: 'tool_call';
  query: string;
  result: string;
}

type DisplayMessage = TextMessage | ThinkingMessage | ToolCallMessage;

// Internal message format for API calls
interface ApiMessage {
  role: 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

const SYSTEM_PROMPT = `You are a Bible study assistant grounded in the Berean Standard Bible (BSB).

IMPORTANT: Always use the provided tools to find and verify passages. Do not quote scripture from memory — use the tools to ensure accuracy.

You have two tools:
- **search_bible**: Semantic search — describe what you're looking for in natural language. Best for topics, themes, and finding passages when you don't know the exact reference.
- **lookup_passage**: Direct lookup by book, chapter, and optional verse range. Use this when you know the specific reference (e.g. "Romans 8", "John 3:16", "Genesis 1:1-5").

When responding:
- Quote verses with their full reference (e.g. "John 3:16")
- Provide context and explanation alongside the verses
- If asked about themes or topics, search for them and present what the Bible says
- You may call tools multiple times with different queries to find comprehensive results`;

const SEARCH_TOOL = {
  type: 'function' as const,
  function: {
    name: 'search_bible',
    description: 'Semantically search the entire Bible (BSB translation, 31,000+ verses). Returns the most relevant verses based on meaning/content similarity. Use this when you want to find passages by topic or theme. Describe the content you are looking for in natural language (e.g. "all things work together for good for those who love God"). Do NOT use book/chapter references here — use lookup_passage for that.',
    parameters: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: "Natural language description of what to find (e.g. 'jesus heals a blind man', 'faith without works')",
        },
      },
      required: ['query'],
    },
  },
};

const LOOKUP_TOOL = {
  type: 'function' as const,
  function: {
    name: 'lookup_passage',
    description: 'Look up a specific Bible passage by book, chapter, and optional verse range. Use this when you know the exact reference (e.g. book="Romans", chapter=8 or book="John", chapter=3, verse_start=16, verse_end=17). Returns the full text of the requested verses.',
    parameters: {
      type: 'object',
      properties: {
        book: {
          type: 'string',
          description: "Book name (e.g. 'Genesis', '1 Corinthians', 'Psalm', 'Song of Solomon')",
        },
        chapter: {
          type: 'number',
          description: 'Chapter number',
        },
        verse_start: {
          type: 'number',
          description: 'Starting verse number (optional — if omitted, returns the entire chapter)',
        },
        verse_end: {
          type: 'number',
          description: 'Ending verse number (optional — if omitted, returns only verse_start)',
        },
      },
      required: ['book', 'chapter'],
    },
  },
};

interface ModelInfo {
  id: string;
  name: string;
  promptPrice: number; // per 1M tokens
  completionPrice: number; // per 1M tokens
  contextLength: number | null;
}

const DEFAULT_MODEL = 'openrouter/free';
const MODEL_STORAGE_KEY = 'openrouter_selected_model';

interface ChatProps {
  apiKey: string | null;
  connecting: boolean;
  onConnect: () => void;
  onDisconnect: () => void;
  encode: (query: string) => Promise<number[] | null>;
  searchPassages: (embedding: number[], topK?: number) => { chapterId: string; title: string; score: number; bestVerseIdx: number; verseCount: number }[];
  isLoading: boolean;
}

function ToolCallDisplay({ query, result }: { query: string; result: string }) {
  const [expanded, setExpanded] = useState(false);
  // If query looks like a reference (starts with capital + has a number), show "Looked up"
  const isLookup = /^[A-Z0-9]/.test(query) && /\d/.test(query) && !query.includes(' for ');
  const label = isLookup ? 'Looked up: ' : 'Searched Bible for: ';
  return (
    <div className="chat-tool-call" onClick={() => setExpanded(!expanded)}>
      <div className="chat-tool-call-header">
        <span className="chat-tool-call-icon">{expanded ? '▼' : '▶'}</span>
        <span className="chat-tool-call-label">{label}</span>
        <span className="chat-tool-call-query">{isLookup ? query : `"${query}"`}</span>
      </div>
      {expanded && (
        <pre className="chat-tool-call-result">{result}</pre>
      )}
    </div>
  );
}

function ThinkingDisplay({ content }: { content: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="chat-thinking" onClick={() => setExpanded(!expanded)}>
      <div className="chat-thinking-header">
        <span className="chat-tool-call-icon">{expanded ? '▼' : '▶'}</span>
        <span className="chat-thinking-label">Thinking...</span>
      </div>
      {expanded && (
        <div className="chat-thinking-content">{content}</div>
      )}
    </div>
  );
}

function formatContext(len: number | null): string {
  if (!len) return '—';
  if (len >= 1_000_000) return `${(len / 1_000_000).toFixed(1)}M`;
  if (len >= 1_000) return `${Math.round(len / 1_000)}k`;
  return String(len);
}

function formatPrice(price: number): string {
  if (price === 0) return 'Free';
  if (price < 0.01) return `$${price.toFixed(4)}`;
  if (price < 1) return `$${price.toFixed(3)}`;
  return `$${price.toFixed(2)}`;
}

function ModelPickerModal({ models, selectedModel, onSelect, onClose }: {
  models: ModelInfo[];
  selectedModel: string;
  onSelect: (id: string) => void;
  onClose: () => void;
}) {
  const [search, setSearch] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    searchRef.current?.focus();
    const handleEsc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleEsc);
    return () => window.removeEventListener('keydown', handleEsc);
  }, [onClose]);

  const filtered = search
    ? models.filter(m => m.name.toLowerCase().includes(search.toLowerCase()) || m.id.toLowerCase().includes(search.toLowerCase()))
    : models;

  return (
    <div className="model-modal-backdrop" onClick={onClose}>
      <div className="model-modal" onClick={e => e.stopPropagation()}>
        <div className="model-modal-header">
          <h3>Select Model</h3>
          <button className="model-modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="model-modal-search">
          <input
            ref={searchRef}
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search models..."
            className="model-modal-search-input"
          />
        </div>
        <div className="model-modal-list">
          {filtered.length === 0 && (
            <div className="model-modal-empty">No models match your search</div>
          )}
          {filtered.map(m => (
            <div
              key={m.id}
              className={`model-modal-row ${m.id === selectedModel ? 'model-modal-row-selected' : ''}`}
              onClick={() => { onSelect(m.id); onClose(); }}
            >
              <div className="model-modal-row-main">
                <span className="model-modal-row-name">{m.name}</span>
                {m.promptPrice === 0 && m.completionPrice === 0 && (
                  <span className="model-modal-badge-free">Free</span>
                )}
              </div>
              <div className="model-modal-row-details">
                <span className="model-modal-row-detail">
                  <span className="model-modal-row-label">In:</span> {formatPrice(m.promptPrice)}/M
                </span>
                <span className="model-modal-row-detail">
                  <span className="model-modal-row-label">Out:</span> {formatPrice(m.completionPrice)}/M
                </span>
                <span className="model-modal-row-detail">
                  <span className="model-modal-row-label">Context:</span> {formatContext(m.contextLength)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function Chat({ apiKey, connecting, onConnect, onDisconnect, encode, searchPassages, isLoading }: ChatProps) {
  const [displayMessages, setDisplayMessages] = useState<DisplayMessage[]>([]);
  const [input, setInput] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [toolStatus, setToolStatus] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState(() => localStorage.getItem(MODEL_STORAGE_KEY) || DEFAULT_MODEL);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Keep full API message history separate from display
  const apiHistoryRef = useRef<ApiMessage[]>([]);
  const modelsFetchedRef = useRef(false);

  // Fetch models when API key becomes available
  useEffect(() => {
    if (!apiKey || modelsFetchedRef.current) return;
    modelsFetchedRef.current = true;
    setModelsLoading(true);

    fetch('https://openrouter.ai/api/v1/models', {
      headers: { 'Authorization': `Bearer ${apiKey}` },
    })
      .then(r => r.json())
      .then(data => {
        if (data.data) {
          const filtered: ModelInfo[] = data.data
            .filter((m: { supported_parameters?: string[]; architecture?: { output_modalities?: string[] } }) =>
              m.supported_parameters?.includes('tools') &&
              m.architecture?.output_modalities?.includes('text')
            )
            .map((m: { id: string; name: string; pricing?: { prompt?: string; completion?: string }; context_length?: number | null }) => ({
              id: m.id,
              name: m.name,
              promptPrice: parseFloat(m.pricing?.prompt || '0') * 1_000_000,
              completionPrice: parseFloat(m.pricing?.completion || '0') * 1_000_000,
              contextLength: m.context_length ?? null,
            }));
          // Keep the API's default sort order (popular/new first)
          setModels(filtered);
        }
      })
      .catch(console.error)
      .finally(() => setModelsLoading(false));
  }, [apiKey]);

  const handleModelSelect = useCallback((modelId: string) => {
    setSelectedModel(modelId);
    localStorage.setItem(MODEL_STORAGE_KEY, modelId);
  }, []);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [displayMessages, toolStatus]);

  const fetchVerseTexts = useCallback(async (chapterId: string): Promise<{ verse: string; text: string }[]> => {
    try {
      const resp = await fetch(`/data/verses/${chapterId}.json`);
      const data = await resp.json();
      return data.verses || [];
    } catch {
      return [];
    }
  }, []);

  const executeSearch = useCallback(async (query: string): Promise<string> => {
    const emb = await encode(query);
    if (!emb) return 'Error: embedding model not ready';

    const results = searchPassages(emb, 10);
    const parts: string[] = [];

    for (const result of results) {
      const verses = await fetchVerseTexts(result.chapterId);
      if (verses.length === 0) continue;

      // Show 10-verse context window centered on the best matching verse
      const contextSize = 10;
      const half = Math.floor(contextSize / 2);
      const start = Math.max(0, result.bestVerseIdx - half);
      const end = Math.min(verses.length, start + contextSize);
      const contextVerses = verses.slice(start, end);

      const verseLines = contextVerses.map(v => `v${v.verse}: "${v.text}"`).join('\n');
      parts.push(`## ${result.title} (score: ${result.score.toFixed(4)})\n${verseLines}`);
    }

    return parts.join('\n\n');
  }, [encode, searchPassages, fetchVerseTexts]);

  const executeLookup = useCallback(async (book: string, chapter: number, verseStart?: number, verseEnd?: number): Promise<string> => {
    const chapterId = `${book}_${chapter}`;
    const verses = await fetchVerseTexts(chapterId);
    if (verses.length === 0) {
      return `No verses found for ${book} ${chapter}. Check the book name and chapter number.`;
    }

    let selected = verses;
    if (verseStart !== undefined) {
      const end = verseEnd ?? verseStart;
      selected = verses.filter(v => {
        const num = parseInt(v.verse, 10);
        return num >= verseStart && num <= end;
      });
      if (selected.length === 0) {
        return `Verses ${verseStart}-${end} not found in ${book} ${chapter}. The chapter has ${verses.length} verses.`;
      }
    }

    const header = verseStart !== undefined
      ? `## ${book} ${chapter}:${verseStart}${verseEnd && verseEnd !== verseStart ? `-${verseEnd}` : ''}`
      : `## ${book} ${chapter}`;
    const verseLines = selected.map(v => `v${v.verse}: "${v.text}"`).join('\n');
    return `${header}\n${verseLines}`;
  }, [fetchVerseTexts]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || streaming || !apiKey) return;

    setInput('');
    setDisplayMessages(prev => [...prev, { type: 'text', role: 'user', content: text }]);
    apiHistoryRef.current = [
      ...apiHistoryRef.current,
      { role: 'user', content: text },
    ];
    setStreaming(true);

    try {
      await runToolLoop(apiKey);
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setDisplayMessages(prev => [...prev, { type: 'text', role: 'assistant', content: `Error: ${err}` }]);
      }
    } finally {
      setStreaming(false);
      setToolStatus(null);
    }
  }, [input, streaming, apiKey]);

  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  }, []);

  const runToolLoop = async (key: string) => {
    const MAX_ITERATIONS = 10;
    for (let i = 0; i < MAX_ITERATIONS; i++) {
      abortRef.current = new AbortController();

      const resp = await fetch('https://openrouter.ai/api/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${key}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [
            { role: 'system', content: SYSTEM_PROMPT },
            ...apiHistoryRef.current,
          ],
          tools: [SEARCH_TOOL, LOOKUP_TOOL],
          stream: true,
        }),
        signal: abortRef.current.signal,
      });

      if (!resp.ok) {
        const errText = await resp.text();
        throw new Error(`OpenRouter API error ${resp.status}: ${errText}`);
      }

      const reader = resp.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let assistantContent = '';
      let thinkingContent = '';
      const toolCallPartials = new Map<number, { id: string; name: string; arguments: string }>();

      // Track indices for streaming updates
      let thinkingIndex: number | null = null;
      let assistantIndex: number | null = null;

      while (true) {
        let aborted = false;
        try {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
        } catch (err) {
          if ((err as Error).name === 'AbortError') {
            aborted = true;
          } else {
            throw err;
          }
        }

        if (aborted) {
          // Save partial content to API history for conversation continuity
          if (assistantContent) {
            apiHistoryRef.current.push({ role: 'assistant', content: assistantContent });
          }
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta;
            if (!delta) continue;

            // Handle thinking/reasoning content
            const reasoning = delta.reasoning || delta.reasoning_content;
            if (reasoning) {
              thinkingContent += reasoning;
              if (thinkingIndex === null) {
                // Add a new thinking message
                setDisplayMessages(prev => {
                  thinkingIndex = prev.length;
                  return [...prev, { type: 'thinking', content: thinkingContent }];
                });
              } else {
                const idx = thinkingIndex;
                const content = thinkingContent;
                setDisplayMessages(prev => {
                  const updated = [...prev];
                  updated[idx] = { type: 'thinking', content };
                  return updated;
                });
              }
            }

            // Handle regular content
            if (delta.content) {
              assistantContent += delta.content;
              if (assistantIndex === null) {
                setDisplayMessages(prev => {
                  assistantIndex = prev.length;
                  return [...prev, { type: 'text', role: 'assistant', content: assistantContent }];
                });
              } else {
                const idx = assistantIndex;
                const content = assistantContent;
                setDisplayMessages(prev => {
                  const updated = [...prev];
                  updated[idx] = { type: 'text', role: 'assistant', content };
                  return updated;
                });
              }
            }

            // Handle tool calls
            if (delta.tool_calls) {
              for (const tc of delta.tool_calls) {
                const idx = tc.index ?? 0;
                if (!toolCallPartials.has(idx)) {
                  toolCallPartials.set(idx, { id: tc.id || '', name: '', arguments: '' });
                }
                const partial = toolCallPartials.get(idx)!;
                if (tc.id) partial.id = tc.id;
                if (tc.function?.name) partial.name += tc.function.name;
                if (tc.function?.arguments) partial.arguments += tc.function.arguments;
              }
            }
          } catch {
            // ignore parse errors in stream
          }
        }
      }

      // Build final tool calls
      const toolCalls = Array.from(toolCallPartials.values()).map(p => ({
        id: p.id,
        type: 'function' as const,
        function: { name: p.name, arguments: p.arguments },
      }));

      if (toolCalls.length === 0) {
        // No tool calls — add to API history and we're done
        if (assistantContent) {
          apiHistoryRef.current.push({ role: 'assistant', content: assistantContent });
        }
        return;
      }

      // Add assistant message with tool_calls to API history
      apiHistoryRef.current.push({
        role: 'assistant',
        content: assistantContent || null,
        tool_calls: toolCalls,
      });

      // Execute each tool call and show inline
      for (const tc of toolCalls) {
        let displayQuery = '';
        let result = '';

        if (tc.function.name === 'search_bible') {
          try {
            const args = JSON.parse(tc.function.arguments);
            displayQuery = args.query || '';
          } catch {
            displayQuery = tc.function.arguments;
          }

          setToolStatus(`Searching Bible for: "${displayQuery}"...`);
          result = await executeSearch(displayQuery);
          setToolStatus(null);
        } else if (tc.function.name === 'lookup_passage') {
          let book = '', chapter = 0, verseStart: number | undefined, verseEnd: number | undefined;
          try {
            const args = JSON.parse(tc.function.arguments);
            book = args.book || '';
            chapter = args.chapter || 0;
            verseStart = args.verse_start;
            verseEnd = args.verse_end;
          } catch {
            book = tc.function.arguments;
          }

          displayQuery = verseStart !== undefined
            ? `${book} ${chapter}:${verseStart}${verseEnd && verseEnd !== verseStart ? `-${verseEnd}` : ''}`
            : `${book} ${chapter}`;
          setToolStatus(`Looking up ${displayQuery}...`);
          result = await executeLookup(book, chapter, verseStart, verseEnd);
          setToolStatus(null);
        }

        // Add tool call display message
        setDisplayMessages(prev => [...prev, {
          type: 'tool_call',
          query: displayQuery,
          result,
        }]);

        // Add tool result to API history
        apiHistoryRef.current.push({
          role: 'tool',
          content: result,
          tool_call_id: tc.id,
        });
      }

      // Loop continues to get the model's response after tool results
    }
  };

  const handleNewChat = useCallback(() => {
    if (streaming) {
      if (abortRef.current) abortRef.current.abort();
    }
    setDisplayMessages([]);
    apiHistoryRef.current = [];
    setInput('');
    setToolStatus(null);
    setStreaming(false);
  }, [streaming]);

  if (!apiKey) {
    return (
      <div className="chat-connect">
        <div className="chat-connect-inner">
          <h3>Chat with the Bible</h3>
          <p>Connect to OpenRouter to start an AI-powered Bible study conversation. The assistant uses semantic search to ground its responses in actual scripture from the Berean Standard Bible.</p>
          <p className="chat-connect-note">OpenRouter provides free access via the <code>openrouter/free</code> model.</p>
          <button onClick={onConnect} disabled={connecting} className="connect-button">
            {connecting ? 'Connecting...' : 'Connect to OpenRouter'}
          </button>
        </div>
      </div>
    );
  }

  const selectedModelName = models.find(m => m.id === selectedModel)?.name || selectedModel;

  return (
    <div className="chat-container">
      {showModelPicker && (
        <ModelPickerModal
          models={models}
          selectedModel={selectedModel}
          onSelect={handleModelSelect}
          onClose={() => setShowModelPicker(false)}
        />
      )}
      <div className="chat-header">
        <div className="chat-header-left">
          <button
            className="model-picker-trigger"
            onClick={() => setShowModelPicker(true)}
            disabled={streaming || modelsLoading}
          >
            {modelsLoading ? 'Loading models...' : selectedModelName}
            <span className="model-picker-arrow">&#9662;</span>
          </button>
        </div>
        <div className="chat-header-right">
          {displayMessages.length > 0 && (
            <button onClick={handleNewChat} className="new-chat-button">New Chat</button>
          )}
          <button onClick={onDisconnect} className="disconnect-button">Disconnect</button>
        </div>
      </div>
      <div className="chat-messages">
        {displayMessages.length === 0 && (
          <div className="chat-empty">
            <p>Ask anything about the Bible. The assistant will search scripture to ground its answers.</p>
            <p className="chat-examples">Try: "What does the Bible say about loving your enemies?" or "Tell me about the parable of the prodigal son"</p>
          </div>
        )}
        {displayMessages.map((msg, i) => {
          if (msg.type === 'thinking') {
            return <ThinkingDisplay key={i} content={msg.content} />;
          }
          if (msg.type === 'tool_call') {
            return <ToolCallDisplay key={i} query={msg.query} result={msg.result} />;
          }
          return (
            <div key={i} className={`chat-message chat-message-${msg.role}`}>
              <div className="chat-message-content">{msg.content}</div>
            </div>
          );
        })}
        {toolStatus && (
          <div className="chat-tool-status">{toolStatus}</div>
        )}
        {streaming && !toolStatus && (
          displayMessages.length === 0 ||
          displayMessages[displayMessages.length - 1].type !== 'text' ||
          (displayMessages[displayMessages.length - 1] as TextMessage).role !== 'assistant'
        ) && (
          <div className="chat-streaming-indicator">
            <span className="chat-spinner" />
            <span>Generating response...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <form onSubmit={handleSubmit} className="chat-input-form">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={isLoading ? 'Loading semantic search model...' : 'Ask about the Bible...'}
          disabled={streaming || isLoading}
          className="chat-input"
        />
        {streaming ? (
          <button type="button" onClick={handleStop} className="chat-stop">Stop</button>
        ) : (
          <button type="submit" disabled={isLoading || !input.trim()} className="chat-send">Send</button>
        )}
      </form>
    </div>
  );
}
