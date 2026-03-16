import { useState, useEffect, useCallback, useRef } from 'react';

const STORAGE_KEY = 'openrouter_api_key';

export function useOpenRouter() {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    // Check URL for OAuth callback code
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    if (code) {
      setConnecting(true);
      const codeVerifier = sessionStorage.getItem('openrouter_code_verifier');
      sessionStorage.removeItem('openrouter_code_verifier');

      fetch('https://openrouter.ai/api/v1/auth/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, code_verifier: codeVerifier, code_challenge_method: 'S256' }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.key) {
            localStorage.setItem(STORAGE_KEY, data.key);
            setApiKey(data.key);
          }
        })
        .catch(console.error)
        .finally(() => {
          // Clean URL
          window.history.replaceState({}, '', window.location.pathname);
          setConnecting(false);
        });
    } else {
      // Check localStorage for existing key
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) setApiKey(stored);
    }
  }, []);

  const connect = useCallback(async () => {
    // Generate code_verifier (random 43-128 char string)
    const buf = new Uint8Array(32);
    crypto.getRandomValues(buf);
    const codeVerifier = Array.from(buf, b => b.toString(16).padStart(2, '0')).join('');
    sessionStorage.setItem('openrouter_code_verifier', codeVerifier);

    // Compute SHA-256 code_challenge
    const encoded = new TextEncoder().encode(codeVerifier);
    const hashBuf = await crypto.subtle.digest('SHA-256', encoded);
    const challenge = btoa(String.fromCharCode(...new Uint8Array(hashBuf)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');

    const callbackUrl = window.location.origin + window.location.pathname;
    window.location.href = `https://openrouter.ai/auth?callback_url=${encodeURIComponent(callbackUrl)}&code_challenge=${challenge}&code_challenge_method=S256`;
  }, []);

  const disconnect = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setApiKey(null);
  }, []);

  return { apiKey, connecting, connect, disconnect };
}
