/**
 * AI Chat Assistant Component.
 * Connects to POST /api/ai/chat — retrieves explanations from Groq via backend.
 * No hardcoded replies. Shows unavailable state when Groq is not configured.
 */
import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { ChatMessage, PatientContext } from '../../types';
import { chatWithAI } from '../../api/endpoints';
import { extractErrorMessage } from '../../api/client';
import { Spinner, ErrorBox, LLMTag, Disclaimer } from '../shared';

interface AIChatProps {
  patientContext: PatientContext;
  llmConfigured: boolean;
}

const SUGGESTED_QUESTIONS = [
  'Explain the current sepsis prediction.',
  'Which features most influenced the model?',
  'Which organ predictions are available?',
  'Explain the difference between sepsis risk and antibiotic susceptibility.',
  'What data is missing from this prediction?',
  'Summarize all available research prediction results.',
];

export function AIChat({ patientContext, llmConfigured }: AIChatProps) {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView?.({ behavior: 'smooth' });
  }, [history, loading]);

  const sendMessage = useCallback(async (messageText: string) => {
    const trimmed = messageText.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: 'user', content: trimmed };
    setHistory(prev => [...prev, userMsg]);
    setInput('');
    setError(null);
    setLoading(true);

    try {
      // Build de-identified context from actual model results only
      const context: Record<string, unknown> = {};
      if (patientContext.sepsis_result) context.sepsis_result = patientContext.sepsis_result;
      if (patientContext.organ_results?.length) context.organ_results = patientContext.organ_results;
      if (patientContext.antibiotic_result) context.antibiotic_result = patientContext.antibiotic_result;

      const resp = await chatWithAI({
        patient_context: context,
        message: trimmed,
        history: history.slice(-10),
      });

      setHistory(prev => [...prev, { role: 'assistant', content: resp.answer }]);
    } catch (err) {
      setError(extractErrorMessage(err));
      // Remove the user message from history on error so they can retry
      setHistory(prev => prev.filter(m => m !== userMsg));
      setInput(trimmed);
    } finally {
      setLoading(false);
    }
  }, [loading, history, patientContext]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  if (!llmConfigured) {
    return (
      <div className="error-box" role="alert" style={{ marginTop: '1rem' }}>
        <strong>Groq LLM not configured.</strong> Set <code>GROQ_API_KEY</code> in{' '}
        <code>backend/.env</code> to enable the AI assistant.
        ML predictions above remain fully functional.
      </div>
    );
  }

  return (
    <div>
      <Disclaimer text="AI assistant uses Groq cloud. Only de-identified model outputs are sent — no patient identifiers. LLM explanations are not clinical advice." />

      {history.length === 0 && (
        <div style={{ marginBottom: '1rem' }}>
          <p style={{ fontSize: '0.82rem', color: 'var(--color-text-muted)', marginBottom: '0.5rem' }}>
            Suggested questions:
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
            {SUGGESTED_QUESTIONS.map(q => (
              <button
                key={q}
                className="btn btn-outline btn-sm"
                onClick={() => sendMessage(q)}
                disabled={loading}
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="chat-container">
        <div className="chat-messages" aria-live="polite">
          {history.map((msg, i) => (
            <div key={i} className={`chat-message ${msg.role}`}>
              <div className="chat-bubble">
                {msg.role === 'assistant' && (
                  <div style={{ marginBottom: '0.35rem' }}>
                    <LLMTag />
                  </div>
                )}
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="chat-message assistant">
              <div className="chat-bubble" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Spinner size={16} /> Thinking…
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {error && <ErrorBox message={error} onRetry={() => sendMessage(input)} />}

        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            className="form-textarea"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the prediction… (Enter to send, Shift+Enter for new line)"
            rows={2}
            maxLength={1000}
            disabled={loading}
            aria-label="Chat input"
          />
          <button
            className="btn btn-primary"
            onClick={() => sendMessage(input)}
            disabled={loading || !input.trim()}
            aria-label="Send message"
          >
            {loading ? <Spinner size={16} /> : 'Send'}
          </button>
        </div>
        <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.35rem' }}>
          {input.length}/1000 characters
        </p>
      </div>
    </div>
  );
}
