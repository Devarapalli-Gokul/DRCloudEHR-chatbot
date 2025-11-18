import React, { useState, useRef, useEffect } from "react";
import { queryDocument } from "./api.js";
import ChatMessageBubble from "./ChatMessageBubble.jsx";

function App() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      content: "Hi! Ask me questions about your documents and I'll answer based on what's been ingested.",
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [topK, setTopK] = useState(5);
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    const question = inputValue.trim();
    if (!question || isLoading) return;

    // Add user message
    const userMessage = {
      id: Date.now().toString(),
      role: "user",
      content: question,
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
    setIsLoading(true);

    // Add temporary loading message
    const loadingMessageId = `loading-${Date.now()}`;
    setMessages((prev) => [
      ...prev,
      {
        id: loadingMessageId,
        role: "assistant",
        content: "",
        isLoading: true,
      },
    ]);

    try {
      const result = await queryDocument({
        question,
        top_k: topK,
      });

      // Replace loading message with actual answer
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessageId
            ? {
                id: loadingMessageId,
                role: "assistant",
                content: result.answer_text,
              }
            : msg
        )
      );
    } catch (error) {
      console.error("Query error:", error);
      // Replace loading message with error message
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessageId
            ? {
                id: loadingMessageId,
                role: "assistant",
                content: "Sorry, I couldn't answer that due to an error. Please try again.",
              }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  };

  const handleClearChat = () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "Hi! Ask me questions about your documents and I'll answer based on what's been ingested.",
      },
    ]);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h1>Document Q&A System</h1>
        <div className="header-controls">
          <div className="top-k-control">
            <label htmlFor="top-k-input">Top K:</label>
            <input
              id="top-k-input"
              type="number"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="top-k-input"
            />
          </div>
          <button onClick={handleClearChat} className="clear-button">
            Clear chat
          </button>
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <ChatMessageBubble
            key={message.id}
            role={message.role}
            content={message.content}
            isLoading={message.isLoading}
          />
        ))}
        <div ref={chatEndRef} />
      </div>

      <div className="chat-input-container">
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows="1"
            className="chat-input"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !inputValue.trim()}
            className="send-button"
          >
            {isLoading ? (
              <span className="spinner"></span>
            ) : (
              <svg
                width="20"
                height="20"
                viewBox="0 0 20 20"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M18 2L9 11M18 2L12 18L9 11M18 2L2 8L9 11"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
