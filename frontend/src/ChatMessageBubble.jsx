import React from "react";

function ChatMessageBubble({ role, content, isLoading = false }) {
  const isUser = role === "user";

  return (
    <div className={`chat-message ${isUser ? "user-message" : "assistant-message"}`}>
      <div className="message-label">{isUser ? "You" : "Assistant"}</div>
      <div className="message-bubble">
        {isLoading ? (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        ) : (
          <div className="message-content">{content}</div>
        )}
      </div>
    </div>
  );
}

export default ChatMessageBubble;

