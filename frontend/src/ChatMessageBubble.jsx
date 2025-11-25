import React from "react";

function ChatMessageBubble({ role, content, chunks = [], isLoading = false }) {
  const isUser = role === "user";

  // Collect all unique image URLs from chunks
  const allImageUrls = [];
  if (chunks && Array.isArray(chunks)) {
    chunks.forEach((chunk) => {
      if (chunk.image_urls && Array.isArray(chunk.image_urls)) {
        chunk.image_urls.forEach((url) => {
          if (url && !allImageUrls.includes(url)) {
            allImageUrls.push(url);
          }
        });
      }
    });
  }

  // Build full image URLs (prepend API base URL if relative)
  const getImageUrl = (url) => {
    if (!url) return null;
    if (url.startsWith("http")) return url;
    // Relative URL - prepend API base URL
    const apiBase = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
    return `${apiBase}${url}`;
  };

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
          <>
            <div className="message-content">{content}</div>
            {!isUser && allImageUrls.length > 0 && (
              <div className="message-images">
                <div className="images-label">Related images:</div>
                <div className="images-grid">
                  {allImageUrls.map((url, index) => {
                    const fullUrl = getImageUrl(url);
                    return (
                      <div key={index} className="image-wrapper">
                        <img
                          src={fullUrl}
                          alt={`Related image ${index + 1}`}
                          className="related-image"
                          loading="lazy"
                          onError={(e) => {
                            e.target.style.display = "none";
                            e.target.parentElement.innerHTML = `<div style="padding: 1rem; text-align: center; color: #9ca3af; font-size: 0.875rem;">Image failed to load</div>`;
                          }}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default ChatMessageBubble;

