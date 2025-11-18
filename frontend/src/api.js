const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function queryDocument({ question, top_k }) {
  const payload = { question };
  if (top_k) {
    payload.top_k = Number(top_k);
  }

  const response = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to query documents");
  }

  return response.json();
}
