let sessionId = null;
let history = [];

const suggestions = [
  "Who is Marie Curie?",
  "What is quantum entanglement?",
  "History of the internet",
  "How do black holes form?",
  "What is machine learning?"
];

function renderSuggestions() {
  const box = document.getElementById("suggestions");
  box.innerHTML = suggestions
    .map(s => `<span class="sug" onclick="useSuggestion('${s}')">${s}</span>`)
    .join("");
}

function useSuggestion(text) {
  document.getElementById("inp").value = text;
  document.getElementById("suggestions").style.display = "none";
  ask();
}

function addMessage(role, html, meta = "") {
  const msgs = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="avatar ${role}">${role === "bot" ? "W" : "U"}</div>
    <div class="bubble ${role}">
      ${html}
      ${meta ? `<div class="meta">${meta}</div>` : ""}
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function addThinking() {
  return addMessage("bot", `<div class="dots"><span>•</span><span>•</span><span>•</span></div>`);
}

async function ask() {
  const inp = document.getElementById("inp");
  const btn = document.getElementById("btn");
  const question = inp.value.trim();
  if (!question) return;

  document.getElementById("suggestions").style.display = "none";
  inp.value = "";
  btn.disabled = true;

  addMessage("user", question);
  const thinking = addThinking();

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId, history })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    const data = await res.json();
    sessionId = data.session_id;

    history.push({ role: "user", content: question });
    history.push({ role: "assistant", content: data.answer });
    if (history.length > 10) history = history.slice(-10);

    thinking.remove();

    const pills = data.sources
      .map(s => `<a class="pill" href="${s.url}" target="_blank">📄 ${s.title}</a>`)
      .join("");

    const cachedBadge = data.cached
      ? `<span class="cached-badge">⚡ cached</span>` : "";

    const meta = `${data.chunks_used} chunks used ${cachedBadge}`;

    addMessage(
      "bot",
      `${data.answer.replace(/\n/g, "<br>")}<div class="sources">${pills}</div>`,
      meta
    );

  } catch (e) {
    thinking.remove();
    addMessage("bot", `<span style="color:red">⚠️ Error: ${e.message}</span>`);
  }

  btn.disabled = false;
  inp.focus();
}

document.addEventListener("DOMContentLoaded", () => {
  renderSuggestions();
  document.getElementById("inp").addEventListener("keydown", e => {
    if (e.key === "Enter") ask();
  });
});