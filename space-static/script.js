import { pipeline, cos_sim } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0";

const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const btnLabel = askBtn.querySelector(".btn-label");
const spinner = askBtn.querySelector(".spinner");
const result = document.getElementById("result");
const sourcesList = document.getElementById("sources-list");
const errorEl = document.getElementById("error");
const statusBadge = document.getElementById("status-badge");

let embedder = null;
let index = []; // [{page, modality, text, image, embedding}]

function setLoading(isLoading) {
  askBtn.disabled = isLoading;
  spinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Searching" : "Search";
}

function enableUI() {
  input.disabled = false;
  askBtn.disabled = false;
  document.querySelectorAll(".chip").forEach((c) => (c.disabled = false));
}

async function init() {
  try {
    const [model, data] = await Promise.all([
      pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2"),
      fetch("data/index.json").then((r) => {
        if (!r.ok) throw new Error("index.json not found");
        return r.json();
      }),
    ]);
    embedder = model;
    index = data;

    statusBadge.textContent = `index ready · ${index.length} chunks`;
    statusBadge.classList.add("ok");
    enableUI();
  } catch (err) {
    statusBadge.textContent = "failed to load index";
    statusBadge.classList.add("bad");
    errorEl.textContent =
      "Could not load the model or data/index.json. Make sure data/index.json and images/ were copied in (see README.md).";
    errorEl.hidden = false;
  }
}

async function embedQuery(text) {
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

function renderResults(hits) {
  sourcesList.innerHTML = "";
  hits.forEach((h) => {
    const item = document.createElement("div");
    item.className = "source-item";

    if (h.image) {
      const img = document.createElement("img");
      img.src = `images/${h.image}`;
      img.alt = `Page ${h.page} visual`;
      item.appendChild(img);
    }

    const body = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.innerHTML = `<span class="source-page">PAGE ${h.page}</span><span class="source-modality">${h.modality}</span><span class="source-score">${(h.score * 100).toFixed(0)}% match</span>`;

    const snippet = document.createElement("p");
    snippet.className = "source-snippet";
    snippet.textContent = h.text.length > 400 ? h.text.slice(0, 400) + "…" : h.text;

    body.appendChild(meta);
    body.appendChild(snippet);
    item.appendChild(body);
    sourcesList.appendChild(item);
  });
}

async function ask(question) {
  errorEl.hidden = true;
  setLoading(true);
  try {
    const qEmb = await embedQuery(question);
    const scored = index.map((chunk) => ({
      ...chunk,
      score: cos_sim(qEmb, chunk.embedding),
    }));
    scored.sort((a, b) => b.score - a.score);
    renderResults(scored.slice(0, 5));
    result.hidden = false;
  } catch (err) {
    errorEl.textContent = "Search failed — check the console for details.";
    errorEl.hidden = false;
    console.error(err);
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  ask(q);
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    const q = chip.dataset.q;
    input.value = q;
    ask(q);
  });
});

init();
