const API_BASE = window.API_BASE || "http://localhost:8000";

const statusEl = document.getElementById("status");
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const samplesEl = document.getElementById("samples");
const previewWrap = document.getElementById("preview-wrap");
const previewImg = document.getElementById("preview");
const resultEl = document.getElementById("result");
const resultText = document.getElementById("result-text");
const chartEl = document.getElementById("chart");
const errorEl = document.getElementById("error");

let classNames = [];

async function init() {
  try {
    const [health, classesRes] = await Promise.all([
      fetch(`${API_BASE}/api/health`).then((r) => r.json()),
      fetch(`${API_BASE}/api/classes`).then((r) => r.json()),
    ]);
    classNames = classesRes.classes;
    statusEl.textContent = `Backend connected (checkpoint epoch ${health.checkpoint_epoch}, val_acc ${(health.val_acc * 100).toFixed(1)}%)`;
    statusEl.className = "status ok";
    renderSamples();
  } catch (err) {
    statusEl.textContent = `Can't reach the backend at ${API_BASE} — is it running?`;
    statusEl.className = "status error";
  }
}

function renderSamples() {
  samplesEl.innerHTML = "";
  for (const name of classNames) {
    const card = document.createElement("div");
    card.className = "sample";
    card.draggable = true;

    const img = document.createElement("img");
    img.src = `samples/${name}.png`;
    img.alt = name;
    img.draggable = false;

    const label = document.createElement("span");
    label.textContent = name;

    card.append(img, label);
    card.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", name);
    });
    card.addEventListener("click", () => loadSample(name));

    samplesEl.appendChild(card);
  }
}

async function loadSample(name) {
  const blob = await fetch(`samples/${name}.png`).then((r) => r.blob());
  predict(blob, `${name}.png`);
}

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    predict(fileInput.files[0], fileInput.files[0].name);
  }
});

["dragenter", "dragover"].forEach((evt) => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
});

["dragleave", "dragend"].forEach((evt) => {
  dropZone.addEventListener(evt, () => {
    dropZone.classList.remove("drag-over");
  });
});

dropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");

  if (e.dataTransfer.files.length > 0) {
    const file = e.dataTransfer.files[0];
    predict(file, file.name);
    return;
  }

  const sampleName = e.dataTransfer.getData("text/plain");
  if (sampleName) {
    loadSample(sampleName);
  }
});

async function predict(blob, filename) {
  errorEl.hidden = true;

  previewImg.src = URL.createObjectURL(blob);
  previewWrap.hidden = false;

  const formData = new FormData();
  formData.append("file", blob, filename);

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => null);
      throw new Error(data?.detail || "Prediction failed — please try another image.");
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    resultEl.hidden = true;
    errorEl.textContent = err.message || "Prediction failed — please try another image.";
    errorEl.hidden = false;
  }
}

function renderResult(data) {
  const pct = (data.confidence * 100).toFixed(1);
  const label = data.is_uncertain
    ? `<span class="uncertain">Uncertain</span> — best guess: ${data.predicted_class} (${pct}%)`
    : `Predicted: <span class="confident">${data.predicted_class}</span> (${pct}%)`;
  resultText.innerHTML = label;

  chartEl.innerHTML = "";
  for (const name of classNames) {
    const prob = data.probabilities[name];
    const isTop = name === data.predicted_class;

    const col = document.createElement("div");
    col.className = "bar-col" + (isTop ? " top" + (data.is_uncertain ? " uncertain" : "") : "");

    if (isTop) {
      const value = document.createElement("span");
      value.className = "bar-value";
      value.textContent = `${(prob * 100).toFixed(1)}%`;
      col.appendChild(value);
    }

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${Math.max(prob * 100, 1)}%`;
    bar.title = `${name}: ${(prob * 100).toFixed(1)}%`;
    col.appendChild(bar);

    const barLabel = document.createElement("span");
    barLabel.className = "bar-label";
    barLabel.textContent = name;
    col.appendChild(barLabel);

    chartEl.appendChild(col);
  }

  resultEl.hidden = false;
}

init();
