document.getElementById("detectButton").addEventListener("click", async () => {
  const text = document.getElementById("articleInput").value.trim();
  const unverified = document.getElementById("unverifiedSource").checked;

body: JSON.stringify({ text, unverified })


  if (!text) {
    alert("Please paste an article.");
    return;
  }

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, unverified })
  });

  const data = await res.json();

  document.getElementById("result").classList.remove("hidden");

  // Label
  document.getElementById("label").innerHTML =
    `Prediction: <span class="${data.label.toLowerCase()}">${data.label}</span>`;

  // Confidence text
  document.getElementById("confidenceText").innerText =
    `Confidence Score: ${data.confidence}%`;

  // Confidence bar
  const fill = document.getElementById("confidenceFill");
  fill.style.width = data.confidence + "%";
  fill.className = "confidence-fill";

  if (data.confidence >= 75) fill.classList.add("confidence-high");
  else if (data.confidence >= 50) fill.classList.add("confidence-medium");
  else fill.classList.add("confidence-low");

  // Reason
  document.getElementById("reason").innerHTML =
    `<strong>Why this result?</strong><br>${data.reason}`;

  // Highlight keywords in article
  let highlighted = text;
  data.keywords.forEach(word => {
    const regex = new RegExp(`\\b(${word})\\b`, "gi");
    highlighted = highlighted.replace(regex, `<mark>$1</mark>`);
  });

  document.getElementById("highlightedText").innerHTML = highlighted;

  // Keyword chips
  document.getElementById("keywords").innerHTML =
    "Top Influencing Keywords: " +
    data.keywords.map(k => `<span class="chip">${k}</span>`).join(" ");
});
