(() => {
  const form = document.getElementById("prompt-form");
  const output = document.getElementById("generated-text");
  const hopsBody = document.getElementById("hop-body");
  const metrics = document.getElementById("metrics");
  const status = document.getElementById("status");
  const runBtn = document.getElementById("run-btn");

  let source = null;

  const setStatus = (text) => {
    status.textContent = text;
  };

  const clearRun = () => {
    output.textContent = "";
    hopsBody.innerHTML = "";
    metrics.innerHTML = "";
  };

  const addHopRow = (event) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${event.step}</td>
      <td>${event.hop_index}</td>
      <td>${event.node_id}</td>
      <td>[${event.start_layer}, ${event.end_layer})</td>
      <td>${event.hop_latency_ms.toFixed(1)}ms</td>
    `;
    hopsBody.appendChild(row);
    row.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  const renderMetrics = (event) => {
    metrics.innerHTML = `
      <div class="metric-card"><strong>Tokens</strong><br>${event.tokens_generated}</div>
      <div class="metric-card"><strong>Latency</strong><br>${event.total_latency_ms.toFixed(1)}ms</div>
      <div class="metric-card"><strong>Throughput</strong><br>${event.tokens_per_second.toFixed(2)} tok/s</div>
    `;
  };

  const closeSource = () => {
    if (source) {
      source.close();
      source = null;
    }
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    closeSource();
    clearRun();

    const params = new URLSearchParams(new FormData(form));
    setStatus("Connecting...");
    runBtn.disabled = true;

    source = new EventSource(`/api/stream?${params.toString()}`);

    source.addEventListener("start", (eStart) => {
      const event = JSON.parse(eStart.data);
      setStatus(`Streaming ${event.request_id}`);
    });

    source.addEventListener("hop", (eHop) => {
      const event = JSON.parse(eHop.data);
      addHopRow(event);
    });

    source.addEventListener("token", (eToken) => {
      const event = JSON.parse(eToken.data);
      output.textContent = event.accumulated_text;
    });

    source.addEventListener("completed", (eDone) => {
      const event = JSON.parse(eDone.data);
      output.textContent = event.generated_text;
      renderMetrics(event);
      setStatus("Completed");
      runBtn.disabled = false;
      closeSource();
    });

    source.addEventListener("error", (eErr) => {
      if (!eErr.data) {
        setStatus("Disconnected");
      } else {
        const event = JSON.parse(eErr.data);
        setStatus(`Error: ${event.message}`);
      }
      runBtn.disabled = false;
      closeSource();
    });
  });
})();
