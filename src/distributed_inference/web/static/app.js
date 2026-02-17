(() => {
  const form = document.getElementById("prompt-form");
  const output = document.getElementById("generated-text");
  const hopsBody = document.getElementById("hop-body");
  const metrics = document.getElementById("metrics");
  const status = document.getElementById("status");
  const runBtn = document.getElementById("run-btn");

  const nodeForm = document.getElementById("node-form");
  const nodeStatus = document.getElementById("node-status");
  const joinBtn = document.getElementById("join-btn");
  const refreshNodesBtn = document.getElementById("refresh-nodes-btn");
  const nodesBody = document.getElementById("nodes-body");
  const cliPreview = document.getElementById("cli-preview");

  let source = null;
  let refreshTimer = null;

  const setStatus = (text) => {
    status.textContent = text;
  };

  const setNodeStatus = (text) => {
    nodeStatus.textContent = text;
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

  const optNumber = (value) => {
    const trimmed = `${value ?? ""}`.trim();
    if (!trimmed) {
      return null;
    }
    const asNum = Number(trimmed);
    return Number.isFinite(asNum) ? asNum : null;
  };

  const buildNodePayload = () => {
    const payload = {
      coordinator: document.getElementById("node_coordinator").value.trim(),
      node_id: document.getElementById("node_id").value.trim() || null,
      device: document.getElementById("node_device").value.trim() || "auto",
      port: optNumber(document.getElementById("node_port").value),
      max_vram_mb: optNumber(document.getElementById("node_max_vram_mb").value),
      bandwidth_mbps: optNumber(document.getElementById("node_bandwidth_mbps").value),
      latency_ms: optNumber(document.getElementById("node_latency_ms").value),
      log_level: "INFO",
    };
    return payload;
  };

  const updateCliPreview = () => {
    const payload = buildNodePayload();
    const cmd = [
      "python -m distributed_inference.cli.manage_nodes join",
      `--web-url ${window.location.origin}`,
    ];
    if (payload.coordinator) {
      cmd.push(`--coordinator ${payload.coordinator}`);
    }
    if (payload.node_id) {
      cmd.push(`--node-id ${payload.node_id}`);
    }
    if (payload.port !== null) {
      cmd.push(`--port ${payload.port}`);
    }
    if (payload.max_vram_mb !== null) {
      cmd.push(`--max-vram-mb ${payload.max_vram_mb}`);
    }
    if (payload.device) {
      cmd.push(`--device ${payload.device}`);
    }
    if (payload.bandwidth_mbps !== null) {
      cmd.push(`--bandwidth-mbps ${payload.bandwidth_mbps}`);
    }
    if (payload.latency_ms !== null) {
      cmd.push(`--latency-ms ${payload.latency_ms}`);
    }
    cliPreview.textContent = cmd.join(" ");
  };

  const renderNodes = (nodes) => {
    nodesBody.innerHTML = "";
    for (const node of nodes) {
      const row = document.createElement("tr");
      const runtimeStatus = node.running ? "running" : `exited(${node.exit_code})`;
      const registration = node.registration_state || "pending";
      const registrationMsg = node.registration_message
        ? `: ${node.registration_message}`
        : "";
      const statusText = `${runtimeStatus}, ${registration}${registrationMsg}`;
      row.innerHTML = `
        <td>${node.node_id}</td>
        <td>${node.port}</td>
        <td>${node.pid}</td>
        <td>${statusText}</td>
        <td>
          <button class="secondary stop-node-btn" data-node-id="${node.node_id}" ${
            node.running ? "" : "disabled"
          }>Stop</button>
        </td>
      `;
      nodesBody.appendChild(row);
    }
  };

  const fetchNodes = async () => {
    try {
      const res = await fetch("/api/nodes");
      const payload = await res.json();
      const nodes = payload.nodes || [];
      renderNodes(nodes);
      setNodeStatus(`${nodes.length} managed node(s)`);
    } catch (err) {
      setNodeStatus(`Node list error: ${err}`);
    }
  };

  const joinNode = async () => {
    const payload = buildNodePayload();
    joinBtn.disabled = true;
    setNodeStatus("Joining node...");
    try {
      const res = await fetch("/api/nodes/join", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail || "join request failed");
      }
      if (body.admitted) {
        setNodeStatus(
          `Joined ${body.node_id} on port ${body.port} (pid ${body.pid})`
        );
      } else {
        setNodeStatus(
          `Node started but registration pending: ${body.node_id}`
        );
      }
      await fetchNodes();
    } catch (err) {
      setNodeStatus(`Join failed: ${err.message || err}`);
    } finally {
      joinBtn.disabled = false;
    }
  };

  const stopNode = async (nodeId) => {
    setNodeStatus(`Stopping ${nodeId}...`);
    try {
      const res = await fetch(`/api/nodes/${encodeURIComponent(nodeId)}/stop`, {
        method: "POST",
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail || "stop failed");
      }
      setNodeStatus(`Stopped ${nodeId}`);
      await fetchNodes();
    } catch (err) {
      setNodeStatus(`Stop failed: ${err.message || err}`);
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

  if (nodeForm) {
    nodeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      await joinNode();
    });

    nodeForm.addEventListener("input", updateCliPreview);
  }

  if (refreshNodesBtn) {
    refreshNodesBtn.addEventListener("click", async () => {
      await fetchNodes();
    });
  }

  if (nodesBody) {
    nodesBody.addEventListener("click", async (e) => {
      const target = e.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      if (!target.classList.contains("stop-node-btn")) {
        return;
      }
      const nodeId = target.getAttribute("data-node-id");
      if (!nodeId) {
        return;
      }
      target.setAttribute("disabled", "true");
      await stopNode(nodeId);
    });
  }

  updateCliPreview();
  fetchNodes();
  refreshTimer = window.setInterval(fetchNodes, 5000);

  window.addEventListener("beforeunload", () => {
    closeSource();
    if (refreshTimer) {
      window.clearInterval(refreshTimer);
      refreshTimer = null;
    }
  });
})();
