(() => {
  const byId = (id) => document.getElementById(id);
  const statusEl = byId("status");
  const nodeStatusEl = byId("node-status");
  const coordinatorInput = byId("coordinator");
  const globalPromptInput = byId("global_prompt");
  const maxTokensInput = byId("max_tokens");
  const temperatureInput = byId("temperature");
  const topPInput = byId("top_p");
  const topKInput = byId("top_k");
  const startAllBtn = byId("start-all-btn");
  const cancelAllBtn = byId("cancel-all-btn");
  const userForm = byId("user-form");
  const userIdInput = byId("user_id_input");
  const userPromptOverrideInput = byId("user_prompt_override");
  const usersConfigList = byId("users-config-list");
  const usersSummary = byId("users-summary");
  const streamsPlaceholder = byId("streams-placeholder");
  const userStreams = byId("user-streams");
  const hopsBody = byId("hop-body");
  const nodeForm = byId("node-form");
  const joinBtn = byId("join-btn");
  const refreshNodesBtn = byId("refresh-nodes-btn");
  const nodesBody = byId("nodes-body");
  const cliPreview = byId("cli-preview");
  const nodeIdInput = byId("node_id");
  const nodePortInput = byId("node_port");
  const nodeDeviceInput = byId("node_device");
  const nodeMaxVramInput = byId("node_max_vram_mb");
  const nodeBandwidthInput = byId("node_bandwidth_mbps");
  const nodeLatencyInput = byId("node_latency_ms");

  if (
    !statusEl ||
    !nodeStatusEl ||
    !coordinatorInput ||
    !globalPromptInput ||
    !maxTokensInput ||
    !temperatureInput ||
    !topPInput ||
    !topKInput ||
    !startAllBtn ||
    !cancelAllBtn ||
    !userForm ||
    !userIdInput ||
    !userPromptOverrideInput ||
    !usersConfigList ||
    !usersSummary ||
    !streamsPlaceholder ||
    !userStreams ||
    !hopsBody ||
    !nodeForm ||
    !joinBtn ||
    !refreshNodesBtn ||
    !nodesBody ||
    !cliPreview ||
    !nodeIdInput ||
    !nodePortInput ||
    !nodeDeviceInput ||
    !nodeMaxVramInput ||
    !nodeBandwidthInput ||
    !nodeLatencyInput
  ) {
    return;
  }

  let refreshTimer = null;
  const users = new Map();
  const runs = new Map();

  const setStatus = (text) => {
    statusEl.textContent = text;
  };

  const setNodeStatus = (text) => {
    nodeStatusEl.textContent = text;
  };

  const safeNum = (value, digits = 1) => {
    const asNum = Number(value);
    if (!Number.isFinite(asNum)) {
      return "-";
    }
    return asNum.toFixed(digits);
  };

  const makeRequestId = (userId = "") => {
    const clean = `${userId}`.replace(/[^a-zA-Z0-9]/g, "").slice(0, 6);
    if (window.crypto && typeof window.crypto.randomUUID === "function") {
      const suffix = window.crypto.randomUUID().replace(/-/g, "").slice(0, 8);
      return clean ? `${clean}-${suffix}` : suffix;
    }
    return `${clean || "req"}-${Date.now().toString(36)}`;
  };

  const closeRunSource = (run) => {
    if (run.source) {
      run.source.close();
      run.source = null;
    }
  };

  const effectivePrompt = (user) => {
    const local = `${user.promptOverride || ""}`.trim();
    if (local) {
      return local;
    }
    return `${globalPromptInput.value || ""}`.trim();
  };

  const updateLayoutSummary = () => {
    const activeCount = [...runs.values()].filter((run) => run.active).length;
    const totalUsers = users.size;
    usersSummary.textContent = `${totalUsers} configured user(s)`;
    streamsPlaceholder.style.display = totalUsers === 0 ? "block" : "none";
    if (activeCount === 0) {
      setStatus(totalUsers > 0 ? `Ready (${totalUsers} users)` : "Idle");
      return;
    }
    setStatus(`Streaming ${activeCount}/${totalUsers} user(s)`);
  };

  const renderRunMetrics = (run) => {
    const m = run.metrics;
    run.metricsEl.innerHTML = `
      <div class="metric-card"><strong>User</strong><br>${run.userId}</div>
      <div class="metric-card"><strong>Lane</strong><br>${m.laneId ?? "-"}</div>
      <div class="metric-card"><strong>Queue Wait</strong><br>${safeNum(m.queueWaitMs, 1)}ms</div>
      <div class="metric-card"><strong>Retries</strong><br>${m.schedulerRetries ?? 0}</div>
      <div class="metric-card"><strong>Tokens</strong><br>${m.tokensGenerated ?? "-"}</div>
      <div class="metric-card"><strong>Latency</strong><br>${safeNum(m.totalLatencyMs, 1)}ms</div>
      <div class="metric-card"><strong>Throughput</strong><br>${safeNum(m.tokensPerSecond, 2)} tok/s</div>
      <div class="metric-card"><strong>Hops</strong><br>${m.hopsSeen ?? 0}</div>
    `;
  };

  const setUserState = (user, label, cssClass) => {
    user.badge.textContent = label;
    user.badge.className = `run-state ${cssClass}`;
  };

  const updateUserPromptPreview = (user) => {
    const prompt = effectivePrompt(user);
    user.promptEl.textContent = prompt || "(empty prompt)";
  };

  const resetUserOutput = (user, prompt, requestId) => {
    user.outputEl.textContent = `Prompt: ${prompt}`;
    user.metaEl.textContent = `Last request: ${requestId}`;
  };

  const appendHopRow = (event, userId) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${event.request_id}</td>
      <td>${userId || "-"}</td>
      <td>${event.step}</td>
      <td>${event.hop_index}</td>
      <td>${event.node_id}</td>
      <td>[${event.start_layer}, ${event.end_layer})</td>
      <td>${safeNum(event.hop_latency_ms, 1)}ms</td>
      <td>${event.lane_id || "-"}</td>
    `;
    hopsBody.appendChild(row);
    row.scrollIntoView({ behavior: "smooth", block: "end" });
  };

  const buildUserConfigRow = (user) => {
    const row = document.createElement("div");
    row.className = "user-config-row";
    row.dataset.userId = user.userId;
    row.innerHTML = `
      <div class="user-config-head">
        <strong>${user.userId}</strong>
        <span class="muted">Override Prompt</span>
      </div>
      <textarea class="user-prompt-edit" rows="2" placeholder="Optional prompt override"></textarea>
      <div class="user-config-actions">
        <button type="button" class="secondary focus-user-btn">Focus</button>
        <button type="button" class="secondary remove-user-btn">Remove</button>
      </div>
    `;
    const promptEdit = row.querySelector(".user-prompt-edit");
    if (promptEdit) {
      promptEdit.value = user.promptOverride;
    }
    return row;
  };

  const buildUserStreamCard = (user) => {
    const card = document.createElement("article");
    card.className = "user-stream-card";
    card.dataset.userId = user.userId;
    card.innerHTML = `
      <div class="user-head">
        <div class="user-title">
          <strong>${user.userId}</strong>
          <span class="muted">Per-user stream</span>
        </div>
        <span class="run-state pending">Idle</span>
      </div>
      <div class="prompt-chip">
        <span class="muted">Prompt</span>
        <div class="prompt-text"></div>
      </div>
      <pre class="text-stream user-output"></pre>
      <div class="user-metrics metrics"></div>
      <div class="user-actions">
        <button type="button" class="start-user-btn">Start</button>
        <button type="button" class="secondary cancel-user-btn">Cancel</button>
        <button type="button" class="secondary clear-user-btn">Clear</button>
        <button type="button" class="secondary remove-user-btn">Remove</button>
      </div>
      <p class="muted user-meta">No requests yet</p>
    `;
    return card;
  };

  const addUser = (userId, promptOverride = "") => {
    const normalized = `${userId}`.trim();
    if (!normalized) {
      setStatus("User ID is required");
      return;
    }
    if (users.has(normalized)) {
      setStatus(`User ${normalized} already exists`);
      return;
    }

    const user = {
      userId: normalized,
      promptOverride: `${promptOverride || ""}`.trim(),
      activeRequestId: null,
      configRow: null,
      card: null,
      badge: null,
      promptEl: null,
      outputEl: null,
      metricsEl: null,
      metaEl: null,
    };

    user.configRow = buildUserConfigRow(user);
    user.card = buildUserStreamCard(user);
    user.badge = user.card.querySelector(".run-state");
    user.promptEl = user.card.querySelector(".prompt-text");
    user.outputEl = user.card.querySelector(".user-output");
    user.metricsEl = user.card.querySelector(".user-metrics");
    user.metaEl = user.card.querySelector(".user-meta");

    users.set(normalized, user);
    usersConfigList.appendChild(user.configRow);
    userStreams.appendChild(user.card);
    updateUserPromptPreview(user);
    user.outputEl.textContent = "Waiting for stream...";
    renderRunMetrics({
      userId: user.userId,
      metricsEl: user.metricsEl,
      metrics: {
        laneId: null,
        queueWaitMs: null,
        schedulerRetries: 0,
        tokensGenerated: null,
        totalLatencyMs: null,
        tokensPerSecond: null,
        hopsSeen: 0,
      },
    });
    setUserState(user, "Idle", "pending");
    updateLayoutSummary();
    setStatus(`Added user ${normalized}`);
  };

  const removeUser = async (userId) => {
    const user = users.get(userId);
    if (!user) {
      return;
    }
    const activeRun =
      user.activeRequestId ? runs.get(user.activeRequestId) : null;
    if (activeRun && activeRun.active) {
      await cancelRun(activeRun.requestId, "user removed from console");
      activeRun.active = false;
      closeRunSource(activeRun);
      runs.delete(activeRun.requestId);
    }

    if (user.configRow?.parentNode) {
      user.configRow.parentNode.removeChild(user.configRow);
    }
    if (user.card?.parentNode) {
      user.card.parentNode.removeChild(user.card);
    }
    users.delete(userId);
    updateLayoutSummary();
  };

  const openStreamForUser = (user, requestId, prompt, params) => {
    const run = {
      requestId,
      userId: user.userId,
      prompt,
      source: null,
      active: true,
      outputEl: user.outputEl,
      metricsEl: user.metricsEl,
      reconnectAttempts: 0,
      maxReconnectAttempts: 30,
      reconnectDelayMs: 2000,
      expectingReconnect: false,
      metrics: {
        laneId: null,
        queueWaitMs: null,
        schedulerRetries: 0,
        tokensGenerated: null,
        totalLatencyMs: null,
        tokensPerSecond: null,
        hopsSeen: 0,
      },
      paramsString: params.toString(),
    };

    runs.set(requestId, run);
    user.activeRequestId = requestId;
    resetUserOutput(user, prompt, requestId);
    renderRunMetrics(run);
    setUserState(user, "Connecting", "pending");
    updateLayoutSummary();

    const finalizeRun = (label, cssClass) => {
      run.active = false;
      closeRunSource(run);
      runs.delete(requestId);
      if (user.activeRequestId === requestId) {
        user.activeRequestId = null;
      }
      setUserState(user, label, cssClass);
      updateLayoutSummary();
    };

    const handleInferenceError = (message) => {
      if (
        message.includes("Model not set up. Call setup_model() first.") &&
        run.reconnectAttempts < run.maxReconnectAttempts
      ) {
        run.reconnectAttempts += 1;
        run.expectingReconnect = true;
        setUserState(
          user,
          `Warming ${run.reconnectAttempts}/${run.maxReconnectAttempts}`,
          "pending",
        );
        closeRunSource(run);
        window.setTimeout(() => {
          if (!runs.has(run.requestId) || !run.active) {
            return;
          }
          run.expectingReconnect = false;
          attachSource();
        }, run.reconnectDelayMs);
        return;
      }
      run.expectingReconnect = false;
      run.outputEl.textContent += `\n\nERROR: ${message}`;
      finalizeRun("Error", "error");
    };

    const attachSource = () => {
      run.expectingReconnect = false;
      const source = new EventSource(`/api/stream?${run.paramsString}`);
      run.source = source;

      source.addEventListener("start", () => {
        setUserState(user, "Streaming", "running");
      });

      source.addEventListener("hop", (eHop) => {
        const event = JSON.parse(eHop.data);
        run.metrics.laneId = event.lane_id || run.metrics.laneId;
        run.metrics.queueWaitMs = event.queue_wait_ms ?? run.metrics.queueWaitMs;
        run.metrics.schedulerRetries =
          event.scheduler_retries ?? run.metrics.schedulerRetries;
        run.metrics.hopsSeen += 1;
        appendHopRow(event, run.userId);
        renderRunMetrics(run);
      });

      source.addEventListener("token", (eToken) => {
        const event = JSON.parse(eToken.data);
        run.metrics.laneId = event.lane_id || run.metrics.laneId;
        run.metrics.queueWaitMs = event.queue_wait_ms ?? run.metrics.queueWaitMs;
        run.metrics.schedulerRetries =
          event.scheduler_retries ?? run.metrics.schedulerRetries;
        run.outputEl.textContent = event.accumulated_text;
        renderRunMetrics(run);
      });

      source.addEventListener("completed", (eDone) => {
        const event = JSON.parse(eDone.data);
        run.outputEl.textContent = event.generated_text;
        run.metrics.laneId = event.lane_id || run.metrics.laneId;
        run.metrics.queueWaitMs = event.queue_wait_ms ?? run.metrics.queueWaitMs;
        run.metrics.schedulerRetries =
          event.scheduler_retries ?? run.metrics.schedulerRetries;
        run.metrics.tokensGenerated = event.tokens_generated;
        run.metrics.totalLatencyMs = event.total_latency_ms;
        run.metrics.tokensPerSecond = event.tokens_per_second;
        renderRunMetrics(run);
        finalizeRun("Completed", "done");
      });

      source.addEventListener("inference_error", (eErr) => {
        if (!run.active) {
          closeRunSource(run);
          return;
        }
        try {
          const event = JSON.parse(eErr.data || "{}");
          handleInferenceError(`${event.message || "inference failed"}`);
        } catch (err) {
          handleInferenceError(`inference failed (${err})`);
        }
      });

      source.onerror = () => {
        if (!run.active) {
          closeRunSource(run);
          return;
        }
        if (run.expectingReconnect) {
          return;
        }
        finalizeRun("Disconnected", "error");
      };
    };

    attachSource();
  };

  const startStreamForUser = (userId) => {
    const user = users.get(userId);
    if (!user) {
      return;
    }
    if (user.activeRequestId && runs.get(user.activeRequestId)?.active) {
      setStatus(`User ${user.userId} already has an active stream`);
      return;
    }

    const prompt = effectivePrompt(user);
    if (!prompt) {
      setStatus(`Prompt is empty for ${user.userId}`);
      return;
    }

    const requestId = makeRequestId(user.userId);
    const params = new URLSearchParams();
    params.set("request_id", requestId);
    params.set("prompt", prompt);
    params.set("coordinator", `${coordinatorInput.value || ""}`.trim());
    params.set("max_tokens", `${maxTokensInput.value || "50"}`);
    params.set("temperature", `${temperatureInput.value || "0.7"}`);
    params.set("top_p", `${topPInput.value || "0.9"}`);
    params.set("top_k", `${topKInput.value || "50"}`);
    params.set("user_id", user.userId);

    openStreamForUser(user, requestId, prompt, params);
  };

  const cancelRun = async (requestId, reason = "cancelled from web console") => {
    const run = runs.get(requestId);
    if (!run) {
      return;
    }
    const user = users.get(run.userId);
    if (!user) {
      closeRunSource(run);
      runs.delete(requestId);
      return;
    }
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(requestId)}/cancel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason }),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.detail || "cancel failed");
      }
      run.outputEl.textContent += `\n\nCANCEL: ${payload.status}`;
      setUserState(user, "Cancelling", "pending");
    } catch (err) {
      run.outputEl.textContent += `\n\nCANCEL ERROR: ${err.message || err}`;
      setUserState(user, "Cancel Error", "error");
    }
  };

  const cancelAllRuns = async () => {
    const activeRuns = [...runs.values()].filter((run) => run.active);
    if (activeRuns.length === 0) {
      setStatus("No active runs");
      return;
    }
    await Promise.all(activeRuns.map((run) => cancelRun(run.requestId)));
  };

  const clearUserOutput = (userId) => {
    const user = users.get(userId);
    if (!user) {
      return;
    }
    user.outputEl.textContent = "Cleared";
    user.metaEl.textContent = "No requests yet";
  };

  const refreshPromptPreviews = () => {
    for (const user of users.values()) {
      updateUserPromptPreview(user);
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
    return {
      coordinator: `${coordinatorInput.value || ""}`.trim(),
      node_id: `${nodeIdInput.value || ""}`.trim() || null,
      device: `${nodeDeviceInput.value || ""}`.trim() || "auto",
      port: optNumber(nodePortInput.value),
      max_vram_mb: optNumber(nodeMaxVramInput.value),
      bandwidth_mbps: optNumber(nodeBandwidthInput.value),
      latency_ms: optNumber(nodeLatencyInput.value),
      log_level: "INFO",
    };
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
      const actionBtn = node.running
        ? `<button class="secondary stop-node-btn" data-node-id="${node.node_id}">Stop</button>`
        : `<button class="secondary remove-node-btn" data-node-id="${node.node_id}">Remove</button>`;
      row.innerHTML = `
        <td>${node.node_id}</td>
        <td>${node.port}</td>
        <td>${node.pid}</td>
        <td>${statusText}</td>
        <td>${actionBtn}</td>
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
        setNodeStatus(`Joined ${body.node_id} on port ${body.port} (pid ${body.pid})`);
      } else {
        setNodeStatus(`Node started but registration pending: ${body.node_id}`);
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

  const removeNode = async (nodeId) => {
    setNodeStatus(`Removing ${nodeId}...`);
    try {
      const res = await fetch(`/api/nodes/${encodeURIComponent(nodeId)}/remove`, {
        method: "POST",
      });
      const body = await res.json();
      if (!res.ok) {
        throw new Error(body.detail || "remove failed");
      }
      setNodeStatus(`Removed ${nodeId}`);
      await fetchNodes();
    } catch (err) {
      setNodeStatus(`Remove failed: ${err.message || err}`);
    }
  };

  userForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const userId = `${userIdInput.value || ""}`.trim();
    const promptOverride = `${userPromptOverrideInput.value || ""}`.trim();
    addUser(userId, promptOverride);
    userForm.reset();
    userIdInput.focus();
  });

  usersConfigList.addEventListener("input", (e) => {
    const target = e.target;
    if (!(target instanceof HTMLTextAreaElement)) {
      return;
    }
    if (!target.classList.contains("user-prompt-edit")) {
      return;
    }
    const row = target.closest(".user-config-row");
    const userId = row?.getAttribute("data-user-id");
    if (!userId) {
      return;
    }
    const user = users.get(userId);
    if (!user) {
      return;
    }
    user.promptOverride = target.value.trim();
    updateUserPromptPreview(user);
  });

  usersConfigList.addEventListener("click", async (e) => {
    const target = e.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const row = target.closest(".user-config-row");
    const userId = row?.getAttribute("data-user-id");
    if (!userId) {
      return;
    }
    if (target.classList.contains("focus-user-btn")) {
      const user = users.get(userId);
      user?.card?.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    if (target.classList.contains("remove-user-btn")) {
      await removeUser(userId);
    }
  });

  userStreams.addEventListener("click", async (e) => {
    const target = e.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const card = target.closest(".user-stream-card");
    const userId = card?.getAttribute("data-user-id");
    if (!userId) {
      return;
    }

    if (target.classList.contains("start-user-btn")) {
      startStreamForUser(userId);
      return;
    }
    if (target.classList.contains("cancel-user-btn")) {
      const user = users.get(userId);
      const requestId = user?.activeRequestId;
      if (requestId) {
        await cancelRun(requestId, `cancelled by ${userId}`);
      }
      return;
    }
    if (target.classList.contains("clear-user-btn")) {
      clearUserOutput(userId);
      return;
    }
    if (target.classList.contains("remove-user-btn")) {
      await removeUser(userId);
    }
  });

  startAllBtn.addEventListener("click", () => {
    if (users.size === 0) {
      setStatus("Add at least one user first");
      return;
    }
    for (const userId of users.keys()) {
      startStreamForUser(userId);
    }
  });

  cancelAllBtn.addEventListener("click", async () => {
    await cancelAllRuns();
  });

  globalPromptInput.addEventListener("input", refreshPromptPreviews);
  coordinatorInput.addEventListener("input", updateCliPreview);

  nodeForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    await joinNode();
  });

  nodeForm.addEventListener("input", updateCliPreview);

  refreshNodesBtn.addEventListener("click", async () => {
    await fetchNodes();
  });

  nodesBody.addEventListener("click", async (e) => {
    const target = e.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const nodeId = target.getAttribute("data-node-id");
    if (!nodeId) {
      return;
    }
    if (
      !target.classList.contains("stop-node-btn") &&
      !target.classList.contains("remove-node-btn")
    ) {
      return;
    }
    target.setAttribute("disabled", "true");
    if (target.classList.contains("stop-node-btn")) {
      await stopNode(nodeId);
    } else {
      await removeNode(nodeId);
    }
  });

  updateCliPreview();
  fetchNodes();
  updateLayoutSummary();
  refreshTimer = window.setInterval(fetchNodes, 5000);

  window.addEventListener("beforeunload", () => {
    for (const run of runs.values()) {
      closeRunSource(run);
    }
    if (refreshTimer) {
      window.clearInterval(refreshTimer);
      refreshTimer = null;
    }
  });
})();
