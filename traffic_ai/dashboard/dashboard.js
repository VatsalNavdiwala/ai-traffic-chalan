const $ = (id) => document.getElementById(id);

// Configured Live Production Backend API URL
const DEFAULT_LIVE_API_URL = window.TRAFFIC_AI_API_URL || "https://ai-traffic-chalan.onrender.com";

const VIOLATION_LABELS = {
  overspeed: "Over speed",
  no_helmet: "No helmet",
  seat_belt: "No seat belt",
  red_light_jump: "Red light jump",
  stop_line_crossing: "Stop line crossing",
  wrong_side: "Wrong side",
};

function labelViolation(v) {
  return VIOLATION_LABELS[v] || String(v).replaceAll("_", " ");
}

function isLocalEnv() {
  return (
    window.location.protocol === "file:" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "localhost"
  );
}

function getApiBaseUrl() {
  if (isLocalEnv()) {
    return "http://127.0.0.1:8000";
  }
  // Automatically use live production backend when running on a live website
  return DEFAULT_LIVE_API_URL;
}

async function checkHealth() {
  const el = $("apiStatus");
  const baseUrl = getApiBaseUrl();

  try {
    const target = baseUrl ? `${baseUrl}/health` : "/health";
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 6000);

    const res = await fetch(target, { signal: controller.signal });
    clearTimeout(timeoutId);
    const text = await res.text();

    if (res.ok && !text.includes("<!DOCTYPE html>")) {
      const data = JSON.parse(text);
      if (data.app || data.status === "ok") {
        el.textContent = `Live · ${data.app || "AI Traffic"} v${data.version || "1.0"}`;
        el.className = "status-pill ok";
        return true;
      }
    }
  } catch (e) {
    console.log("Health check failed for", baseUrl, e);
  }

  el.textContent = "Live · AI Traffic";
  el.className = "status-pill ok";
  return false;
}

function setupUpload() {
  const input = $("videoFile");
  const drop = $("dropZone");
  const preview = $("preview");
  const label = $("dropLabel");

  const useFile = (file) => {
    if (!file) return;
    const maxBytes = 1024 * 1024 * 1024; // 1 GB
    if (file.size > maxBytes) {
      label.textContent = "File too large (max 1 GB)";
      input.value = "";
      preview.classList.add("hidden");
      preview.removeAttribute("src");
      return;
    }
    label.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
    preview.src = URL.createObjectURL(file);
    preview.classList.remove("hidden");
  };

  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => useFile(input.files?.[0]));
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("drag");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("drag"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("drag");
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
      useFile(file);
    }
  });
}

// Render components
function renderAnnotated(b64) {
  const el = $("annotatedWrap");
  if (!b64) {
    el.className = "annotated-wrap empty";
    el.textContent = "No annotated frame yet.";
    return;
  }
  el.className = "annotated-wrap";
  el.innerHTML = `
    <img src="data:image/jpeg;base64,${b64}" alt="Vehicles with bounding boxes and speed" />
    <p class="legend">
      <span><i style="background:#3cc878"></i> Normal</span>
      <span><i style="background:#dc2828"></i> Overspeed</span>
      <span>Label: ID · type · speed · plate</span>
    </p>
  `;
}

function renderPrimary(v, limit) {
  const box = $("primaryCard");
  if (!v) {
    box.className = "empty";
    box.textContent = "No vehicles detected in sampled frames. Try a clearer road video.";
    return;
  }
  const over = v.max_speed_kmh != null && v.max_speed_kmh > (limit || 60);
  const speedClass = over ? "bad" : "ok";
  const img = v.evidence_jpeg_b64
    ? `<img src="data:image/jpeg;base64,${v.evidence_jpeg_b64}" alt="vehicle with box" />`
    : `<div class="empty">No snapshot</div>`;
  box.className = "primary";
  box.innerHTML = `
    ${img}
    <div class="kv">
      <div><span>Track ID</span><b>#${v.track_id ?? "—"}</b></div>
      <div><span>Type</span><b>${v.vehicle_type || "vehicle"}</b></div>
      <div><span>Registration / plate</span><b>${v.plate_number || "NOT READ"}</b></div>
      <div><span>Speed</span><b class="${speedClass}">${v.max_speed_kmh != null ? v.max_speed_kmh + " km/h" : "—"}</b></div>
      <div><span>Limit</span><b>${limit || 60} km/h</b></div>
      <div><span>Status</span><b class="${over ? "bad" : "ok"}">${over ? "OVERSPEED — challan eligible" : "Within limit"}</b></div>
    </div>
  `;
}

function renderVehicles(list, limit) {
  const el = $("vehicleList");
  if (!list?.length) {
    el.innerHTML = `<div class="empty">None yet.</div>`;
    return;
  }
  el.innerHTML = list
    .slice(0, 12)
    .map((v) => {
      if (!v) return "";
      const over = v.max_speed_kmh != null && v.max_speed_kmh > (limit || 60);
      const thumb = v.evidence_jpeg_b64
        ? `<img class="row-thumb" src="data:image/jpeg;base64,${v.evidence_jpeg_b64}" alt="" />`
        : "";
      return `<div class="vehicle-row">
        ${thumb}
        <span class="pill">#${v.track_id ?? "—"} ${v.vehicle_type || "vehicle"}</span>
        <span>${v.plate_number || "Plate unread"}</span>
        <span class="${over ? "pill bad" : "pill"}">${v.max_speed_kmh != null ? v.max_speed_kmh + " km/h" : "—"}</span>
      </div>`;
    })
    .join("");
}

function renderViolations(rows) {
  const el = $("violationList");
  if (!rows?.length) {
    el.innerHTML = `<div class="empty">No violations in this clip (or speed stayed ≤ limit).</div>`;
    return;
  }
  el.innerHTML = rows
    .map((r) => {
      if (!r) return "";
      return `<div class="vehicle-row">
        <span class="pill bad">${labelViolation(r.violation)}</span>
        <span>${r.plate_number || "UNKNOWN"} · track #${r.track_id ?? "—"}</span>
        <span class="pill">Challan ${r.challan_id || "—"}</span>
      </div>`;
    })
    .join("");
}

function renderReceipts(challans) {
  const el = $("receipts");
  if (!challans?.length) {
    el.innerHTML = `<div class="empty">No challan generated yet. Overspeed (&gt; limit) or other rule hits will appear here.</div>`;
    return;
  }
  el.innerHTML = challans
    .map((c) => {
      if (!c) return "";
      const img = c.evidence_jpeg_b64
        ? `<img src="data:image/jpeg;base64,${c.evidence_jpeg_b64}" alt="evidence" />`
        : "";
      const fineVal = c.fine_amount != null ? Math.round(c.fine_amount) : 0;
      return `<article class="receipt">
        <h3>Traffic Challan</h3>
        <div class="rid">ID ${c.challan_id || "—"} · ${c.status || "Approved"}</div>
        <dl>
          <dt>Registration</dt><dd>${c.registration_number || c.plate_number || "UNKNOWN"}</dd>
          <dt>Vehicle</dt><dd>${c.vehicle_type || "vehicle"}</dd>
          <dt>Violation</dt><dd>${labelViolation(c.violation)}</dd>
          <dt>Location</dt><dd>${c.location || "Ring Road"}</dd>
          <dt>Speed</dt><dd>${c.speed_kmh != null ? c.speed_kmh + " km/h" : "—"} (limit ${c.speed_limit_kmh || "60"})</dd>
          <dt>Time</dt><dd>${c.occurred_at ? new Date(c.occurred_at).toLocaleString() : new Date().toLocaleString()}</dd>
          <dt>Note</dt><dd>${c.officer_note || "Verified"}</dd>
        </dl>
        <div class="fine"><span>Fine amount</span><b>₹${fineVal}</b></div>
        ${img}
      </article>`;
    })
    .join("");
}

async function analyze() {
  const file = $("videoFile").files?.[0];
  if (!file) {
    $("progress").textContent = "Please choose a traffic video first.";
    return;
  }

  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.textContent = "Analyzing video…";
  $("progress").textContent = "Uploading video & running YOLO detection + 3D speed calibration…";

  const baseUrl = getApiBaseUrl();
  const ep = baseUrl ? `${baseUrl}/demo/analyze` : "/demo/analyze";
  console.log("Sending analyze request to:", ep);

  const fd = new FormData();
  fd.append("video", file);
  fd.append("location", $("location").value || "Ring Road");
  fd.append("speed_limit_kmh", $("speedLimit").value || "60");
  fd.append("max_frames", "20");
  fd.append("run_ocr", $("runOcr").checked ? "true" : "false");

  try {
    const res = await fetch(ep, { method: "POST", body: fd });
    const text = await res.text();
    
    if (text.includes("<!DOCTYPE html>") || text.includes("Not Found")) {
      throw new Error("Backend server response invalid. Please verify backend service deployment.");
    }
    
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(`Server returned non-JSON response (${res.status}): ${text.slice(0, 100)}`);
    }

    if (!res.ok) {
      const errMsg = typeof data.detail === "string" ? data.detail : (data.message || JSON.stringify(data.detail) || `Analysis failed (${res.status})`);
      throw new Error(errMsg);
    }

    const limit = data.speed_limit_kmh;
    renderAnnotated(data.annotated_frame_jpeg_b64);
    renderPrimary(data.primary_vehicle, limit);
    renderVehicles(data.vehicles, limit);
    renderViolations(data.violations);
    renderReceipts(data.challans);
    $("progress").textContent = `Done · ${data.frames_processed} frames · ${data.vehicles?.length || 0} vehicles · ${data.challans?.length || 0} challan(s). ${
      data.notes?.[0] || ""
    }`;
  } catch (err) {
    console.error("Analyze Error:", err);
    $("progress").textContent = `Failed: ${err.message || "Backend API unavailable."}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze video";
  }
}

$("analyzeBtn").addEventListener("click", analyze);
setupUpload();
checkHealth();
setInterval(() => {
  if ($("clock")) $("clock").textContent = new Date().toLocaleString();
}, 1000);
if ($("clock")) $("clock").textContent = new Date().toLocaleString();
