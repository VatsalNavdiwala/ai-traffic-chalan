const $ = (id) => document.getElementById(id);

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

async function checkHealth() {
  const el = $("apiStatus");
  try {
    const res = await fetch("/health");
    const data = await res.json();
    el.textContent = `Live · ${data.app} v${data.version}`;
    el.className = "status-pill ok";
  } catch {
    el.textContent = "API offline";
    el.className = "status-pill err";
  }
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

function renderPrimary(v, limit) {
  const box = $("primaryCard");
  if (!v) {
    box.className = "empty";
    box.textContent = "No vehicles detected in sampled frames. Try a clearer road video.";
    return;
  }
  const over = v.max_speed_kmh != null && v.max_speed_kmh > limit;
  const speedClass = over ? "bad" : "ok";
  const img = v.evidence_jpeg_b64
    ? `<img src="data:image/jpeg;base64,${v.evidence_jpeg_b64}" alt="vehicle" />`
    : `<div class="empty">No snapshot</div>`;
  box.className = "primary";
  box.innerHTML = `
    ${img}
    <div class="kv">
      <div><span>Track ID</span><b>#${v.track_id}</b></div>
      <div><span>Type</span><b>${v.vehicle_type}</b></div>
      <div><span>Registration / plate</span><b>${v.plate_number || "NOT READ"}</b></div>
      <div><span>Speed</span><b class="${speedClass}">${v.max_speed_kmh != null ? v.max_speed_kmh + " km/h" : "—"}</b></div>
      <div><span>Limit</span><b>${limit} km/h</b></div>
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
      const over = v.max_speed_kmh != null && v.max_speed_kmh > limit;
      return `<div class="vehicle-row">
        <span class="pill">#${v.track_id} ${v.vehicle_type}</span>
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
    .map(
      (r) => `<div class="vehicle-row">
        <span class="pill bad">${labelViolation(r.violation)}</span>
        <span>${r.plate_number || "UNKNOWN"} · track #${r.track_id}</span>
        <span class="pill">Challan ${r.challan_id}</span>
      </div>`
    )
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
      const img = c.evidence_jpeg_b64
        ? `<img src="data:image/jpeg;base64,${c.evidence_jpeg_b64}" alt="evidence" />`
        : "";
      return `<article class="receipt">
        <h3>Traffic Challan</h3>
        <div class="rid">ID ${c.challan_id} · ${c.status}</div>
        <dl>
          <dt>Registration</dt><dd>${c.registration_number}</dd>
          <dt>Vehicle</dt><dd>${c.vehicle_type}</dd>
          <dt>Violation</dt><dd>${labelViolation(c.violation)}</dd>
          <dt>Location</dt><dd>${c.location}</dd>
          <dt>Speed</dt><dd>${c.speed_kmh != null ? c.speed_kmh + " km/h" : "—"} (limit ${c.speed_limit_kmh})</dd>
          <dt>Time</dt><dd>${new Date(c.occurred_at).toLocaleString()}</dd>
          <dt>Note</dt><dd>${c.officer_note}</dd>
        </dl>
        <div class="fine"><span>Fine amount</span><b>₹${Math.round(c.fine_amount)}</b></div>
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
  btn.textContent = "Analyzing (CPU may take 1–3 min)…";
  $("progress").textContent = "Uploading & running YOLO + tracking + OCR + speed…";

  const fd = new FormData();
  fd.append("video", file);
  fd.append("location", $("location").value || "Ring Road");
  fd.append("speed_limit_kmh", $("speedLimit").value || "60");
  fd.append("max_frames", "60");
  fd.append("run_ocr", $("runOcr").checked ? "true" : "false");

  try {
    const res = await fetch("/demo/analyze", { method: "POST", body: fd });
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error(text.slice(0, 200) || res.statusText);
    }
    if (!res.ok) throw new Error(data.detail || text);

    const limit = data.speed_limit_kmh;
    renderPrimary(data.primary_vehicle, limit);
    renderVehicles(data.vehicles, limit);
    renderViolations(data.violations);
    renderReceipts(data.challans);
    $("progress").textContent = `Done · ${data.frames_processed} frames · ${data.vehicles.length} vehicles · ${data.challans.length} challan(s). ${
      data.notes?.[0] || ""
    }`;
  } catch (err) {
    $("progress").textContent = `Failed: ${err.message}`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze video";
  }
}

$("analyzeBtn").addEventListener("click", analyze);
setupUpload();
checkHealth();
setInterval(() => {
  $("clock").textContent = new Date().toLocaleString();
}, 1000);
$("clock").textContent = new Date().toLocaleString();
