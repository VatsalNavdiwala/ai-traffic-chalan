const $ = (id) => document.getElementById(id);

const DIR_IDS = {
  north: { count: "nCount", green: "nGreen" },
  south: { count: "sCount", green: "sGreen" },
  east: { count: "eCount", green: "eGreen" },
  west: { count: "wCount", green: "wGreen" },
};

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

function payload() {
  const emergency = $("emergency").value || null;
  return {
    north: Number($("north").value) || 0,
    south: Number($("south").value) || 0,
    east: Number($("east").value) || 0,
    west: Number($("west").value) || 0,
    rush_hour: $("rushHour").checked,
    emergency_direction: emergency,
  };
}

function renderDecision(data) {
  const maxGreen = Math.max(...data.phases.map((p) => p.green_seconds), 1);
  const list = $("timingList");
  list.innerHTML = "";

  let best = null;
  for (const p of data.phases) {
    if (!best || p.green_seconds > best.green_seconds) best = p;
  }

  document.querySelectorAll(".arm").forEach((arm) => {
    arm.classList.remove("active", "amber");
  });

  for (const p of data.phases) {
    const ids = DIR_IDS[p.direction];
    if (!ids) continue;
    $(ids.count).textContent = p.waiting_vehicles;
    $(ids.green).textContent = p.green_seconds;

    const arm = document.querySelector(`.arm.${p.direction}`);
    if (arm) {
      if (data.emergency_override && p.direction === data.emergency_direction) {
        arm.classList.add("active");
      } else if (!data.emergency_override && best && p.direction === best.direction) {
        arm.classList.add("active");
      } else if (p.green_seconds >= 20) {
        arm.classList.add("amber");
      }
    }

    const li = document.createElement("li");
    const pct = Math.round((p.green_seconds / maxGreen) * 100);
    li.innerHTML = `
      <span>${p.direction.toUpperCase()}</span>
      <div class="bar"><span style="width:${pct}%"></span></div>
      <span>${p.green_seconds}s</span>
    `;
    list.appendChild(li);
  }

  const badge = $("overrideBadge");
  if (data.emergency_override) {
    badge.classList.remove("hidden");
    badge.textContent = `EMERGENCY → ${String(data.emergency_direction || "").toUpperCase()}`;
  } else {
    badge.classList.add("hidden");
  }
}

async function runSignal() {
  const btn = $("runBtn");
  btn.disabled = true;
  btn.textContent = "Computing…";
  try {
    const res = await fetch("/signal/decide", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload()),
    });
    if (!res.ok) throw new Error(await res.text());
    renderDecision(await res.json());
  } catch (err) {
    $("timingList").innerHTML = `<li style="color:#ff8a7a">Error: ${err.message}</li>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Signal AI";
  }
}

async function previewChallan() {
  const body = {
    plate_number: $("plate").value.trim() || "UNKNOWN",
    violation_type: $("violation").value,
    location: $("location").value.trim() || "Unknown",
    fine_amount: Number($("fine").value) || 1000,
  };
  $("smsOut").textContent = "Generating…";
  try {
    const res = await fetch("/challans/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    $("smsOut").textContent = data.sms || JSON.stringify(data, null, 2);
  } catch (err) {
    $("smsOut").textContent = `Error: ${err.message}`;
  }
}

function loadDemo() {
  $("north").value = 45;
  $("south").value = 10;
  $("east").value = 6;
  $("west").value = 18;
  $("rushHour").checked = false;
  $("emergency").value = "";
  runSignal();
}

function tickClock() {
  $("clock").textContent = new Date().toLocaleString();
}

$("runBtn").addEventListener("click", runSignal);
$("demoBtn").addEventListener("click", loadDemo);
$("challanBtn").addEventListener("click", previewChallan);

checkHealth();
tickClock();
setInterval(tickClock, 1000);
loadDemo();
