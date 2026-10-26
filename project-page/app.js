const demos = {
  alert: {
    tab: "tab-alert",
    task: "T2 · EVENT NOTIFICATION",
    title: "Wait for the event.",
    prompt: "“Tell me when the crossing signal turns green.”",
    art: "art-alert",
    moments: [
      { time: "00:03", cue: "The signal is red", status: "STAY SILENT", response: "The requested event has not happened. The assistant keeps watching.", active: false },
      { time: "00:08", cue: "A pedestrian approaches", status: "STAY SILENT", response: "The scene has changed, but the signal is still red.", active: false },
      { time: "00:12", cue: "The signal turns green", status: "RESPOND NOW", response: "“The crossing signal just turned green.”", active: true },
    ],
  },
  continuous: {
    tab: "tab-continuous",
    task: "T7 · CONTINUOUS RESPONSES",
    title: "Keep pace with change.",
    prompt: "“Describe the key steps as this demonstration unfolds.”",
    art: "art-continuous",
    moments: [
      { time: "00:06", cue: "Parts are laid out", status: "FIRST UPDATE", response: "“The presenter has arranged the parts on the table.”", active: true },
      { time: "00:19", cue: "Pieces are joined", status: "NEXT UPDATE", response: "“The two main pieces are being fitted together.”", active: true },
      { time: "00:34", cue: "The assembly is tested", status: "NEXT UPDATE", response: "“The assembled device is now being tested.”", active: true },
    ],
  },
  assistance: {
    tab: "tab-assistance",
    task: "T8 · GUIDED ASSISTANCE",
    title: "Guide the next step.",
    prompt: "“Help me pack these items as I go.”",
    art: "art-assistance",
    moments: [
      { time: "00:04", cue: "The bag is open", status: "GUIDE STEP 1", response: "“Place the larger items at the bottom of the bag.”", active: true },
      { time: "00:17", cue: "Large items are packed", status: "GUIDE STEP 2", response: "“Add the smaller items on top so they remain easy to reach.”", active: true },
      { time: "00:31", cue: "The bag is nearly full", status: "GUIDE STEP 3", response: "“Check that nothing is caught before closing the bag.”", active: true },
    ],
  },
};

const demoTabs = [...document.querySelectorAll(".demo-tab")];
const momentButtons = document.getElementById("moment-buttons");
let currentDemo = "alert";

function renderMoment(index) {
  const demo = demos[currentDemo];
  const moment = demo.moments[index];
  document.getElementById("demo-time").textContent = moment.time;
  document.getElementById("scene-caption").textContent = moment.cue;
  document.getElementById("moment-counter").textContent = `${String(index + 1).padStart(2, "0")} / 03`;
  document.getElementById("response-status").textContent = moment.status;
  document.getElementById("response-status").className = moment.active ? "status-respond" : "status-wait";
  document.getElementById("demo-response").textContent = moment.response;
  document.getElementById("stage-scrub-fill").style.width = `${((index + 1) / demo.moments.length) * 100}%`;
  document.getElementById("demo-art").dataset.moment = String(index);
  [...momentButtons.children].forEach((button, buttonIndex) => {
    button.classList.toggle("is-active", buttonIndex === index);
    button.setAttribute("aria-pressed", String(buttonIndex === index));
  });
}

function renderDemo(key) {
  currentDemo = key;
  const demo = demos[key];
  demoTabs.forEach((tab) => {
    const selected = tab.dataset.demo === key;
    tab.classList.toggle("is-active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.getElementById("demo-panel").setAttribute("aria-labelledby", demo.tab);
  document.getElementById("demo-task").textContent = demo.task;
  document.getElementById("demo-title").textContent = demo.title;
  document.getElementById("demo-prompt").textContent = demo.prompt;
  document.getElementById("demo-art").className = `stage-art ${demo.art}`;
  momentButtons.replaceChildren();
  demo.moments.forEach((moment, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "moment-button";
    button.innerHTML = `<strong>${moment.time}</strong><span>${moment.cue}</span>`;
    button.addEventListener("click", () => renderMoment(index));
    momentButtons.append(button);
  });
  renderMoment(0);
}

demoTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => renderDemo(tab.dataset.demo));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    let next = index;
    if (event.key === "ArrowLeft") next = (index + demoTabs.length - 1) % demoTabs.length;
    if (event.key === "ArrowRight") next = (index + 1) % demoTabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = demoTabs.length - 1;
    renderDemo(demoTabs[next].dataset.demo);
    demoTabs[next].focus();
  });
});

const resultSets = {
  explicit: {
    tab: "result-explicit-tab",
    overline: "JOINT CONTENT + TIMING ACCURACY",
    heading: "Explicit-request tasks",
    description: "Avg. gives equal weight to T1–T4, T6 and T7. T5 is reported separately in the full leaderboard.",
    axis: ["0", "5", "10", "15", "20", "25%"],
    max: 25,
    rows: [
      { group: "Proprietary models", model: "Gemini-3.5-Flash", score: 20.2, color: "teal" },
      { group: "Open-source models", model: "Qwen3-Omni", score: 6.7, color: "violet" },
      { group: "Streaming Models", model: "MOSS-VL", score: 5.4, color: "coral" },
    ],
  },
  assistance: {
    tab: "result-assistance-tab",
    overline: "TIMELY REFERENCE-TARGET RECALL",
    heading: "Assistance tasks",
    description: "Overall pools successful matches across all 360 T8 targets, including reminders, audiovisual QA and guided interaction.",
    axis: ["0", "10", "20", "30", "40", "50%"],
    max: 50,
    rows: [
      { group: "Proprietary models", model: "Gemini-3.1-Pro", score: 46.39, color: "teal" },
      { group: "Open-source models", model: "Qwen3-VL", score: 15.0, color: "violet" },
      { group: "Streaming Models", model: "AURA", score: 6.94, color: "coral" },
    ],
  },
};

const resultTabs = [...document.querySelectorAll(".result-tab")];

function renderResults(key) {
  const set = resultSets[key];
  resultTabs.forEach((tab) => {
    const selected = tab.dataset.result === key;
    tab.classList.toggle("is-active", selected);
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  document.getElementById("result-panel").setAttribute("aria-labelledby", set.tab);
  document.getElementById("result-overline").textContent = set.overline;
  document.getElementById("result-heading").textContent = set.heading;
  document.getElementById("result-description").textContent = set.description;
  const chart = document.getElementById("result-chart");
  chart.replaceChildren();
  set.rows.forEach((row) => {
    const line = document.createElement("div");
    line.className = "chart-row";
    const meta = document.createElement("div");
    meta.className = "chart-meta";
    const group = document.createElement("span");
    group.textContent = row.group;
    const model = document.createElement("strong");
    model.textContent = row.model;
    meta.append(group, model);
    const barTrack = document.createElement("div");
    barTrack.className = "bar-track";
    const bar = document.createElement("div");
    bar.className = `bar bar-${row.color}`;
    bar.style.width = `${(row.score / set.max) * 100}%`;
    barTrack.append(bar);
    const score = document.createElement("strong");
    score.className = "chart-score";
    score.textContent = `${row.score.toFixed(key === "explicit" ? 1 : 2)}%`;
    line.append(meta, barTrack, score);
    chart.append(line);
  });
  const axis = document.getElementById("chart-axis");
  axis.replaceChildren(...set.axis.map((label) => {
    const item = document.createElement("span");
    item.textContent = label;
    return item;
  }));
}

resultTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => renderResults(tab.dataset.result));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? resultTabs.length - 1 : (index + 1) % resultTabs.length;
    renderResults(resultTabs[next].dataset.result);
    resultTabs[next].focus();
  });
});

renderDemo("alert");
renderResults("explicit");
