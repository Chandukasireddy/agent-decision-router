// Clef System 1 Maze Navigator Frontend Application

let currentMaze = null;
let isAutoPlaying = false;
let autoPlayTimer = null;
let stepCount = 0;

// DOM Elements
const mazeGridEl = document.getElementById("maze-grid");
const btnStep = document.getElementById("btn-step");
const btnAutoPlay = document.getElementById("btn-autoplay");
const btnReset = document.getElementById("btn-reset");
const btnModalReset = document.getElementById("btn-modal-reset");
const presetSelect = document.getElementById("preset-select");
const speedSlider = document.getElementById("speed-slider");
const speedLabel = document.getElementById("speed-label");

// Stats & Inspector Elements
const statSteps = document.getElementById("stat-steps");
const statDistance = document.getElementById("stat-distance");
const statCache = document.getElementById("stat-cache");
const statLatency = document.getElementById("stat-latency");
const badgeStatus = document.getElementById("decision-status-badge");
const decisionAction = document.getElementById("decision-action");
const decisionConfidence = document.getElementById("decision-confidence");
const decisionSource = document.getElementById("decision-source");
const probListEl = document.getElementById("prob-list");
const gaugeHazard = document.getElementById("gauge-hazard");
const gaugeGoal = document.getElementById("gauge-goal");
const promptPreview = document.getElementById("prompt-preview");
const goalModal = document.getElementById("goal-modal");
const goalSummary = document.getElementById("goal-summary");

// Direction friendly labels
const DIR_LABELS = {
  move_north: "move_north (Up)",
  move_south: "move_south (Down)",
  move_east: "move_east (Right)",
  move_west: "move_west (Left)",
};

// Initialize
async function init() {
  await fetchMazeState();
  setupEventListeners();
}

async function fetchMazeState() {
  try {
    const res = await fetch("/api/state");
    currentMaze = await res.json();
    renderMaze();
    updateHeaderStats();
  } catch (err) {
    console.error("Failed to fetch maze state:", err);
  }
}

function renderMaze() {
  if (!currentMaze) return;

  const { width, height, grid, start_pos, goal_pos, agent_pos, path_history, preset } = currentMaze;

  if (preset && presetSelect.value !== preset) {
    presetSelect.value = preset;
  }

  const cellSize = width >= 19 ? 30 : (width >= 17 ? 32 : (width >= 15 ? 35 : 40));
  mazeGridEl.style.setProperty("--cell-size", `${cellSize}px`);
  mazeGridEl.style.gridTemplateColumns = `repeat(${width}, ${cellSize}px)`;
  mazeGridEl.style.gridTemplateRows = `repeat(${height}, ${cellSize}px)`;
  mazeGridEl.innerHTML = "";

  const pathSet = new Set(path_history.map(([x, y]) => `${x},${y}`));

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const cell = document.createElement("div");
      cell.classList.add("maze-cell");

      const char = grid[y][x];
      const isStart = (x === start_pos[0] && y === start_pos[1]);
      const isGoal = (x === goal_pos[0] && y === goal_pos[1]);
      const isAgent = (x === agent_pos[0] && y === agent_pos[1]);
      const isVisited = pathSet.has(`${x},${y}`) && !isStart && !isGoal && !isAgent;

      if (char === "#") {
        cell.classList.add("cell-wall");
      } else {
        cell.classList.add("cell-floor");
        if (isStart) cell.classList.add("cell-start");
        if (isGoal) cell.classList.add("cell-goal");
        if (isVisited) cell.classList.add("cell-visited");
        if (isAgent) cell.classList.add("cell-agent");
      }

      // Inner icon/label
      if (isAgent) {
        cell.innerHTML = '<span class="agent-icon">🧭</span>';
      } else if (isStart) {
        cell.innerText = "S";
      } else if (isGoal) {
        cell.innerText = "G";
      }

      mazeGridEl.appendChild(cell);
    }
  }

  // Update distance
  const manhattan = Math.abs(agent_pos[0] - goal_pos[0]) + Math.abs(agent_pos[1] - goal_pos[1]);
  statDistance.innerText = `${manhattan} tiles`;
}

async function performStep() {
  if (currentMaze && currentMaze.is_goal) {
    stopAutoPlay();
    return;
  }

  btnStep.disabled = true;

  try {
    const res = await fetch("/api/step", { method: "POST" });
    const data = await res.json();

    stepCount++;
    statSteps.innerText = stepCount;

    // Update current maze agent pos and history
    currentMaze.agent_pos = data.agent_pos;
    currentMaze.path_history = data.path_history;
    currentMaze.is_goal = data.is_goal;

    renderMaze();
    updateInspector(data);

    if (data.is_goal) {
      stopAutoPlay();
      showGoalCelebration();
    }
  } catch (err) {
    console.error("Step execution failed:", err);
  } finally {
    btnStep.disabled = false;
  }
}

function updateInspector(data) {
  // Action banner
  decisionAction.innerText = DIR_LABELS[data.action] || data.action;
  decisionConfidence.innerText = `Conf: ${(data.confidence * 100).toFixed(1)}%`;
  decisionSource.innerText = `Source: ${data.source}`;

  // Badge Status
  badgeStatus.className = "badge";
  if (data.status === "EXECUTABLE") {
    badgeStatus.classList.add("executable");
    badgeStatus.innerText = "EXECUTABLE (≥0.70)";
  } else if (data.status === "CACHE_HIT") {
    badgeStatus.classList.add("cache");
    badgeStatus.innerText = "FAST-PATH CACHE";
  } else if (data.status === "TASK_COMPLETED") {
    badgeStatus.classList.add("goal");
    badgeStatus.innerText = "GOAL REACHED";
  } else {
    badgeStatus.innerText = data.status;
  }

  // Latency & Cache Stats
  statLatency.innerText = `${data.execution_time_ms} ms`;
  if (data.cache_stats) {
    statCache.innerText = `${data.cache_stats.hits} hits (${(data.cache_stats.hit_rate * 100).toFixed(0)}%)`;
  }

  // Probabilities list
  renderProbabilities(data.probabilities, data.action);

  // Gauges
  if (data.is_destructive) {
    gaugeHazard.className = "gauge-status hazard";
    gaugeHazard.innerText = "DEAD-END DETECTED";
  } else {
    gaugeHazard.className = "gauge-status safe";
    gaugeHazard.innerText = "SAFE CORRIDOR";
  }

  if (data.is_goal) {
    gaugeGoal.className = "gauge-status done";
    gaugeGoal.innerText = "TRUE (COMPLETED)";
  } else {
    gaugeGoal.className = "gauge-status";
    gaugeGoal.innerText = "FALSE (IN TRANSIT)";
  }

  // Prompt preview
  if (data.state_prompt) {
    promptPreview.innerText = data.state_prompt;
  }
}

function renderProbabilities(probs, chosenAction) {
  probListEl.innerHTML = "";

  const allDirs = ["move_north", "move_south", "move_east", "move_west"];

  allDirs.forEach(dirKey => {
    const val = probs ? (probs[dirKey] || 0) : 0;
    const pct = (val * 100).toFixed(1);
    const isChosen = (dirKey === chosenAction);

    const row = document.createElement("div");
    row.className = `prob-row ${isChosen ? "active" : ""}`;

    row.innerHTML = `
      <span class="prob-name">${DIR_LABELS[dirKey]}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width: ${pct}%"></div>
      </div>
      <span class="prob-num">${pct}%</span>
    `;

    probListEl.appendChild(row);
  });
}

function updateHeaderStats() {
  statSteps.innerText = stepCount;
}

function showGoalCelebration() {
  goalSummary.innerText = `The agent navigated from Start (S) to Goal (G) in ${stepCount} Clef System 1 steps!`;
  goalModal.classList.remove("hidden");
}

function hideGoalModal() {
  goalModal.classList.add("hidden");
}

async function resetMaze() {
  stopAutoPlay();
  hideGoalModal();
  stepCount = 0;
  statSteps.innerText = "0";
  decisionAction.innerText = "Awaiting Step...";
  decisionConfidence.innerText = "Conf: --";
  decisionSource.innerText = "Source: --";
  badgeStatus.className = "badge";
  badgeStatus.innerText = "READY";
  statLatency.innerText = "-- ms";

  try {
    const res = await fetch("/api/reset", { method: "POST" });
    currentMaze = await res.json();
    renderMaze();
    renderProbabilities({}, "");
  } catch (err) {
    console.error("Reset failed:", err);
  }
}

async function changePreset(preset) {
  stopAutoPlay();
  hideGoalModal();
  stepCount = 0;
  statSteps.innerText = "0";

  try {
    const res = await fetch("/api/preset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preset }),
    });
    currentMaze = await res.json();
    renderMaze();
    renderProbabilities({}, "");
  } catch (err) {
    console.error("Failed to switch preset:", err);
  }
}

function toggleAutoPlay() {
  if (isAutoPlaying) {
    stopAutoPlay();
  } else {
    startAutoPlay();
  }
}

function startAutoPlay() {
  if (currentMaze && currentMaze.is_goal) return;
  isAutoPlaying = true;
  btnAutoPlay.innerHTML = '<span class="btn-icon">⏸</span> Pause';
  btnAutoPlay.classList.add("btn-primary");
  btnAutoPlay.classList.remove("btn-secondary");

  const speed = parseInt(speedSlider.value, 10);
  performStep();
  autoPlayTimer = setInterval(performStep, speed);
}

function stopAutoPlay() {
  isAutoPlaying = false;
  clearInterval(autoPlayTimer);
  btnAutoPlay.innerHTML = '<span class="btn-icon">▶</span> Auto-Play';
  btnAutoPlay.classList.add("btn-secondary");
  btnAutoPlay.classList.remove("btn-primary");
}

function setupEventListeners() {
  btnStep.addEventListener("click", performStep);
  btnAutoPlay.addEventListener("click", toggleAutoPlay);
  btnReset.addEventListener("click", resetMaze);
  btnModalReset.addEventListener("click", resetMaze);

  presetSelect.addEventListener("change", (e) => {
    changePreset(e.target.value);
  });

  speedSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    speedLabel.innerText = `${(val / 1000).toFixed(1)}s`;
    if (isAutoPlaying) {
      clearInterval(autoPlayTimer);
      autoPlayTimer = setInterval(performStep, parseInt(val, 10));
    }
  });
}

// Kickoff
document.addEventListener("DOMContentLoaded", init);
