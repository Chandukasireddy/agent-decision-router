"""FastAPI REST service for agent-decision-router with Jev/Clef Visual Explorer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from agent_decision_router.config import get_settings
from agent_decision_router.models import (
    DecisionResult,
    RouteRequest,
    RouteResponse,
    ToolDefinition,
    ModelTierDefinition,
    SkillDefinition,
)
from agent_decision_router.engine import DecisionEngine
from agent_decision_router.registry import create_default_coding_registry
from agent_decision_router.backend import get_backend


# Shared engine instance for FastAPI service
_engine: Optional[DecisionEngine] = None


def get_api_engine() -> DecisionEngine:
    """Retrieve shared API DecisionEngine instance."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = DecisionEngine(
            registry=create_default_coding_registry(),
            backend=get_backend(),
            settings=settings,
        )
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown."""
    get_api_engine()
    yield


app = FastAPI(
    title="agent-decision-router API",
    description="Confidence-aware tool & skill decision layer for AI coding agents powered by Jev / Clef.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>agent-decision-router | Jev & Clef Decision Explorer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 500: '#6366f1', 600: '#4f46e5' }
          }
        }
      }
    }
  </script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #080c14; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .glass { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(16px); border: 1px solid rgba(255, 255, 255, 0.07); }
    .glass-card { background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }
    .bar-anim { transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1); }
    .tab-active { border-bottom: 2px solid #6366f1; color: #ffffff; font-weight: 600; }
  </style>
</head>
<body class="text-slate-100 min-h-screen flex flex-col justify-between">

  <!-- Top Navigation Header -->
  <header class="border-b border-slate-800 glass sticky top-0 z-50">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-500 via-purple-500 to-cyan-400 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25">
          <i class="fa-solid fa-compass text-lg"></i>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="font-bold text-base sm:text-lg tracking-tight text-white">agent-decision-router</h1>
            <span class="text-xs bg-indigo-500/20 text-indigo-300 font-mono px-2 py-0.5 rounded-full border border-indigo-500/30">Clef / Jev Powered</span>
          </div>
          <p class="text-xs text-slate-400 hidden sm:block">Confidence-Gated System 1 Pre-Flight Decision & Auto Model Selection</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <span class="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-medium">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Cloudflare Workers AI Ready
        </span>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition border border-slate-700">REST Docs</a>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="max-w-7xl mx-auto px-4 sm:px-6 flex gap-6 text-sm text-slate-400 border-t border-slate-800/80">
      <button onclick="switchTab('playground')" id="tab-btn-playground" class="py-3 px-1 tab-active flex items-center gap-2 hover:text-white transition">
        <i class="fa-solid fa-bolt"></i> Decision Playground
      </button>
      <button onclick="switchTab('models')" id="tab-btn-models" class="py-3 px-1 flex items-center gap-2 hover:text-white transition">
        <i class="fa-solid fa-microchip"></i> Public AI Models
      </button>
      <button onclick="switchTab('skills')" id="tab-btn-skills" class="py-3 px-1 flex items-center gap-2 hover:text-white transition">
        <i class="fa-solid fa-toolbox"></i> Agent Skills
      </button>
      <button onclick="switchTab('howitworks')" id="tab-btn-howitworks" class="py-3 px-1 flex items-center gap-2 hover:text-white transition">
        <i class="fa-solid fa-graduation-cap"></i> How Jev Works
      </button>
    </div>
  </header>

  <!-- Main Content Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 py-8 flex-1 w-full">

    <!-- =======================================================================
         TAB 1: DECISION PLAYGROUND
         ======================================================================= -->
    <div id="tab-playground" class="space-y-8">
      <!-- Input Panel -->
      <section class="glass rounded-2xl p-6 shadow-xl border border-slate-800">
        <div class="flex items-center justify-between mb-3">
          <label class="text-sm font-semibold text-slate-200 flex items-center gap-2">
            <i class="fa-solid fa-terminal text-indigo-400"></i> Enter User Prompt or Agent Task Goal
          </label>
          <span class="text-xs text-slate-400">Sub-second structured evaluation</span>
        </div>
        <textarea id="promptInput" rows="3" class="w-full bg-slate-900/90 border border-slate-700/80 rounded-xl p-3.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono transition" placeholder="e.g. Run pytest tests/auth_test.py to inspect login errors"></textarea>

        <!-- Presets -->
        <div class="mt-3 flex flex-wrap items-center gap-2">
          <span class="text-xs text-slate-400 font-medium mr-1">Try Scenarios:</span>
          <button onclick="setPrompt('Run pytest tests/auth_test.py to inspect login errors')" class="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition border border-slate-700/60">🧪 Unit Tests (pytest)</button>
          <button onclick="setPrompt('Refactor entire authentication architecture to zero-trust OAuth2 with PKCE, migrate sessions, and audit timing attacks')" class="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-cyan-300 transition border border-cyan-500/30">🏗️ Architecture Refactor</button>
          <button onclick="setPrompt('Prove race condition in concurrent lock-free queue and verify memory orderings')" class="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-purple-300 transition border border-purple-500/30">🧠 Deep Logic (DeepSeek)</button>
          <button onclick="setPrompt('Inspect screenshot of mobile modal and debug why button overflows')" class="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-amber-300 transition border border-amber-500/30">👁️ UI Vision (GPT-4o)</button>
          <button onclick="setPrompt('Fix small typo in header comment in README.md')" class="text-xs px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-emerald-300 transition border border-emerald-500/30">✏️ Simple Typo (Haiku)</button>
          <button onclick="setPrompt('rm -rf /var/lib/docker && drop database production')" class="text-xs px-2.5 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 transition border border-rose-500/30">⚠️ Destructive Operation</button>
        </div>

        <div class="mt-5 flex items-center justify-between pt-4 border-t border-slate-800">
          <label class="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
            <input type="checkbox" id="skipCache" class="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0">
            <span>Bypass Stage 1 Fast-Path Cache</span>
          </label>
          <button id="routeBtn" onclick="evaluateRoute()" class="px-6 py-2.5 rounded-xl bg-gradient-to-r from-indigo-500 via-indigo-600 to-cyan-500 hover:opacity-95 text-white font-medium text-sm shadow-lg shadow-indigo-500/25 flex items-center gap-2 transition active:scale-95">
            <i class="fa-solid fa-bolt"></i> Evaluate with Jev / Clef
          </button>
        </div>
      </section>

      <!-- Output Container -->
      <div id="outputStage" class="space-y-6 hidden">
        
        <!-- Layer 0: Verdict Header -->
        <div id="verdictBanner" class="glass rounded-2xl p-5 border-l-4 transition-all shadow-xl">
          <div class="flex flex-wrap items-center justify-between gap-4">
            <div class="flex items-center gap-4">
              <div id="statusIcon" class="w-12 h-12 rounded-xl flex items-center justify-center text-xl shadow-md"></div>
              <div>
                <div class="flex items-center gap-2">
                  <span id="statusBadge" class="text-lg font-bold tracking-tight"></span>
                  <span id="sourceBadge" class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 font-mono"></span>
                </div>
                <p id="rationaleText" class="text-xs text-slate-400 mt-1"></p>
              </div>
            </div>
            <div class="flex items-center gap-8 text-sm">
              <div class="text-right">
                <span class="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Latency</span>
                <span id="latencyValue" class="font-mono font-bold text-yellow-400 text-lg"></span>
              </div>
              <div class="text-right">
                <span class="text-xs text-slate-400 block uppercase tracking-wider font-semibold">Confidence</span>
                <span id="confidenceValue" class="font-mono font-bold text-lg"></span>
              </div>
            </div>
          </div>
        </div>

        <!-- 2-Column: Model Auto-Selector & Skill Decision -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

          <!-- Layer 1: Public AI Model Auto-Selector -->
          <div class="glass rounded-2xl p-6 shadow-xl border border-slate-800">
            <div class="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
              <h2 class="font-semibold text-sm flex items-center gap-2 text-cyan-400">
                <i class="fa-solid fa-microchip"></i> 1. AI Model Auto-Selector (Public Models)
              </h2>
              <span id="selectedModelBadge" class="text-xs font-mono font-bold px-3 py-1 rounded-md bg-cyan-500/20 text-cyan-300 border border-cyan-500/30"></span>
            </div>
            <div id="modelBarsContainer" class="space-y-4"></div>
          </div>

          <!-- Layer 2: Domain Skill & Tool Selection -->
          <div class="glass rounded-2xl p-6 shadow-xl border border-slate-800">
            <div class="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
              <h2 class="font-semibold text-sm flex items-center gap-2 text-purple-400">
                <i class="fa-solid fa-toolbox"></i> 2. Specialized Skill & Tool Decision
              </h2>
              <span id="selectedSkillBadge" class="text-xs font-mono font-bold px-3 py-1 rounded-md bg-purple-500/20 text-purple-300 border border-purple-500/30"></span>
            </div>
            <div id="toolBarsContainer" class="space-y-4"></div>
          </div>

        </div>

        <!-- 2-Column: Safety Gates & Agent Execution Directive -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

          <!-- Layer 3: Safety & Policy Gates -->
          <div class="glass rounded-2xl p-6 shadow-xl border border-slate-800">
            <div class="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
              <h2 class="font-semibold text-sm flex items-center gap-2 text-amber-400">
                <i class="fa-solid fa-shield-halved"></i> 3. Safety & Policy Gates (Noul Boolean Probability)
              </h2>
              <span class="text-xs text-slate-500">Calibrated Guards</span>
            </div>
            <div class="space-y-3.5">
              <div class="flex items-center justify-between p-3.5 rounded-xl bg-slate-900/70 border border-slate-800">
                <div>
                  <span class="font-medium text-sm block text-slate-200">Destructive Action?</span>
                  <span class="text-xs text-slate-500">Mutates disk, git history, or environment</span>
                </div>
                <span id="destructivePill" class="text-xs font-semibold px-3 py-1 rounded-full"></span>
              </div>
              <div class="flex items-center justify-between p-3.5 rounded-xl bg-slate-900/70 border border-slate-800">
                <div>
                  <span class="font-medium text-sm block text-slate-200">Human Confirmation?</span>
                  <span class="text-xs text-slate-500">Halts execution for explicit approval</span>
                </div>
                <span id="confirmationPill" class="text-xs font-semibold px-3 py-1 rounded-full"></span>
              </div>
              <div class="flex items-center justify-between p-3.5 rounded-xl bg-slate-900/70 border border-slate-800">
                <div>
                  <span class="font-medium text-sm block text-slate-200">Task Completed?</span>
                  <span class="text-xs text-slate-500">Signals agent loop to finish</span>
                </div>
                <span id="completionPill" class="text-xs font-semibold px-3 py-1 rounded-full"></span>
              </div>
            </div>
          </div>

          <!-- Layer 4: Downstream Agent Execution Directive -->
          <div class="glass rounded-2xl p-6 shadow-xl border border-slate-800 flex flex-col justify-between">
            <div>
              <div class="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
                <h2 class="font-semibold text-sm flex items-center gap-2 text-emerald-400">
                  <i class="fa-solid fa-paper-plane"></i> 4. Downstream Agent Directive
                </h2>
                <span class="text-xs text-slate-500 font-mono">Ready-to-Dispatch</span>
              </div>
              <pre id="directiveJson" class="text-xs font-mono bg-slate-900/90 p-4 rounded-xl border border-slate-800 text-emerald-300 overflow-x-auto leading-relaxed"></pre>
            </div>
            <div class="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <span>Autonomous Execution Status:</span>
              <span id="autonomousPill" class="font-mono font-bold"></span>
            </div>
          </div>

        </div>

      </div>
    </div>

    <!-- =======================================================================
         TAB 2: PUBLIC AI MODELS DIRECTORY
         ======================================================================= -->
    <div id="tab-models" class="space-y-6 hidden">
      <div class="glass rounded-2xl p-6 mb-6">
        <h2 class="text-lg font-bold text-white mb-1 flex items-center gap-2">
          <i class="fa-solid fa-microchip text-cyan-400"></i> Publicly Available AI Models Catalogue
        </h2>
        <p class="text-xs text-slate-400">
          The decision router uses Jev/Clef to evaluate task requirements and automatically route to the optimal model based on capability, cost, and context size.
        </p>
      </div>
      <div id="modelsGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>
    </div>

    <!-- =======================================================================
         TAB 3: AGENT SKILLS DIRECTORY
         ======================================================================= -->
    <div id="tab-skills" class="space-y-6 hidden">
      <div class="glass rounded-2xl p-6 mb-6">
        <h2 class="text-lg font-bold text-white mb-1 flex items-center gap-2">
          <i class="fa-solid fa-toolbox text-purple-400"></i> Agent Domain Skills Catalogue
        </h2>
        <p class="text-xs text-slate-400">
          High-level operational skill domains mapped to underlying granular tools for specialized agent execution.
        </p>
      </div>
      <div id="skillsGrid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"></div>
    </div>

    <!-- =======================================================================
         TAB 4: HOW JEV WORKS (EDUCATIONAL)
         ======================================================================= -->
    <div id="tab-howitworks" class="space-y-6 hidden">
      <div class="glass rounded-2xl p-8 max-w-4xl mx-auto space-y-6">
        <div>
          <h2 class="text-xl font-bold text-white mb-2 flex items-center gap-2">
            <i class="fa-solid fa-graduation-cap text-indigo-400"></i> How Jev & Clef Work: Structured Evaluation vs Token Generation
          </h2>
          <p class="text-sm text-slate-300 leading-relaxed">
            Traditional LLMs (GPT-4, Claude) are <strong>generative token predictors</strong>: you ask a question, and they predict the next token string. This is slow (3-10s), expensive, and susceptible to hallucination.
          </p>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800">
            <h3 class="font-bold text-amber-400 text-sm mb-2 flex items-center gap-2">
              <i class="fa-solid fa-triangle-exclamation"></i> Traditional LLM Routing
            </h3>
            <ul class="text-xs text-slate-400 space-y-2">
              <li>• Outputs freeform strings (needs regex/JSON parsing).</li>
              <li>• Uncalibrated confidence: LLM can be 100% wrong with 100% false certainty.</li>
              <li>• Latency: 2,000ms – 6,000ms.</li>
              <li>• High token billing per request.</li>
            </ul>
          </div>
          <div class="p-4 rounded-xl bg-indigo-950/40 border border-indigo-500/30">
            <h3 class="font-bold text-cyan-400 text-sm mb-2 flex items-center gap-2">
              <i class="fa-solid fa-check"></i> Jev / Clef Evaluation
            </h3>
            <ul class="text-xs text-slate-300 space-y-2">
              <li>• Evaluates a <code>state</code> against typed mathematical questions.</li>
              <li>• Returns <strong>calibrated probabilities</strong> (0.0 to 1.0) across candidate criteria.</li>
              <li>• Latency: 300ms – 600ms (via Cloudflare Workers AI free tier).</li>
              <li>• Confidence-gated: Refuses weak matches (&lt;0.40) to prevent hallucinated actions.</li>
            </ul>
          </div>
        </div>

        <div class="space-y-3 pt-4 border-t border-slate-800">
          <h3 class="font-bold text-sm text-white">The Three Typed Evaluation Primitives in Jev:</h3>
          <div class="space-y-2 text-xs">
            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span class="font-mono font-bold text-cyan-400">1. Choice Question</span>
              <p class="text-slate-400 mt-1">Given N options with text criteria, returns normalized softmax probabilities across all options. Used for <strong>Auto Model Selection</strong> and <strong>Skill/Tool Decision</strong>.</p>
            </div>
            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span class="font-mono font-bold text-emerald-400">2. Noul Question</span>
              <p class="text-slate-400 mt-1">Evaluates a boolean hypothesis and returns a calibrated probability (0.00 to 1.00). Used for <strong>is_destructive</strong> safety checks and <strong>task_completion</strong> termination.</p>
            </div>
            <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-800">
              <span class="font-mono font-bold text-purple-400">3. Score Question</span>
              <p class="text-slate-400 mt-1">Calibrated numeric rating along an ordinal scale (e.g. frustration level 0 to 2).</p>
            </div>
          </div>
        </div>

      </div>
    </div>

  </main>

  <footer class="border-t border-slate-800/80 glass py-4 text-center text-xs text-slate-500">
    agent-decision-router • Powered by Cloudflare Workers AI & TypeSafe Evaluation Architecture
  </footer>

  <script>
    let currentModels = [];
    let currentSkills = [];

    async function loadCatalogues() {
      try {
        const [modelsRes, skillsRes] = await Promise.all([
          fetch('/v1/models'),
          fetch('/v1/skills')
        ]);
        currentModels = await modelsRes.json();
        currentSkills = await skillsRes.json();
        renderModelsCatalogue(currentModels);
        renderSkillsCatalogue(currentSkills);
      } catch (e) {
        console.error("Failed to load catalogues", e);
      }
    }

    function renderModelsCatalogue(models) {
      const grid = document.getElementById('modelsGrid');
      if (!grid) return;
      grid.innerHTML = '';
      models.forEach(m => {
        grid.innerHTML += `
          <div class="glass-card rounded-2xl p-5 hover:border-slate-700 transition">
            <div class="flex items-center justify-between mb-3">
              <span class="text-xs font-mono font-bold px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">${m.provider}</span>
              <span class="text-xs font-mono text-slate-400"><i class="fa-regular fa-clock mr-1"></i>${m.context_window}</span>
            </div>
            <h3 class="font-bold text-base text-white font-mono">${m.name}</h3>
            <p class="text-xs text-slate-400 mt-1 mb-3">${m.description}</p>
            <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
              <span class="text-indigo-400 font-semibold block mb-1">Routing Criteria:</span>
              ${m.criteria}
            </div>
            <div class="mt-3 flex items-center justify-between text-xs text-slate-500 font-mono">
              <span>Specialty: <strong class="text-slate-300">${m.specialty}</strong></span>
              <span class="capitalize px-2 py-0.5 rounded bg-slate-800 text-slate-400">${m.cost_tier}</span>
            </div>
          </div>
        `;
      });
    }

    function renderSkillsCatalogue(skills) {
      const grid = document.getElementById('skillsGrid');
      if (!grid) return;
      grid.innerHTML = '';
      skills.forEach(s => {
        const toolsHtml = s.tools.map(t => `<span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-xs border border-slate-700">${t}</span>`).join(' ');
        grid.innerHTML += `
          <div class="glass-card rounded-2xl p-5 hover:border-slate-700 transition">
            <div class="flex items-center justify-between mb-3">
              <span class="text-xs font-mono font-bold px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">${s.name}</span>
              <span class="text-xs text-slate-500">${s.metadata.category || 'General'}</span>
            </div>
            <p class="text-xs text-slate-300 mb-3">${s.description}</p>
            <div class="p-3 rounded-xl bg-slate-900/80 border border-slate-800 text-xs text-slate-300 mb-3">
              <span class="text-purple-400 font-semibold block mb-1">Activation Criteria:</span>
              ${s.criteria}
            </div>
            <div>
              <span class="text-xs text-slate-400 block mb-1">Member Tools:</span>
              <div class="flex flex-wrap gap-1.5">${toolsHtml}</div>
            </div>
          </div>
        `;
      });
    }

    function switchTab(tabId) {
      ['playground', 'models', 'skills', 'howitworks'].forEach(t => {
        const tabEl = document.getElementById('tab-' + t);
        const btnEl = document.getElementById('tab-btn-' + t);
        if (t === tabId) {
          tabEl.classList.remove('hidden');
          btnEl.className = 'py-3 px-1 tab-active flex items-center gap-2 hover:text-white transition';
        } else {
          tabEl.classList.add('hidden');
          btnEl.className = 'py-3 px-1 flex items-center gap-2 hover:text-white transition';
        }
      });
    }

    function setPrompt(text) {
      document.getElementById('promptInput').value = text;
      evaluateRoute();
    }

    async function evaluateRoute() {
      const prompt = document.getElementById('promptInput').value.trim();
      if (!prompt) return;

      const btn = document.getElementById('routeBtn');
      const skipCache = document.getElementById('skipCache').checked;
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner animate-spin"></i> Evaluating with Jev...';

      try {
        const resp = await fetch('/v1/route', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state: prompt, skip_cache: skipCache })
        });
        const data = await resp.json();
        if (data && data.decision) {
          renderDecision(data.decision);
        }
      } catch (err) {
        alert('Routing failed: ' + err);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Evaluate with Jev / Clef';
      }
    }

    function renderDecision(d) {
      document.getElementById('outputStage').classList.remove('hidden');

      // 0. Verdict
      const banner = document.getElementById('verdictBanner');
      const icon = document.getElementById('statusIcon');
      const badge = document.getElementById('statusBadge');
      const rationale = document.getElementById('rationaleText');

      document.getElementById('latencyValue').innerText = d.execution_time_ms.toFixed(1) + ' ms';
      document.getElementById('confidenceValue').innerText = (d.confidence * 100).toFixed(1) + '%';
      document.getElementById('sourceBadge').innerText = d.source;

      if (d.status === 'EXECUTABLE') {
        banner.className = 'glass rounded-2xl p-5 border-l-4 border-emerald-500 shadow-emerald-500/10 shadow-lg';
        icon.className = 'w-12 h-12 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-xl shadow-md';
        icon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
        badge.className = 'text-lg font-bold text-emerald-400';
        badge.innerText = 'EXECUTABLE: ' + d.action;
      } else if (d.status === 'LOOKAHEAD_RESOLVED') {
        banner.className = 'glass rounded-2xl p-5 border-l-4 border-amber-500 shadow-amber-500/10 shadow-lg';
        icon.className = 'w-12 h-12 rounded-xl bg-amber-500/20 text-amber-400 flex items-center justify-center text-xl shadow-md';
        icon.innerHTML = '<i class="fa-solid fa-code-fork"></i>';
        badge.className = 'text-lg font-bold text-amber-400';
        badge.innerText = 'LOOKAHEAD RESOLVED: ' + d.action;
      } else if (d.status === 'DECLINED') {
        banner.className = 'glass rounded-2xl p-5 border-l-4 border-rose-500 shadow-rose-500/10 shadow-lg';
        icon.className = 'w-12 h-12 rounded-xl bg-rose-500/20 text-rose-400 flex items-center justify-center text-xl shadow-md';
        icon.innerHTML = '<i class="fa-solid fa-ban"></i>';
        badge.className = 'text-lg font-bold text-rose-400';
        badge.innerText = 'DECLINED (Refused Weak Match)';
      } else if (d.status === 'TASK_COMPLETED') {
        banner.className = 'glass rounded-2xl p-5 border-l-4 border-purple-500 shadow-purple-500/10 shadow-lg';
        icon.className = 'w-12 h-12 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center text-xl shadow-md';
        icon.innerHTML = '<i class="fa-solid fa-flag-checkered"></i>';
        badge.className = 'text-lg font-bold text-purple-400';
        badge.innerText = 'TASK COMPLETED';
      } else {
        banner.className = 'glass rounded-2xl p-5 border-l-4 border-cyan-500 shadow-cyan-500/10 shadow-lg';
        icon.className = 'w-12 h-12 rounded-xl bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xl shadow-md';
        icon.innerHTML = '<i class="fa-solid fa-bolt"></i>';
        badge.className = 'text-lg font-bold text-cyan-400';
        badge.innerText = d.status + ': ' + d.action;
      }
      rationale.innerText = d.reason || 'Decision made by Jev/Clef System 1 evaluation model.';

      // 1. Model Selection
      document.getElementById('selectedModelBadge').innerText = 'Chosen Model: ' + (d.selected_model || 'gemini_2_0_flash');
      const modelContainer = document.getElementById('modelBarsContainer');
      modelContainer.innerHTML = '';
      const mProbs = d.model_probabilities || {};
      const sortedModels = Object.entries(mProbs).sort((a,b) => b[1] - a[1]);
      sortedModels.forEach(([mName, prob]) => {
        const isSelected = (mName === d.selected_model);
        const percent = (prob * 100).toFixed(1);
        modelContainer.innerHTML += `
          <div>
            <div class="flex justify-between text-xs mb-1.5 font-mono">
              <span class="${isSelected ? 'text-cyan-300 font-bold' : 'text-slate-400'}">${isSelected ? '● ' : ''}${mName}</span>
              <span class="${isSelected ? 'text-cyan-300 font-bold' : 'text-slate-500'}">${percent}%</span>
            </div>
            <div class="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden p-0.5 border border-slate-800">
              <div class="h-full rounded-full bar-anim ${isSelected ? 'bg-cyan-400 shadow-sm shadow-cyan-400' : 'bg-slate-700'}" style="width: ${percent}%"></div>
            </div>
          </div>
        `;
      });

      // 2. Skill & Tool Decision
      document.getElementById('selectedSkillBadge').innerText = 'Domain Skill: ' + (d.selected_skill || 'general');
      const toolContainer = document.getElementById('toolBarsContainer');
      toolContainer.innerHTML = '';
      const tProbs = d.probabilities || {};
      const sortedTools = Object.entries(tProbs).sort((a,b) => b[1] - a[1]);
      sortedTools.slice(0, 6).forEach(([tName, prob]) => {
        const isSelected = (tName === d.action);
        const percent = (prob * 100).toFixed(1);
        toolContainer.innerHTML += `
          <div>
            <div class="flex justify-between text-xs mb-1.5 font-mono">
              <span class="${isSelected ? 'text-purple-300 font-bold' : 'text-slate-400'}">${isSelected ? '● ' : ''}${tName}</span>
              <span class="${isSelected ? 'text-purple-300 font-bold' : 'text-slate-500'}">${percent}%</span>
            </div>
            <div class="w-full bg-slate-900 rounded-full h-2.5 overflow-hidden p-0.5 border border-slate-800">
              <div class="h-full rounded-full bar-anim ${isSelected ? 'bg-purple-400 shadow-sm shadow-purple-400' : 'bg-slate-700'}" style="width: ${percent}%"></div>
            </div>
          </div>
        `;
      });

      // 3. Safety Gates
      const destPill = document.getElementById('destructivePill');
      if (d.is_destructive) {
        destPill.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-rose-500/20 text-rose-400 border border-rose-500/30';
        destPill.innerText = 'YES (Conf: ' + d.destructive_confidence.toFixed(2) + ')';
      } else {
        destPill.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        destPill.innerText = 'NO (Safe)';
      }

      const confPill = document.getElementById('confirmationPill');
      if (d.requires_confirmation) {
        confPill.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30';
        confPill.innerText = 'REQUIRED (Pause)';
      } else {
        confPill.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
        confPill.innerText = 'AUTO-EXECUTE';
      }

      const compPill = document.getElementById('completionPill');
      if (d.task_completion) {
        compPill.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-purple-500/20 text-purple-400 border border-purple-500/30';
        compPill.innerText = 'COMPLETED';
      } else {
        compPill.className = 'text-xs font-semibold px-3 py-1 rounded-full bg-slate-800 text-slate-400 border border-slate-700';
        compPill.innerText = 'IN PROGRESS';
      }

      // 4. Downstream Directive
      const directive = d.execution_directive || {};
      document.getElementById('directiveJson').innerText = JSON.stringify(directive, null, 2);
      const autoPill = document.getElementById('autonomousPill');
      if (directive.autonomous_execution) {
        autoPill.className = 'font-mono font-bold text-emerald-400';
        autoPill.innerText = '✔ YES (Approved for Execution)';
      } else {
        autoPill.className = 'font-mono font-bold text-amber-400';
        autoPill.innerText = '✖ NO (Requires Approval or Declined)';
      }
    }

    // Initialize catalogues on page load
    loadCatalogues();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard() -> str:
    """Serve the interactive visual routing dashboard and Jev explorer."""
    return DASHBOARD_HTML


@app.get("/v1/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Healthcheck endpoint reporting backend status and configuration."""
    settings = get_settings()
    engine = get_api_engine()
    return {
        "status": "healthy",
        "backend": settings.router_backend,
        "cloudflare_configured": settings.is_cloudflare_configured,
        "fast_path_cache_enabled": settings.fast_path_cache_enabled,
        "registered_tools_count": len(engine.registry.list_tools()),
        "registered_models_count": len(engine.registry.list_model_tiers()),
        "registered_skills_count": len(engine.registry.list_skills()),
    }


@app.post("/v1/route", response_model=RouteResponse, tags=["Routing"])
async def route_action(request: RouteRequest) -> RouteResponse:
    """Route the next tool and model action given agent state and context."""
    engine = get_api_engine()
    try:
        decision = await engine.route(request)
        return RouteResponse(
            success=True,
            decision=decision,
            meta={"backend": engine.settings.router_backend},
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing error: {str(exc)}",
        )


@app.get("/v1/models", response_model=List[ModelTierDefinition], tags=["Registry"])
async def list_models() -> List[ModelTierDefinition]:
    """Retrieve all registered public AI models and their selection criteria."""
    engine = get_api_engine()
    return engine.registry.list_model_tiers()


@app.get("/v1/skills", response_model=List[SkillDefinition], tags=["Registry"])
async def list_skills() -> List[SkillDefinition]:
    """Retrieve all registered agent skills and their member tools."""
    engine = get_api_engine()
    return engine.registry.list_skills()



@app.post("/v1/register", tags=["Registry"])
async def register_tool(tool: ToolDefinition) -> Dict[str, Any]:
    """Dynamically register a new tool or update an existing one."""
    engine = get_api_engine()
    registered = engine.registry.register_tool(
        name=tool.name,
        description=tool.description,
        criteria=tool.criteria,
        parameters=tool.parameters,
        is_destructive=tool.is_destructive,
        metadata=tool.metadata,
    )
    return {
        "success": True,
        "registered_tool": registered.model_dump(),
        "total_tools": len(engine.registry.list_tools()),
    }


@app.delete("/v1/tools/{tool_name}", tags=["Registry"])
async def unregister_tool(tool_name: str) -> Dict[str, Any]:
    """Unregister a tool from the router."""
    engine = get_api_engine()
    removed = engine.registry.unregister(tool_name)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found",
        )
    return {"success": True, "unregistered": tool_name}


@app.get("/v1/tools", response_model=List[ToolDefinition], tags=["Registry"])
async def list_tools() -> List[ToolDefinition]:
    """Retrieve all registered tools and their routing criteria."""
    engine = get_api_engine()
    return engine.registry.list_tools()


@app.get("/v1/cache/stats", tags=["Cache"])
async def cache_stats() -> Dict[str, Any]:
    """Get Stage 1 Fast-Path Cache telemetry and performance statistics."""
    engine = get_api_engine()
    stats = engine.cache.stats.to_dict()
    stats["current_cache_size"] = engine.cache.size()
    return stats


@app.post("/v1/cache/clear", tags=["Cache"])
async def cache_clear() -> Dict[str, Any]:
    """Clear all cached decisions from the Fast-Path Cache."""
    engine = get_api_engine()
    engine.cache.clear()
    return {
        "success": True,
        "message": "Cache successfully cleared",
        "current_cache_size": engine.cache.size(),
    }
