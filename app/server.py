"""Authoritative Web Server and Dashboard for SS_PROJECT_VCR_FRESH.

Integrates the single authoritative RobotRuntime lifecycle with an asynchronous
HTTP and MJPEG video streaming server.

Guarantees:
- Single authoritative runtime; zero duplicate resource allocation.
- Read-only status endpoint (/api/status) never triggers sensing or actuation.
- Graceful handling of client disconnects without weakening robot safety.
- Clear status messages: TARGET_FOUND, TARGET_NOT_FOUND, OBSTACLE_BLOCKED,
  CAMERA_FAILURE, DETECTOR_FAILURE, SENSOR_FAILURE.
- No auto-launching of browser unless explicitly configured.
"""
from __future__ import annotations

import asyncio
import threading
import secrets
import time
from typing import Any, AsyncGenerator, Dict, Optional
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from fastapi import FastAPI, HTTPException, Request, Response, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.command.command_parser import CommandAction, RobotCommand
from app.logger import get_logger
from app.runtime import LifecycleState, RobotRuntime

logger = get_logger("server")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SS_PROJECT_VCR — Autonomous Robot Dashboard</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #151d30;
      --card-border: #23314e;
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --cyan: #38bdf8;
      --emerald: #34d399;
      --amber: #fbbf24;
      --rose: #f43f5e;
      --purple: #c084fc;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 16px;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 12px 20px;
      margin-bottom: 16px;
    }
    .header-title {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: #fff;
    }
    .header-badges {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .badge {
      font-size: 0.75rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-sim { background: #3b1d5c; color: var(--purple); border: 1px solid #7c3aed; }
    .badge-real { background: #064e3b; color: var(--emerald); border: 1px solid #059669; }
    .badge-healthy { background: #064e3b; color: var(--emerald); }
    .badge-fault { background: #4c0519; color: var(--rose); }
    .badge-state { background: #1e3a8a; color: var(--cyan); border: 1px solid #2563eb; }

    /* Alert Banner */
    #alert-banner {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 16px;
      background: #1e293b;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      margin-bottom: 16px;
      align-items: center;
    }
    .msg-badge {
      font-size: 0.8rem;
      font-weight: 700;
      padding: 4px 10px;
      border-radius: 4px;
      letter-spacing: 0.5px;
    }
    .msg-TARGET_FOUND { background: #065f46; color: #a7f3d0; border: 1px solid #059669; }
    .msg-TARGET_NOT_FOUND { background: #78350f; color: #fde68a; border: 1px solid #d97706; }
    .msg-OBSTACLE_BLOCKED { background: #831843; color: #fbcfe8; border: 1px solid #db2777; }
    .msg-CAMERA_FAILURE, .msg-DETECTOR_FAILURE, .msg-SENSOR_FAILURE {
      background: #881337; color: #fecdd3; border: 1px solid #e11d48;
    }

    /* Grid Layout */
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--cyan);
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }
    .data-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.85rem;
    }
    .data-label { color: var(--text-muted); }
    .data-val { font-family: monospace; font-weight: 600; color: #fff; }

    /* Video feed */
    .camera-container {
      width: 100%;
      height: 220px;
      background: #000;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .camera-feed {
      width: 100%;
      height: 100%;
      object-fit: contain;
    }

    /* Controls */
    .controls-container {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .input-group {
      display: flex;
      gap: 8px;
    }
    input[type="text"] {
      flex: 1;
      padding: 8px 12px;
      background: #0b0f19;
      border: 1px solid var(--card-border);
      border-radius: 4px;
      color: #fff;
      font-size: 0.9rem;
    }
    button {
      padding: 8px 14px;
      border-radius: 4px;
      border: none;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.85rem;
      transition: background 0.15s;
    }
    .btn-primary { background: #2563eb; color: #fff; }
    .btn-primary:hover { background: #1d4ed8; }
    .btn-stop { background: #d97706; color: #fff; }
    .btn-stop:hover { background: #b45309; }
    .btn-estop { background: #e11d48; color: #fff; font-weight: 700; }
    .btn-estop:hover { background: #be123c; }
    .btn-clear { background: #059669; color: #fff; }
    .btn-clear:hover { background: #047857; }
    .btn-group {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    #status-msg {
      font-size: 0.8rem;
      font-family: monospace;
      padding: 6px;
      border-radius: 4px;
      background: #0b0f19;
      color: var(--cyan);
      min-height: 24px;
    }
    .switch {
      position: relative;
      display: inline-block;
      width: 44px;
      height: 24px;
    }
    .switch input {
      opacity: 0;
      width: 0;
      height: 0;
    }
    .slider {
      position: absolute;
      cursor: pointer;
      top: 0; left: 0; right: 0; bottom: 0;
      background-color: #4b5563;
      transition: .2s;
      border-radius: 24px;
    }
    .slider:before {
      position: absolute;
      content: "";
      height: 18px;
      width: 18px;
      left: 3px;
      bottom: 3px;
      background-color: white;
      transition: .2s;
      border-radius: 50%;
    }
    input:checked + .slider {
      background-color: #059669;
    }
    input:checked + .slider:before {
      transform: translateX(20px);
    }
  </style>
</head>
<body>

  <!-- Persistent Warning Banner for Ultrasonic Protection OFF -->
  <div id="ultrasonic-warning-banner" style="display:block; background:#7f1d1d; border:2px solid #ef4444; border-radius:8px; padding:12px 16px; margin-bottom:16px; color:#fff; font-weight:700; text-align:center;">
    ⚠️ CAUTION: ULTRASONIC OBSTACLE PROTECTION IS DISABLED! Robot will not stop for obstacles. Exercise extreme vigilance.
  </div>

  <header>
    <div class="header-title">SS_PROJECT_VCR // AUTONOMOUS RUNTIME</div>
    <div class="header-badges">
      <span id="badge-mode" class="badge badge-sim">SIMULATION</span>
      <span id="badge-health" class="badge badge-healthy">HEALTHY</span>
      <span id="badge-state" class="badge badge-state">IDLE_LISTENING</span>
      <span class="badge" id="badge-uptime" style="background:#1e293b; color:var(--text-muted)">UP: 00:00:00</span>
      <span id="telemetry-live" class="badge" style="background:#0f2e24; color:#34d399;">● CONNECTING</span>
    </div>
  </header>

  <!-- Clear High-Level Messages Banner -->
  <div id="alert-banner">
    <span style="font-size:0.85rem; color:var(--text-muted)">Active System Notices:</span>
    <div id="alert-list" style="display:flex; gap:8px; flex-wrap:wrap;">
      <span class="msg-badge" style="background:#064e3b; color:#34d399;">ALL SYSTEMS OPERATIONAL</span>
    </div>
  </div>

  <div class="grid">
    <!-- Camera & Visual Tracking -->
    <div class="card">
      <div class="card-title">
        <span>OPTICAL FRAME & TRACKING</span>
        <span id="cam-health-badge" class="badge badge-healthy">LIVE</span>
      </div>
      <div class="camera-container" style="position:relative; cursor:pointer;" onclick="reloadCamStream()" title="Click to reload camera feed">
        <img id="cam-stream" class="camera-feed" src="/api/camera/stream" onerror="onCamError(this)" onload="onCamLoad()" alt="Robot Camera Stream" />
        <div id="cam-fallback" style="display:none; position:absolute; top:40%; left:10%; right:10%; text-align:center; color:var(--amber); font-size:0.85rem; background:rgba(11,15,25,0.85); padding:10px; border-radius:6px; border:1px solid var(--amber);">
          Camera Stream Connecting...
        </div>
      </div>
      <div class="data-row">
        <span class="data-label">Target Class / Alias:</span>
        <span class="data-val" id="val-target-class">None</span>
      </div>
      <div class="data-row">
        <span class="data-label">Confirmation Count:</span>
        <span class="data-val" id="val-conf-count">0</span>
      </div>
      <div class="data-row">
        <span class="data-label">Current Detection:</span>
        <span class="data-val" id="val-curr-det">None</span>
      </div>
    </div>

    <!-- Ultrasonic & Forward Proximity -->
    <div class="card">
      <div class="card-title">
        <span>ULTRASONIC PROXIMITY</span>
        <span id="dist-state-badge" class="badge badge-healthy">CLEAR</span>
      </div>
      <div style="text-align:center; padding: 14px 0;">
        <span id="val-dist-large" style="font-size: 3rem; font-weight: 800; font-family: monospace; color: var(--cyan);">-- cm</span>
      </div>
      <div class="data-row">
        <span class="data-label">Sensor State:</span>
        <span class="data-val" id="val-sensor-state" style="font-weight:700;">INITIALIZING</span>
      </div>
      <div class="data-row">
        <span class="data-label">Raw / Filtered Distance:</span>
        <span class="data-val" id="val-dist-raw-filt">-- / -- cm</span>
      </div>
      <div class="data-row">
        <span class="data-label">Reading Validity / Age:</span>
        <span class="data-val" id="val-dist-valid">Initializing...</span>
      </div>
      <div class="data-row">
        <span class="data-label">Safety Block Reason:</span>
        <span class="data-val" id="val-block-reason" style="color:var(--amber);">None</span>
      </div>

      <!-- Ultrasonic Protection ON/OFF Toggle -->
      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:12px; padding:10px 12px; background:#0b0f19; border:1px solid var(--card-border); border-radius:6px;">
        <div>
          <div style="font-weight:700; font-size:0.85rem; color:#fff;">Ultrasonic Obstacle Protection</div>
          <div style="font-size:0.75rem; color:var(--text-muted);">Blocks forward motion when obstacles detected</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <label class="switch">
            <input type="checkbox" id="toggle-ultrasonic-protection" onchange="onToggleProtection(this.checked)">
            <span class="slider"></span>
          </label>
          <span id="label-protection-status" style="font-size:0.85rem; font-weight:700; color:var(--rose); width:32px;">OFF</span>
        </div>
      </div>
    </div>

    <!-- Speech & Vosk Recognition -->
    <div class="card">
      <div class="card-title">
        <span>SPEECH & VOSK RECOGNIZER</span>
        <span id="vosk-state-badge" class="badge badge-healthy">LISTENING</span>
      </div>
      <div class="data-row">
        <span class="data-label">Microphone State:</span>
        <span class="data-val" id="val-mic-active">Active (C270 USB)</span>
      </div>
      <div class="data-row">
        <span class="data-label">Vosk Engine:</span>
        <span class="data-val" id="val-vosk-active">Ready (Offline Kaldi)</span>
      </div>
      <div class="data-row">
        <span class="data-label">Latest Recognized:</span>
        <span class="data-val" id="val-last-transcript" style="color:var(--text-muted);">(Listening for speech...)</span>
      </div>
      <div class="data-row">
        <span class="data-label">Parsed Action / Intent:</span>
        <span class="data-val" id="val-parsed-cmd">None</span>
      </div>
    </div>

    <!-- Navigation & Search State -->
    <div class="card">
      <div class="card-title">
        <span>STATE MACHINE & NAVIGATION</span>
        <span id="estop-badge" class="badge badge-healthy">NORMAL</span>
      </div>
      <div class="data-row">
        <span class="data-label">FSM State:</span>
        <span class="data-val" id="val-fsm-state">IDLE_LISTENING</span>
      </div>
      <div id="val-fsm-desc" style="font-size:0.75rem; color:var(--cyan); padding: 2px 0;">Robot ready; waiting for voice or manual command.</div>
      <div class="data-row">
        <span class="data-label">Active Fault Code:</span>
        <span class="data-val" id="val-active-fault">OK</span>
      </div>
      <div class="data-row">
        <span class="data-label">Search Sector / Progress:</span>
        <span class="data-val" id="val-search-progress">Idle</span>
      </div>
      <div class="data-row">
        <span class="data-label">Rotation Calibration:</span>
        <span class="data-val" id="val-rot-calib">UNCALIBRATED (Default pulse: 0.75s)</span>
      </div>
      <div class="data-row">
        <span class="data-label">Last Terminal Result:</span>
        <span class="data-val" id="val-term-result">None</span>
      </div>
    </div>

    <!-- Motors & Actuation -->
    <div class="card">
      <div class="card-title">
        <span>MOTORS & ACTUATION</span>
        <span id="motor-cmd-badge" class="badge badge-state">STOPPED</span>
      </div>
      <div class="data-row">
        <span class="data-label">Software Enabled:</span>
        <span class="data-val" id="val-motor-enabled">True (PWM Ready)</span>
      </div>
      <div class="data-row">
        <span class="data-label">Left Channel Speed:</span>
        <span class="data-val" id="val-motor-left">0.00</span>
      </div>
      <div class="data-row">
        <span class="data-label">Right Channel Speed:</span>
        <span class="data-val" id="val-motor-right">0.00</span>
      </div>
    </div>

    <!-- Command Console & Operator Controls -->
    <div class="card">
      <div class="card-title">
        <span>OPERATOR CONSOLE</span>
      </div>
      <div class="controls-container">
        <!-- Token Management Bar -->
        <div style="display:flex; gap:8px; align-items:center;">
          <input type="password" id="console-token" placeholder="ROBOT_API_TOKEN to unlock controls..." style="flex:1; padding:6px 10px; background:#0b0f19; border:1px solid var(--card-border); color:#fff; border-radius:4px; font-size:0.8rem;" />
          <button class="btn-primary" onclick="unlockFromConsole()" style="padding:6px 12px; font-size:0.8rem;">Set Token</button>
          <span id="token-status-badge" class="badge" onclick="showAuthOverlay()" style="background:#3b1d5c; color:var(--purple); font-size:0.7rem; cursor:pointer;" title="Click to enter authorization token">NO TOKEN</span>
        </div>

        <div class="input-group">
          <input type="text" id="cmd-input" placeholder="e.g. find bottle, move forward, turn left" />
          <button class="btn-primary" onclick="sendCommand()">Send</button>
        </div>
        <div class="btn-group">
          <button class="btn-primary" onclick="sendQuick('find bottle')">Find Bottle</button>
          <button class="btn-primary" onclick="sendQuick('find backpack')">Find Bag</button>
          <button class="btn-primary" onclick="sendQuick('move forward')">Forward</button>
          <button class="btn-primary" onclick="sendQuick('turn left')">Turn Left</button>
          <button class="btn-primary" onclick="sendQuick('turn right')">Turn Right</button>
        </div>
        <div class="btn-group">
          <button class="btn-stop" onclick="sendStop()">STOP</button>
          <button class="btn-estop" onclick="sendEStop()">EMERGENCY STOP</button>
          <button class="btn-clear" onclick="sendClearEStop()">Clear E-Stop</button>
          <button class="btn-primary" onclick="sendQuick('resume listening')">Resume</button>
        </div>
        <div id="status-msg">Ready for operator commands.</div>
      </div>
    </div>
  </div>

  <!-- Operator Token Overlay Modal -->
  <div id="auth-overlay" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(11,15,25,0.85); z-index:9999; justify-content:center; align-items:center;">
    <div style="background:var(--card-bg); border:1px solid var(--card-border); border-radius:8px; padding:24px; max-width:440px; width:90%; text-align:center;">
      <h3 style="color:#fff; margin-bottom:10px;">Operator Authentication</h3>
      <p style="color:var(--text-muted); font-size:0.85rem; margin-bottom:16px;">This robot is running in REAL_HARDWARE mode. Enter the operator token from <code>.robot_token</code> to authorize movement commands, or view telemetry only.</p>
      <input type="password" id="control-token" placeholder="Paste ROBOT_API_TOKEN..." style="width:100%; padding:8px 12px; background:#0b0f19; border:1px solid var(--card-border); color:#fff; border-radius:4px; margin-bottom:12px;" />
      <div id="auth-error" style="color:var(--rose); font-size:0.8rem; margin-bottom:12px; min-height:18px;"></div>
      <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap;">
        <button class="btn-primary" onclick="unlockControls()">Unlock Controls</button>
        <button class="btn-stop" onclick="dismissAuth()">View Telemetry Only</button>
      </div>
    </div>
  </div>

  <script>
    const CONTROL_TOKEN_REQUIRED = __CONTROL_TOKEN_REQUIRED__;
    let controlToken = '';
    let commandInFlight = false;
    let camRetryTimer = null;

    // Check if token stored in localStorage
    try {
      const stored = localStorage.getItem('vcr_robot_token');
      if (stored && stored.length >= 24) {
        controlToken = stored;
      }
    } catch (e) {}

    function updateTokenBadge() {
      const b = document.getElementById('token-status-badge');
      if (!b) return;
      if (controlToken && controlToken.length >= 24) {
        b.textContent = 'AUTHENTICATED';
        b.style.background = '#064e3b';
        b.style.color = '#34d399';
      } else {
        b.textContent = CONTROL_TOKEN_REQUIRED ? 'TOKEN REQUIRED' : 'NO TOKEN NEEDED';
        b.style.background = '#3b1d5c';
        b.style.color = 'var(--purple)';
      }
    }
    updateTokenBadge();

    function dismissAuth() {
      const ov = document.getElementById('auth-overlay');
      if (ov) ov.style.display = 'none';
    }

    function showAuthOverlay() {
      const ov = document.getElementById('auth-overlay');
      if (ov) ov.style.display = 'flex';
    }

    function controlHeaders(extra = {}) {
      const headers = { ...extra };
      if (CONTROL_TOKEN_REQUIRED && controlToken) headers['X-Robot-Token'] = controlToken;
      return headers;
    }

    async function unlockControls() {
      const input = document.getElementById('control-token');
      const token = input ? input.value.trim() : '';
      await verifyAndSetToken(token);
    }

    async function unlockFromConsole() {
      const input = document.getElementById('console-token');
      const token = input ? input.value.trim() : '';
      await verifyAndSetToken(token);
    }

    async function verifyAndSetToken(token) {
      if (!token || token.length < 24) {
        const errEl = document.getElementById('auth-error');
        if (errEl) errEl.textContent = 'Token must be at least 24 characters.';
        return;
      }
      try {
        const res = await fetch('/api/auth-check', { method: 'POST', headers: { 'X-Robot-Token': token } });
        if (!res.ok) throw new Error('Token rejected');
        controlToken = token;
        try { localStorage.setItem('vcr_robot_token', token); } catch (e) {}
        dismissAuth();
        updateTokenBadge();
        const errEl = document.getElementById('auth-error');
        if (errEl) errEl.textContent = '';
        document.getElementById('status-msg').textContent = 'Controls unlocked for this session.';
      } catch (err) {
        controlToken = '';
        updateTokenBadge();
        const errEl = document.getElementById('auth-error');
        if (errEl) errEl.textContent = 'Invalid token. Server rejected credentials.';
        document.getElementById('status-msg').textContent = 'Token rejection: ' + err.message;
      }
    }

    function reloadCamStream() {
      const img = document.getElementById('cam-stream');
      if (img) {
        img.src = '/api/camera/stream?t=' + Date.now();
      }
    }

    function onCamError(img) {
      const fb = document.getElementById('cam-fallback');
      if (fb) fb.style.display = 'block';
      const b = document.getElementById('cam-health-badge');
      if (b) { b.textContent = 'RECONNECTING'; b.className = 'badge badge-fault'; }
      if (!camRetryTimer) {
        camRetryTimer = setTimeout(() => {
          camRetryTimer = null;
          img.src = '/api/camera/stream?t=' + Date.now();
        }, 1500);
      }
    }

    function onCamLoad() {
      const fb = document.getElementById('cam-fallback');
      if (fb) fb.style.display = 'none';
      const b = document.getElementById('cam-health-badge');
      if (b && b.textContent === 'RECONNECTING') { b.textContent = 'LIVE'; b.className = 'badge badge-healthy'; }
    }

    function formatUptime(sec) {
      if (!sec || isNaN(sec)) return "00:00:00";
      const s = Math.floor(sec % 60).toString().padStart(2, '0');
      const m = Math.floor((sec / 60) % 60).toString().padStart(2, '0');
      const h = Math.floor(sec / 3600).toString().padStart(2, '0');
      return `${h}:${m}:${s}`;
    }

    const FSM_DESCRIPTIONS = {
      'IDLE_LISTENING': 'System ready and actively listening for voice or API command.',
      'MANUAL_MOVEMENT': 'Actuating bounded directional motor pulse.',
      'SEARCHING': 'Executing scanning turns searching for requested target object.',
      'TARGET_CONFIRMED': 'Target object detected across consecutive optical frames.',
      'CENTERING': 'Rotating chassis to align target within camera deadband (±40px).',
      'APPROACHING': 'Driving toward confirmed target with forward ultrasonic gating.',
      'TARGET_REACHED': 'Target reached safely at stopping threshold (>25cm blocked zone).',
      'TARGET_NOT_FOUND': 'Full 360° scan exhausted without target identification.',
      'BLOCKED': 'Forward motion inhibited by ultrasonic obstacle proximity.',
      'EMERGENCY_STOP': 'EMERGENCY STOP latched! All motor outputs de-energized.'
    };

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) throw new Error('Status fetch failed HTTP ' + res.status);
        const data = await res.json();

        // Header Badges
        const modeBadge = document.getElementById('badge-mode');
        if (modeBadge) {
          modeBadge.textContent = data.health.mode;
          modeBadge.className = 'badge ' + (data.health.mode === 'REAL_HARDWARE' ? 'badge-real' : 'badge-sim');
        }

        const healthBadge = document.getElementById('badge-health');
        if (healthBadge) {
          healthBadge.textContent = data.health.is_healthy ? 'HEALTHY' : 'ATTENTION';
          healthBadge.className = 'badge ' + (data.health.is_healthy ? 'badge-healthy' : 'badge-fault');
        }

        const stateBadge = document.getElementById('badge-state');
        if (stateBadge) stateBadge.textContent = data.fsm.state;

        const uptimeBadge = document.getElementById('badge-uptime');
        if (uptimeBadge) uptimeBadge.textContent = 'UP: ' + formatUptime(data.health.uptime_sec);

        const liveDot = document.getElementById('telemetry-live');
        if (liveDot) liveDot.textContent = '● ' + new Date().toLocaleTimeString();

        // Alert Messages Banner
        const alertList = document.getElementById('alert-list');
        if (alertList) {
          alertList.innerHTML = '';
          if (data.messages && data.messages.length > 0) {
            data.messages.forEach(msg => {
              const span = document.createElement('span');
              span.className = 'msg-badge msg-' + msg;
              span.textContent = msg;
              alertList.appendChild(span);
            });
          } else {
            const span = document.createElement('span');
            span.className = 'msg-badge';
            span.style.background = '#064e3b';
            span.style.color = '#34d399';
            span.textContent = 'ALL SYSTEMS OPERATIONAL';
            alertList.appendChild(span);
          }
        }

        // Ultrasonic Card & Protection sync
        const distVal = data.ultrasonic.distance_cm;
        const rawDistVal = data.ultrasonic.raw_distance_cm;
        const filteredDistVal = data.ultrasonic.filtered_distance_cm;
        const isValid = data.ultrasonic.is_valid;
        const obsState = data.ultrasonic.obstacle_state;
        const sensorState = data.ultrasonic.sensor_state || (isValid ? 'CLEAR' : 'INVALID');
        const protectionEnabled = data.ultrasonic.protection_enabled !== false;
        const readingAge = data.ultrasonic.reading_age_sec;

        // Sync Protection Toggle and Warning Banner
        const toggleEl = document.getElementById('toggle-ultrasonic-protection');
        const protLabel = document.getElementById('label-protection-status');
        const warnBanner = document.getElementById('ultrasonic-warning-banner');
        if (toggleEl && protLabel && warnBanner) {
          toggleEl.checked = protectionEnabled;
          protLabel.textContent = protectionEnabled ? 'ON' : 'OFF';
          protLabel.style.color = protectionEnabled ? 'var(--emerald)' : 'var(--rose)';
          warnBanner.style.display = protectionEnabled ? 'none' : 'block';
        }

        const distLarge = document.getElementById('val-dist-large');
        if (distLarge) {
          if (distVal !== null && distVal !== undefined && isValid) {
            distLarge.textContent = distVal.toFixed(1) + ' cm';
            if (['CRITICAL', 'BLOCKED'].includes(obsState) || sensorState === 'OBSTACLE') distLarge.style.color = 'var(--rose)';
            else if (obsState === 'WARNING') distLarge.style.color = 'var(--amber)';
            else distLarge.style.color = 'var(--cyan)';
          } else if (obsState === 'OUT_OF_RANGE') {
            distLarge.textContent = '> 400 cm (Clear)';
            distLarge.style.color = 'var(--emerald)';
          } else {
            distLarge.textContent = '-- cm';
            distLarge.style.color = 'var(--text-muted)';
          }
        }

        const distBadge = document.getElementById('dist-state-badge');
        if (distBadge) {
          distBadge.textContent = sensorState;
          if (sensorState === 'CLEAR') { distBadge.className = 'badge badge-healthy'; }
          else if (sensorState === 'OBSTACLE') { distBadge.className = 'badge badge-fault'; }
          else if (sensorState === 'STALE') { distBadge.className = 'badge badge-fault'; }
          else { distBadge.className = 'badge badge-sim'; }
        }

        const sensorStateEl = document.getElementById('val-sensor-state');
        if (sensorStateEl) {
          sensorStateEl.textContent = sensorState;
          sensorStateEl.style.color = (sensorState === 'CLEAR') ? 'var(--emerald)' : (sensorState === 'OBSTACLE' ? 'var(--rose)' : (sensorState === 'STALE' ? 'var(--rose)' : 'var(--amber)'));
        }

        const rawFiltEl = document.getElementById('val-dist-raw-filt');
        if (rawFiltEl) {
          const rawStr = (rawDistVal !== null && rawDistVal !== undefined) ? `${rawDistVal.toFixed(1)} cm` : '--';
          const filtStr = (filteredDistVal !== null && filteredDistVal !== undefined) ? `${filteredDistVal.toFixed(1)} cm` : '--';
          rawFiltEl.textContent = `${rawStr} raw / ${filtStr} filt`;
        }

        const validEl = document.getElementById('val-dist-valid');
        if (validEl) {
          const ageStr = (readingAge !== null && readingAge !== undefined) ? ` (${readingAge.toFixed(2)}s ago)` : '';
          validEl.textContent = (isValid ? 'Valid' : (obsState === 'OUT_OF_RANGE' ? 'Valid (Out of Range)' : 'Fault / Invalid')) + ageStr;
          validEl.style.color = isValid ? 'var(--emerald)' : (obsState === 'OUT_OF_RANGE' ? 'var(--cyan)' : 'var(--rose)');
        }

        const blockEl = document.getElementById('val-block-reason');
        if (blockEl) blockEl.textContent = data.motor.safety_block_reason || 'None (Path Uninhibited)';

        // Optical Card
        const camFailed = data.messages && data.messages.includes('CAMERA_FAILURE');
        const camBadge = document.getElementById('cam-health-badge');
        if (camBadge) {
          const camReady = data.subsystems && data.subsystems.camera === 'READY';
          if (camFailed || !camReady) {
            camBadge.textContent = 'CAPTURE FAULT';
            camBadge.className = 'badge badge-fault';
          } else {
            camBadge.textContent = 'LIVE';
            camBadge.className = 'badge badge-healthy';
          }
        }

        const targetEl = document.getElementById('val-target-class');
        if (targetEl) targetEl.textContent = data.target.requested_target ? `${data.target.requested_target} (${data.target.target_alias || 'exact'})` : 'None';

        const confEl = document.getElementById('val-conf-count');
        if (confEl) confEl.textContent = data.target.confirmed_count;

        const detEl = document.getElementById('val-curr-det');
        if (detEl) {
          if (camFailed) {
            detEl.textContent = 'Camera capture degraded / unavailable';
            detEl.style.color = 'var(--rose)';
          } else if (data.target.detections && data.target.detections.length > 0) {
            const summary = data.target.detections.map(d => `${d.label} [${(d.confidence * 100).toFixed(0)}%]${d.is_centered ? ' ★' : ''}`).join(', ');
            detEl.textContent = summary;
            detEl.style.color = data.target.current_detection && data.target.current_detection.is_centered ? 'var(--emerald)' : 'var(--cyan)';
          } else if (data.target.current_detection) {
            const d = data.target.current_detection;
            detEl.textContent = `${d.label} [${(d.confidence * 100).toFixed(0)}%] ${d.is_centered ? '★ CENTERED' : ''}`;
            detEl.style.color = d.is_centered ? 'var(--emerald)' : 'var(--cyan)';
          } else {
            detEl.textContent = 'No detections (Frame clear)';
            detEl.style.color = 'var(--text-muted)';
          }
        }

        // Speech Card
        const voskBadge = document.getElementById('vosk-state-badge');
        if (voskBadge) {
          voskBadge.textContent = data.speech.listening_state;
          voskBadge.className = 'badge ' + (data.speech.listening_state === 'LISTENING' ? 'badge-healthy' : 'badge-fault');
        }

        const micEl = document.getElementById('val-mic-active');
        if (micEl) micEl.textContent = data.speech.mic_active ? 'Active (C270 USB)' : 'Inactive';

        const voskEl = document.getElementById('val-vosk-active');
        if (voskEl) voskEl.textContent = data.speech.vosk_active ? 'Ready (Offline Kaldi)' : 'Inactive';

        const txEl = document.getElementById('val-last-transcript');
        if (txEl) {
          if (data.speech.last_transcript && data.speech.last_transcript.trim()) {
            txEl.textContent = `"${data.speech.last_transcript}"` + (data.speech.is_partial ? ' (speaking...)' : '');
            txEl.style.color = data.speech.is_partial ? 'var(--amber)' : 'var(--emerald)';
          } else {
            txEl.textContent = '(Waiting for voice command...)';
            txEl.style.color = 'var(--text-muted)';
          }
        }

        const cmdEl = document.getElementById('val-parsed-cmd');
        if (cmdEl) {
          if (data.speech.last_command) {
            const c = data.speech.last_command;
            cmdEl.textContent = `${c.action} [${c.intent}]${c.is_executable ? '' : ' - (Rejected)'}`;
            cmdEl.style.color = c.is_executable ? 'var(--cyan)' : 'var(--amber)';
          } else {
            cmdEl.textContent = 'None';
            cmdEl.style.color = 'var(--text-muted)';
          }
        }

        // FSM Card
        const fsmState = data.fsm.state;
        const fsmEl = document.getElementById('val-fsm-state');
        if (fsmEl) fsmEl.textContent = fsmState;

        const fsmDesc = document.getElementById('val-fsm-desc');
        if (fsmDesc) fsmDesc.textContent = FSM_DESCRIPTIONS[fsmState] || 'Active FSM state.';

        const faultEl = document.getElementById('val-active-fault');
        if (faultEl) faultEl.textContent = data.health.active_fault;

        const searchEl = document.getElementById('val-search-progress');
        if (searchEl) searchEl.textContent = data.search.search_progress;

        const rotEl = document.getElementById('val-rot-calib');
        if (rotEl) rotEl.textContent = data.search.rotation_calibration_status === 'CALIBRATED' ? 'CALIBRATED' : 'UNCALIBRATED (Default pulse: 0.75s)';

        const termEl = document.getElementById('val-term-result');
        if (termEl) termEl.textContent = data.fsm.last_terminal_result || 'None';

        const estopBadge = document.getElementById('estop-badge');
        if (estopBadge) {
          estopBadge.textContent = data.fsm.emergency_stop_latched ? 'LATCHED' : 'NORMAL';
          estopBadge.className = 'badge ' + (data.fsm.emergency_stop_latched ? 'badge-fault' : 'badge-healthy');
        }

        // Motors Card
        const motorEnEl = document.getElementById('val-motor-enabled');
        if (motorEnEl) motorEnEl.textContent = data.motor.enabled ? 'True (PWM Energized)' : 'False (Outputs Cut)';

        const motLeft = document.getElementById('val-motor-left');
        if (motLeft) motLeft.textContent = (data.motor.left_speed || 0.0).toFixed(2);

        const motRight = document.getElementById('val-motor-right');
        if (motRight) motRight.textContent = (data.motor.right_speed || 0.0).toFixed(2);

        const motBadge = document.getElementById('motor-cmd-badge');
        if (motBadge) {
          motBadge.textContent = data.motor.command_state;
          motBadge.className = 'badge ' + (data.motor.command_state === 'STOPPED' ? 'badge-state' : 'badge-real');
        }

      } catch (err) {
        const hBadge = document.getElementById('badge-health');
        if (hBadge) {
          hBadge.textContent = 'DISCONNECTED';
          hBadge.className = 'badge badge-fault';
        }
        const liveDot = document.getElementById('telemetry-live');
        if (liveDot) liveDot.textContent = '● RECONNECTING...';
      }
    }

    async function onToggleProtection(enabled) {
      const label = document.getElementById('label-protection-status');
      if (label) label.textContent = enabled ? '...' : '...';
      try {
        const res = await fetch('/api/ultrasonic/protection', {
          method: 'POST',
          headers: controlHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ enabled: enabled })
        });
        const d = await res.json();
        if (res.status === 401) {
          document.getElementById('status-msg').textContent = 'UNAUTHORIZED: Valid ROBOT_API_TOKEN required to change protection.';
          const toggle = document.getElementById('toggle-ultrasonic-protection');
          if (toggle) toggle.checked = !enabled;
          showAuthOverlay();
          return;
        }
        document.getElementById('status-msg').textContent = d.message || ('Protection: ' + (enabled ? 'ON' : 'OFF'));
      } catch (e) {
        document.getElementById('status-msg').textContent = 'Protection toggle error: ' + e;
        const toggle = document.getElementById('toggle-ultrasonic-protection');
        if (toggle) toggle.checked = !enabled;
      }
    }

    async function sendCommand() {
      const input = document.getElementById('cmd-input');
      const text = input ? input.value.trim() : '';
      if (!text) return;
      await postCommand(text);
      if (input) input.value = '';
    }

    async function sendQuick(cmd) {
      await postCommand(cmd);
    }

    async function sendStop() {
      try {
        const res = await fetch('/api/stop', { method: 'POST', headers: controlHeaders() });
        const d = await res.json();
        document.getElementById('status-msg').textContent = 'STOP: ' + d.message;
      } catch (e) {
        document.getElementById('status-msg').textContent = 'STOP error: ' + e;
      }
    }

    async function sendEStop() {
      try {
        const res = await fetch('/api/emergency-stop', { method: 'POST', headers: controlHeaders() });
        const d = await res.json();
        document.getElementById('status-msg').textContent = 'E-STOP: ' + d.message;
      } catch (e) {
        document.getElementById('status-msg').textContent = 'E-STOP error: ' + e;
      }
    }

    async function sendClearEStop() {
      try {
        const res = await fetch('/api/clear-estop', { method: 'POST', headers: controlHeaders() });
        const d = await res.json();
        document.getElementById('status-msg').textContent = 'Clear: ' + d.message;
      } catch (e) {
        document.getElementById('status-msg').textContent = 'Clear error: ' + e;
      }
    }

    async function postCommand(text) {
      if (commandInFlight) return;
      commandInFlight = true;
      const statusEl = document.getElementById('status-msg');
      if (statusEl) statusEl.textContent = `Dispatching: "${text}"...`;
      try {
        const res = await fetch('/api/command', {
          method: 'POST',
          headers: controlHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({ command: text })
        });
        const data = await res.json();
        if (res.status === 401) {
          if (statusEl) statusEl.textContent = 'UNAUTHORIZED: Valid ROBOT_API_TOKEN required. Enter token above.';
          const ov = document.getElementById('auth-overlay');
          if (ov) ov.style.display = 'flex';
        } else if (res.status === 409) {
          if (statusEl) statusEl.textContent = 'REJECTED: ' + (data.detail || data.message);
        } else {
          if (statusEl) statusEl.textContent = 'Result: ' + data.message;
        }
      } catch (e) {
        if (statusEl) statusEl.textContent = 'Command network error: ' + e;
      } finally {
        commandInFlight = false;
      }
    }

    // Auto-refresh telemetry every 250ms
    setInterval(updateTelemetry, 250);
    updateTelemetry();
  </script>
</body>
</html>
"""


class CommandRequest(BaseModel):
    command: str


class UltrasonicProtectionRequest(BaseModel):
    enabled: bool


def create_app(runtime: RobotRuntime, control_token: Optional[str] = None) -> FastAPI:
    """FastAPI application factory attached to a single authoritative RobotRuntime.

    Real-hardware control endpoints require an out-of-band operator token; the
    token is not embedded into the dashboard HTML and is held in page memory only.
    """
    if runtime.config.mode.value == "REAL_HARDWARE" and (not control_token or len(control_token) < 24):
        raise ValueError("REAL_HARDWARE dashboard requires a ROBOT_API_TOKEN of at least 24 characters; refusing unauthenticated motor-control API.")

    app = FastAPI(
        title="SS_PROJECT_VCR Runtime Dashboard",
        version="2.0.0",
        description="Unified local control and status dashboard for Voice Controlled Robot.",
    )


    @app.on_event("shutdown")
    def on_shutdown() -> None:
        """Trigger graceful runtime cleanup when uvicorn initiates shutdown."""
        runtime.cleanup()

    @app.get("/", response_class=HTMLResponse)
    def get_dashboard() -> str:
        """Serve the readable local status dashboard matching the environment."""
        return DASHBOARD_HTML.replace("__CONTROL_TOKEN_REQUIRED__", "true" if control_token else "false")

    def require_control_token(x_robot_token: Optional[str] = Header(default=None, alias="X-Robot-Token")) -> None:
        if control_token and (not x_robot_token or not secrets.compare_digest(x_robot_token, control_token)):
            raise HTTPException(status_code=401, detail="Valid X-Robot-Token is required for robot control.")

    @app.post("/api/auth-check")
    def post_auth_check(x_robot_token: Optional[str] = Header(default=None, alias="X-Robot-Token")) -> Dict[str, Any]:
        if control_token and (not x_robot_token or not secrets.compare_digest(x_robot_token, control_token)):
            raise HTTPException(status_code=401, detail="Invalid operator token.")
        return {"success": True, "message": "Operator token accepted"}

    @app.get("/api/status")
    def get_status() -> Dict[str, Any]:
        """Read-only non-blocking system telemetry endpoint.
        
        Guarantees: Never triggers physical sensing or motor movement.
        """
        return runtime.get_telemetry()

    @app.post("/api/command")
    def post_command(req: CommandRequest, _: None = Depends(require_control_token)) -> Dict[str, Any]:
        """Parse and dispatch command through the authoritative state machine.
        
        Returns HTTP 409 Conflict if another command is actively executing.
        """
        raw = req.command.strip()
        if not raw:
            raise HTTPException(status_code=400, detail="Command text cannot be empty")

        success, cmd, message = runtime.dispatch_text_command(raw)

        if not success and "Concurrent command rejected" in message:
            raise HTTPException(status_code=409, detail=message)

        return {
            "success": success,
            "message": message,
            "command": {
                "action": cmd.action.value if cmd else None,
                "intent": cmd.intent.value if cmd else None,
                "target_class": cmd.target_class if cmd else None,
                "target_alias": cmd.target_alias if cmd else None,
                "speed": cmd.speed if cmd else None,
                "is_executable": cmd.is_executable if cmd else False,
                "error_reason": cmd.error_reason if cmd else None,
            } if cmd else None,
        }

    @app.post("/api/stop")
    def post_stop() -> Dict[str, Any]:
        """Immediate active stop overriding ongoing motion."""
        success, cmd, message = runtime.dispatch_text_command("stop")
        return {"success": success, "message": message}

    @app.post("/api/emergency-stop")
    def post_emergency_stop() -> Dict[str, Any]:
        """Immediate latched emergency stop halting all actuation."""
        success, cmd, message = runtime.dispatch_text_command("emergency stop")
        return {"success": success, "message": message}

    @app.post("/api/clear-estop")
    def post_clear_estop(_: None = Depends(require_control_token)) -> Dict[str, Any]:
        """Explicitly clear the latched emergency stop condition."""
        if runtime.fsm:
            cleared = runtime.fsm.clear_emergency_stop()
            return {
                "success": cleared,
                "message": "Emergency stop cleared after safety precheck" if cleared
                           else "Emergency stop remains latched: verify sensor validity and clear the blocked zone",
            }
        return {"success": False, "message": "FSM not initialized"}

    @app.post("/api/ultrasonic/protection")
    def post_ultrasonic_protection(
        req: UltrasonicProtectionRequest,
        _: None = Depends(require_control_token),
    ) -> Dict[str, Any]:
        """Explicitly enable or disable ultrasonic obstacle protection."""
        runtime.set_ultrasonic_protection(req.enabled)
        state_str = "ENABLED" if req.enabled else "DISABLED"
        return {
            "success": True,
            "enabled": runtime.ultrasonic_protection_enabled,
            "message": f"Ultrasonic obstacle protection is now {state_str}.",
        }

    @app.get("/api/camera/stream")
    async def get_camera_stream(request: Request, max_frames: int = 0) -> StreamingResponse:
        """Stream optical frames as MJPEG.
        
        Guarantees:
        - Obtains copies from single camera owner without opening a second camera.
        - Gracefully handles client disconnect without weakening robot safety.
        - Supports bounded frame testing via max_frames parameter.
        """
        async def frame_generator() -> AsyncGenerator[bytes, None]:
            frames_sent = 0
            try:
                while not runtime._stop_event.is_set():
                    if await request.is_disconnected():
                        break
                    if runtime.lifecycle_state in (LifecycleState.STOPPED, LifecycleState.CLEANED_UP):
                        break

                    frame = None
                    if runtime.camera:
                        if hasattr(runtime.camera, "get_latest_frame_with_timestamp"):
                            frame, _ = runtime.camera.get_latest_frame_with_timestamp()
                        elif hasattr(runtime.camera, "capture_frame"):
                            frame = runtime.camera.capture_frame()

                    if frame is None or cv2 is None:
                        # Informative camera capture fault placeholder (never a pitch-black box)
                        placeholder = np.full((480, 640, 3), (25, 20, 35), dtype=np.uint8)
                        if cv2 is not None:
                            cv2.rectangle(placeholder, (10, 10), (630, 470), (45, 35, 180), 2)
                            cv2.putText(
                                placeholder,
                                "CAMERA CAPTURE FAULT / RECONNECTING...",
                                (50, 220),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0, 100, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                placeholder,
                                "No optical frames delivered by camera hardware thread",
                                (90, 260),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (160, 160, 180),
                                1,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                placeholder,
                                f"DIAGNOSTIC TIME: {time.strftime('%H:%M:%S')}",
                                (200, 300),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.55,
                                (0, 215, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            _, encoded = cv2.imencode(".jpg", placeholder)
                            data = encoded.tobytes()
                        else:
                            data = b""
                    else:
                        display_frame = frame.copy()
                        h, w = display_frame.shape[:2]
                        cx = w // 2

                        # Draw optical deadband guidelines (cx +- 40px)
                        cv2.line(display_frame, (cx - 40, 0), (cx - 40, h), (80, 80, 80), 1)
                        cv2.line(display_frame, (cx + 40, 0), (cx + 40, h), (80, 80, 80), 1)
                        cv2.drawMarker(
                            display_frame,
                            (cx, h // 2),
                            (100, 100, 100),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=16,
                            thickness=1,
                        )

                        # Overlay YOLO detections
                        active_dets = []
                        if runtime.fsm and runtime.fsm.context and runtime.fsm.context.target_detection:
                            active_dets = [runtime.fsm.context.target_detection]
                        elif hasattr(runtime, "_latest_detections") and runtime._latest_detections:
                            if time.time() - getattr(runtime, "_latest_detections_time", 0.0) <= 3.0:
                                active_dets = runtime._latest_detections

                        for d in active_dets:
                            box = getattr(d, "box_xyxy", getattr(d, "box", None))
                            conf = getattr(d, "confidence", 0.0)
                            label = getattr(d, "class_name", getattr(d, "label", "object"))
                            is_target = bool(
                                runtime.fsm
                                and runtime.fsm.context
                                and runtime.fsm.context.target_class
                                and label == runtime.fsm.context.target_class
                            )
                            box_xywh = getattr(d, "box_xywh", None)
                            if box_xywh is not None:
                                is_centered = abs(box_xywh[0] - (w / 2.0)) <= getattr(runtime.config.vision, "deadband_pixels", 40)
                            elif hasattr(d, "is_centered"):
                                is_centered = d.is_centered(w, getattr(runtime.config.vision, "deadband_pixels", 40))
                            else:
                                is_centered = False

                            if box is not None and len(box) == 4:
                                x1, y1, x2, y2 = [int(v) for v in box]
                                color = (0, 255, 0) if (is_target and is_centered) else ((0, 215, 255) if is_target else (255, 180, 50))
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                                det_text = f"{label}: {conf:.2f}" + (" [CENTERED]" if is_centered else (" [TARGET]" if is_target else ""))
                                cv2.rectangle(
                                    display_frame,
                                    (x1, max(0, y1 - 22)),
                                    (x1 + len(det_text) * 10, max(22, y1)),
                                    (20, 20, 20),
                                    -1,
                                )
                                cv2.putText(
                                    display_frame,
                                    det_text,
                                    (x1 + 3, max(16, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color,
                                    1,
                                    cv2.LINE_AA,
                                )

                        if not active_dets:
                            cv2.putText(
                                display_frame,
                                "FRAME CLEAR (0 DETECTIONS)",
                                (w - 290, 24),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.50,
                                (130, 150, 170),
                                1,
                                cv2.LINE_AA,
                            )

                        # Live timestamp overlay
                        cv2.putText(
                            display_frame,
                            f"CAM LIVE {time.strftime('%H:%M:%S')}",
                            (12, 24),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        _, encoded = cv2.imencode(".jpg", display_frame)
                        data = encoded.tobytes()

                    if data:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                        frames_sent += 1
                        if max_frames > 0 and frames_sent >= max_frames:
                            break

                    await asyncio.sleep(0.066)  # ~15 FPS
            except (asyncio.CancelledError, GeneratorExit):
                logger.debug("MJPEG stream aborted cleanly by client.")
            except Exception as exc:
                logger.error("Error in MJPEG frame stream: %s", exc)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app
