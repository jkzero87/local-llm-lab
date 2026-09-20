// ui.js — browser front-end for the C. elegans chemotaxis sim.
//
// Renders the concentration field (background), the agent (triangle + two
// sensor dots), its fading trail and the source (circle) on a 600x400 canvas.
// Wires the controls (play/pause, reset, noise slider, ablation checkboxes)
// to a live agent and repaints only the field when the source moves.
//
// Loaded as an ES module by index.html. No other dependencies, no DOM globals
// beyond the canvas and the control elements.

import { createWorld } from './world.js';
import { createBrain } from './nn.js';
import { createAgent } from './agent.js';

// ---------------------------------------------------------------------------
// DOM handles.
// ---------------------------------------------------------------------------
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const els = {
  play: document.getElementById('play'),
  reset: document.getElementById('reset'),
  noise: document.getElementById('noise'),
  noiseVal: document.getElementById('noiseVal'),
  pirouette: document.getElementById('pirouette'),
  weathervane: document.getElementById('weathervane'),
  ablateLeft: document.getElementById('ablateLeft'),
  ablateRight: document.getElementById('ablateRight'),
  steps: document.getElementById('steps'),
  dist: document.getElementById('dist'),
  conc: document.getElementById('conc'),
  piro: document.getElementById('piros'),
};

// ---------------------------------------------------------------------------
// Simulation state.
// ---------------------------------------------------------------------------
const START = { x: 100, y: 200, theta: 0 };
const STEPS_PER_FRAME = 3;

const world = createWorld({});
const brain = createBrain({});
const agent = createAgent({
  world,
  brain,
  x: START.x,
  y: START.y,
  theta: START.theta,
  noise: 0,
});

let playing = true;
let steps = 0;
let pirouettes = 0;

// ---------------------------------------------------------------------------
// Concentration field: sampled once into an offscreen canvas, repainted only
// when the source moves. Each grid cell is filled with a colour whose
// brightness is proportional to the local concentration.
// ---------------------------------------------------------------------------
const CELL = 5;
const field = document.createElement('canvas');
field.width = world.width;
field.height = world.height;
const fieldCtx = field.getContext('2d');

function paintField() {
  fieldCtx.fillStyle = '#05080c';
  fieldCtx.fillRect(0, 0, world.width, world.height);
  for (let cy = 0; cy < world.height; cy += CELL) {
    for (let cx = 0; cx < world.width; cx += CELL) {
      const c = world.concentration(cx + CELL / 2, cy + CELL / 2);
      const v = Math.round(255 * Math.pow(c, 0.85));
      fieldCtx.fillStyle =
        `rgb(${Math.round(v * 0.25)}, ${Math.round(v * 0.7)}, ${v})`;
      fieldCtx.fillRect(cx, cy, CELL, CELL);
    }
  }
}
paintField();

// ---------------------------------------------------------------------------
// Drawing.
// ---------------------------------------------------------------------------
function drawTrail() {
  const n = agent.trail.length;
  if (n < 2) return;
  ctx.lineWidth = 1;
  for (let i = 1; i < n; i++) {
    const a = i / n;
    ctx.strokeStyle = `rgba(140, 200, 255, ${(a * 0.45).toFixed(3)})`;
    ctx.beginPath();
    ctx.moveTo(agent.trail[i - 1].x, agent.trail[i - 1].y);
    ctx.lineTo(agent.trail[i].x, agent.trail[i].y);
    ctx.stroke();
  }
}

function drawAgent() {
  const { x, y, theta } = agent;
  const r = 7;
  ctx.fillStyle = '#ffd45e';
  ctx.beginPath();
  ctx.moveTo(x + Math.cos(theta) * r, y + Math.sin(theta) * r);
  ctx.lineTo(x + Math.cos(theta + 2.6) * r * 0.9, y + Math.sin(theta + 2.6) * r * 0.9);
  ctx.lineTo(x + Math.cos(theta - 2.6) * r * 0.9, y + Math.sin(theta - 2.6) * r * 0.9);
  ctx.closePath();
  ctx.fill();

  // Sensors at their real positions.
  const s = agent.sensors();
  ctx.fillStyle = '#6ee7ff';
  ctx.beginPath();
  ctx.arc(s.left.x, s.left.y, 2.2, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#ff8f8f';
  ctx.beginPath();
  ctx.arc(s.right.x, s.right.y, 2.2, 0, Math.PI * 2);
  ctx.fill();
}

function drawSource() {
  const { x, y } = world.source;
  ctx.fillStyle = '#ff5e5e';
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = 'rgba(255, 94, 94, 0.5)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 13, 0, Math.PI * 2);
  ctx.stroke();
}

function updateReadout() {
  const dx = agent.x - world.source.x;
  const dy = agent.y - world.source.y;
  els.steps.textContent = steps;
  els.dist.textContent = Math.hypot(dx, dy).toFixed(1);
  els.conc.textContent = world.concentration(agent.x, agent.y).toFixed(3);
  els.piro.textContent = pirouettes;
}

function draw() {
  ctx.drawImage(field, 0, 0);
  drawTrail();
  drawAgent();
  drawSource();
  updateReadout();
}

// ---------------------------------------------------------------------------
// Main loop: requestAnimationFrame, 3 simulation steps per frame.
// ---------------------------------------------------------------------------
function tick() {
  if (playing) {
    for (let i = 0; i < STEPS_PER_FRAME; i++) {
      const res = agent.step(1);
      steps += 1;
      if (res.pirouetted) pirouettes += 1;
    }
  }
  draw();
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------------------
// Controls.
// ---------------------------------------------------------------------------
els.play.addEventListener('click', () => {
  playing = !playing;
  els.play.textContent = playing ? 'Pause' : 'Play';
});

els.reset.addEventListener('click', () => {
  agent.reset({ x: START.x, y: START.y, theta: START.theta });
  steps = 0;
  pirouettes = 0;
  updateReadout();
});

els.noise.addEventListener('input', () => {
  const noise = parseFloat(els.noise.value);
  agent.setFlags({ noise });
  els.noiseVal.textContent = noise.toFixed(2);
});

function bindFlag(id, key) {
  els[id].addEventListener('change', () => {
    agent.setFlags({ [key]: els[id].checked });
  });
}
bindFlag('pirouette', 'pirouetteEnabled');
bindFlag('weathervane', 'weathervaneEnabled');
bindFlag('ablateLeft', 'ablateLeft');
bindFlag('ablateRight', 'ablateRight');

canvas.addEventListener('click', (ev) => {
  const rect = canvas.getBoundingClientRect();
  const x = (ev.clientX - rect.left) * (world.width / rect.width);
  const y = (ev.clientY - rect.top) * (world.height / rect.height);
  world.moveSource(x, y);
  paintField();
});

requestAnimationFrame(tick);
