# Project spec — C. elegans chemotaxis agent

## Goal
A browser simulation of a single agent that finds a chemical source by
chemotaxis, driven by a small hand-wired neural network. No build step,
no npm, no frameworks. Plain ES modules loaded by index.html.

## World (world.js)
- 2D continuous space, 600x400 units.
- One source at (sx, sy). Concentration at point p = exp(-d^2 / (2*sigma^2)),
  d = distance from p to source, sigma = 120.
- Source can be moved at runtime.

## Agent body (agent.js)
- Position (x, y), heading theta, speed v.
- Two sensors placed at +/- 15 degrees from heading, 8 units ahead.
- Each sensor reads concentration from world.

## Brain (nn.js)
- Feedforward network, 2 inputs -> 3 hidden -> 2 outputs, tanh activations.
- Weights are fixed constants, NOT trained.
- Output 0 = turn rate, output 1 = forward speed multiplier.
- Sensory adaptation: each input is (current reading - running average of
  that sensor), running average with decay 0.95. This makes the agent
  respond to CHANGE, not absolute level.

## Behaviour
- Weathervane: continuous steering from the network output.
- Pirouette: if the running average of concentration has been falling for
  N consecutive steps, trigger a large random reorientation.
- Both must be individually switchable off (for ablation experiments).

## Simulation (sim.js)
- Fixed timestep loop. Records per-trial metrics: steps to reach source
  (within 20 units), final distance, mean distance over the trial.

## Benchmark (bench.js)
- Runs headless in node. N trials per condition, random start positions,
  fixed seed. Conditions: intact, no-pirouette, no-weathervane, ASEL
  ablated, sensor noise 0.1, moving source, and a random-walk control.
- Writes results.json with per-condition aggregates.

## UI (ui.js + index.html)
- Canvas rendering: concentration field as background, agent as a body
  with two visible sensors, trail of past positions.
- Controls: play/pause, reset, drag to move source, noise slider,
  checkboxes for each ablation, live readout of current metrics.

## Rules
- Every file is a standalone ES module with named exports.
- No external dependencies of any kind.
- bench.js must run under plain `node bench.js`.
