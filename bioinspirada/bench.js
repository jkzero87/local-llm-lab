// bench.js — headless benchmark, plain `node bench.js`, no args.
//
// Seeded mulberry32 RNG (seed 12345) installed into agent.js via
// setRandom() so every run is reproducible. N trials per condition using
// the SAME 200 start states (generated once, replayed per condition).
// Writes results.json and prints a plain-text table to stdout.

import { writeFileSync } from 'node:fs';
import { createWorld } from './world.js';
import { createBrain } from './nn.js';
import { createAgent, setRandom } from './agent.js';
import { runTrial } from './sim.js';

const SEED = 12345;
const N = 200;

// mulberry32: tiny seeded PRNG; returns a function yielding floats in [0,1).
function mulberry32(seed) {
  let a = seed | 0;
  return function () {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = mulberry32(SEED);
setRandom(rng); // agent.js now draws every random value from this stream

// Generate the N shared start states ONCE; every condition replays this
// same array (uniform x in [20,200], y in [20,380], heading in [0,2π)).
const starts = Array.from({ length: N }, () => ({
  x: 20 + rng() * 180,
  y: 20 + rng() * 360,
  theta: rng() * 2 * Math.PI,
}));

const WORLD_W = 600;
const WORLD_H = 400;

// Condition list: exact keys, in fixed order. Each entry describes the
// agent flags for that condition and an optional per-step hook.
const conditions = [
  { name: 'intact', flags: {}, onStep: null },
  { name: 'no_pirouette', flags: { pirouetteEnabled: false }, onStep: null },
  { name: 'no_weathervane', flags: { weathervaneEnabled: false }, onStep: null },
  { name: 'ablate_ASEL', flags: { ablateLeft: true }, onStep: null },
  { name: 'noise_0.1', flags: { noise: 0.1 }, onStep: null },
  {
    // Defaults, but the source jumps to a new random location every 500
    // steps (before steps 500, 1000, 1500).
    name: 'moving_source',
    flags: {},
    onStep(world, i) {
      if ((i + 1) % 500 === 0) {
        world.moveSource(20 + rng() * (WORLD_W - 40), 20 + rng() * (WORLD_H - 40));
      }
    },
  },
  {
    // Control: no brain — fixed speed 0.5 (effective 1.0 with baseSpeed 2.0),
    // each step heading changes by uniform [-0.4, 0.4]. Same body/world/starts.
    name: 'random_walk',
    flags: {
      brain: { step: () => ({ turn: 0, speed: 0.5 }), reset() {} },
      pirouetteEnabled: false,
    },
    onStep(_world, _i, agent) {
      agent.theta += (rng() - 0.5) * 0.8;
    },
  },
];

function aggregate(trials) {
  const successes = trials.filter((t) => t.reached);
  const n = trials.length;
  return {
    successRate: successes.length / n,
    meanSteps:
      successes.length > 0
        ? successes.reduce((sum, t) => sum + t.steps, 0) / successes.length
        : null,
    meanFinalDistance: trials.reduce((sum, t) => sum + t.finalDistance, 0) / n,
    meanMeanDistance: trials.reduce((sum, t) => sum + t.meanDistance, 0) / n,
    n,
  };
}

const results = {};
for (const cond of conditions) {
  const trials = [];
  for (const s of starts) {
    const world = createWorld({});
    const agent = createAgent({
      world,
      brain: createBrain({}),
      x: s.x,
      y: s.y,
      theta: s.theta,
      ...cond.flags,
    });
    trials.push(
      runTrial({
        world,
        agent,
        onStep: cond.onStep ? (i, a) => cond.onStep(world, i, a) : null,
      }),
    );
  }
  results[cond.name] = aggregate(trials);
}

const output = {
  seed: SEED,
  n: N,
  generatedAt: new Date().toISOString(),
  conditions: results,
};
writeFileSync('results.json', JSON.stringify(output, null, 2) + '\n');

// Plain-text table, same numbers as results.json.
const header =
  'condition'.padEnd(17) +
  'success'.padEnd(10) +
  'meanSteps'.padEnd(11) +
  'meanFinalDist'.padEnd(14) +
  'meanMeanDist'.padEnd(14) +
  'n';
console.log(header);
console.log('-'.repeat(header.length));
for (const [name, c] of Object.entries(results)) {
  console.log(
    name.padEnd(17) +
    c.successRate.toFixed(3).padStart(10) +
    (c.meanSteps === null ? '-'.padStart(11) : c.meanSteps.toFixed(1).padStart(11)) +
    c.meanFinalDistance.toFixed(1).padStart(14) +
    c.meanMeanDistance.toFixed(1).padStart(14) +
    String(c.n).padStart(4),
  );
}
