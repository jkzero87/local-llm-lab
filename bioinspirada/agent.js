// agent.js — the body of the C. elegans chemotaxis agent.
//
// Couples a world (concentration field) and a brain (nn.js) into a steerable
// body: two sensors at ±sensorAngle from the heading, a weathervane steering
// term from the network's turn output, and a pirouette — a large random
// reorientation — triggered when the brain's temporal signal (mean reading
// below its running average) has been negative for N consecutive steps.
//
// No DOM. Every random call goes through the module-level rnd() so bench.js
// can inject a seeded generator via setRandom(fn).

// ---------------------------------------------------------------------------
// Single random source. Defaults to Math.random; swap it for a seeded
// generator at module scope (all agents created afterwards use it).
// ---------------------------------------------------------------------------
let randomFn = Math.random;
export function setRandom(fn) {
  randomFn = fn;
}
function rnd() {
  return randomFn();
}

// Create an agent. See spec.md for the behavioural contract.
export function createAgent(opts) {
  const { world, brain, x, y, theta } = opts;

  // Runtime-switchable option flags (setFlags merges into these).
  const flags = {
    sensorAngle: opts.sensorAngle ?? 0.26,
    sensorDist: opts.sensorDist ?? 8,
    baseSpeed: opts.baseSpeed ?? 2.0,
    turnGain: opts.turnGain ?? 0.35,
    noise: opts.noise ?? 0,
    pirouetteEnabled: opts.pirouetteEnabled ?? true,
    weathervaneEnabled: opts.weathervaneEnabled ?? true,
    ablateLeft: opts.ablateLeft ?? false,
    ablateRight: opts.ablateRight ?? false,
  };

  const TRAIL_CAP = 500;
  const PIROUETTE_THRESHOLD = 8;

  // Pirouette state: how many consecutive steps the brain's temporal signal
  // has been negative (mean reading below its running average).
  let fallingCount = 0;

  const agent = {
    x,
    y,
    theta,
    trail: [],

    // Sensor positions at theta ± sensorAngle, sensorDist ahead, with the
    // world's concentration sampled there. Ablation forces a reading to 0
    // first; noise (if any) is added afterwards.
    sensors() {
      const la = agent.theta + flags.sensorAngle;
      const ra = agent.theta - flags.sensorAngle;
      const lx = agent.x + Math.cos(la) * flags.sensorDist;
      const ly = agent.y + Math.sin(la) * flags.sensorDist;
      const rx = agent.x + Math.cos(ra) * flags.sensorDist;
      const ry = agent.y + Math.sin(ra) * flags.sensorDist;

      let leftValue = world.concentration(lx, ly);
      let rightValue = world.concentration(rx, ry);
      if (flags.ablateLeft) leftValue = 0;
      if (flags.ablateRight) rightValue = 0;
      if (flags.noise > 0) {
        leftValue += (rnd() - 0.5) * 2 * flags.noise;
        rightValue += (rnd() - 0.5) * 2 * flags.noise;
      }

      return {
        left: { x: lx, y: ly, value: leftValue },
        right: { x: rx, y: ry, value: rightValue },
      };
    },

    // One timestep. Returns { turn, speed, pirouetted }.
    step(dt) {
      // 1. Read sensors.
      const s = agent.sensors();

      // 2. Feed the brain.
      const { turn, speed, temporal } = brain.step(s.left.value, s.right.value);

      // 3. Weathervane steering (only when enabled).
      if (flags.weathervaneEnabled) {
        agent.theta += turn * flags.turnGain * dt;
      }

      // 4. Pirouette: count consecutive steps where the brain's temporal
      //    signal is negative; at the threshold, reorient to a uniform random
      //    direction.
      let pirouetted = false;
      if (temporal < 0) {
        fallingCount += 1;
      } else {
        fallingCount = 0;
      }
      if (flags.pirouetteEnabled && fallingCount >= PIROUETTE_THRESHOLD) {
        agent.theta = rnd() * 2 * Math.PI;
        fallingCount = 0;
        pirouetted = true;
      }

      // 5. Advance.
      agent.x += Math.cos(agent.theta) * flags.baseSpeed * speed * dt;
      agent.y += Math.sin(agent.theta) * flags.baseSpeed * speed * dt;

      // 6. Clamp inside the world bounds.
      agent.x = Math.max(0, Math.min(world.width, agent.x));
      agent.y = Math.max(0, Math.min(world.height, agent.y));

      // 7. Record the position.
      agent.trail.push({ x: agent.x, y: agent.y });
      if (agent.trail.length > TRAIL_CAP) {
        agent.trail.splice(0, agent.trail.length - TRAIL_CAP);
      }

      // 8.
      return { turn, speed, pirouetted };
    },

    // Merge partial flags at runtime (e.g. toggle ablations mid-trial).
    setFlags(partial) {
      Object.assign(flags, partial);
    },

    // Reset position, trail, brain and pirouette counter.
    reset({ x, y, theta }) {
      agent.x = x;
      agent.y = y;
      agent.theta = theta;
      agent.trail.length = 0;
      brain.reset();
      fallingCount = 0;
    },
  };

  return agent;
}
