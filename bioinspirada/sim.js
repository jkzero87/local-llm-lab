// sim.js — fixed-timestep trial loop.
//
// Runs one agent in one world until it reaches the source (within
// reachRadius) or maxSteps is exhausted. Pure orchestration: no DOM, no
// randomness of its own — all randomness lives in the agent (setRandom).

export function runTrial(opts) {
  const {
    world,
    agent,
    maxSteps = 2000,
    reachRadius = 20,
    dt = 1,
    onStep = null,
  } = opts;

  let pirouettes = 0;
  let totalDistance = 0;
  let steps = 0;
  let reached = false;

  for (let i = 0; i < maxSteps; i++) {
    // Per-step hook, called before the agent moves — e.g. moving the
    // source for the moving_source condition.
    if (onStep) onStep(i, agent);

    const r = agent.step(dt);
    steps = i + 1;
    if (r.pirouetted) pirouettes += 1;

    // Distance to the source AT THIS STEP (after any onStep change), so
    // the mean stays meaningful when the source moves.
    const d = Math.hypot(world.source.x - agent.x, world.source.y - agent.y);
    totalDistance += d;

    if (d <= reachRadius) {
      reached = true;
      break;
    }
  }

  const finalDistance = Math.hypot(world.source.x - agent.x, world.source.y - agent.y);

  return {
    reached,
    steps,
    finalDistance,
    meanDistance: steps > 0 ? totalDistance / steps : finalDistance,
    pirouettes,
  };
}
