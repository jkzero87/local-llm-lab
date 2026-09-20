// world.js — a continuous 2D chemical field for the C. elegans chemotaxis agent.
//
// A single source emits a chemical plume. The concentration at a point p is a
// Gaussian in the distance to the source:
//
//     concentration(x, y) = exp(-d^2 / (2 * sigma^2)),   d = |p - source|
//
// No randomness, no DOM, no imports.

export function createWorld({ width = 600, height = 400, sx = 450, sy = 200, sigma = 120 } = {}) {
  const world = {
    width,
    height,
    sigma,
    source: { x: sx, y: sy },

    // Concentration at (x, y): exp(-d^2 / (2*sigma^2)), d = distance to source.
    concentration(x, y) {
      const dx = x - world.source.x;
      const dy = y - world.source.y;
      const d2 = dx * dx + dy * dy;
      return Math.exp(-d2 / (2 * sigma * sigma));
    },

    // Relocate the source at runtime.
    moveSource(x, y) {
      world.source.x = x;
      world.source.y = y;
    },
  };

  return world;
}
