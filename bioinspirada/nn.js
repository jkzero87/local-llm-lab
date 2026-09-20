// nn.js — the hand-wired "brain" of the C. elegans chemotaxis agent.
//
// A small feedforward network: 2 inputs -> 3 hidden -> 2 outputs, tanh on both
// the hidden and output layers. The weights are FIXED constants (not trained).
// The two inputs are: spatial = (left - right) * SPATIAL_GAIN, the lateral
// imbalance that drives the weathervane, and temporal = mean reading - mean
// running average, the change in overall level that drives the pirouette.
// Keeping them as separate channels means steering and reorientation no longer
// read the same near-zero signal in a smooth gradient.
//
// No randomness, no DOM, no imports.

// ---------------------------------------------------------------------------
// Fixed weights for the 2 -> 3 -> 2 network. Deep-frozen so it is a constant.
//
//   inputs  : ASEL (LEFT sensor) / ASER (RIGHT sensor)
//   hidden  : AIY / AIB / AIZ
//   outputs : turn (steering rate) / speed (forward speed multiplier)
// ---------------------------------------------------------------------------
export const WEIGHTS = Object.freeze({
  // Layer 1: inputs -> hidden.
  //   Column 0 = ASEL (LEFT sensor),  Column 1 = ASER (RIGHT sensor).
  //   Row 0 = AIY : L - R  (left-minus-right difference -> drives turning)
  //   Row 1 = AIB : L + R  (overall level -> drives speed)
  //   Row 2 = AIZ : R - L  (right-minus-left, mirror of AIY -> symmetry)
  hidden: Object.freeze({
    weights: Object.freeze([
      Object.freeze([1, -1]), // AIY
      Object.freeze([1, 1]), // AIB
      Object.freeze([-1, 1]), // AIZ
    ]),
    biases: Object.freeze([0, 0, 0]),
  }),

  // Layer 2: hidden -> outputs.
  //   Column 0 = AIY,  Column 1 = AIB,  Column 2 = AIZ.
  //   Row 0 = turn  : AIY - AIZ  (steer toward the stronger/rising sensor;
  //                         > 0 turns left, < 0 turns right)
  //   Row 1 = speed : AIB  (faster when the overall level is rising)
  output: Object.freeze({
    weights: Object.freeze([
      Object.freeze([1, 0, -1]), // turn
      Object.freeze([0, 1, 0]), // speed
    ]),
    biases: Object.freeze([0, 0]),
  }),
});

// Gain applied to the lateral imbalance (left - right) to form the spatial
// input. The concentration field varies slowly, so the raw left-minus-right
// difference is tiny; SPATIAL_GAIN scales it up to a meaningful range.
export const SPATIAL_GAIN = 200;

// Create a brain with per-sensor running averages (sensory adaptation).
//
//   step(leftReading, rightReading) -> { turn, speed, temporal }
//     1. avg = decay * avg + (1 - decay) * reading   (per sensor)
//     2. spatial  = (leftReading - rightReading) * SPATIAL_GAIN
//        temporal = mean(readings) - mean(runningAverages)
//        network inputs = [spatial, temporal]
//     3. forward pass 2 -> 3 -> 2, tanh on hidden and output
//     4. turn = output[0]; speed = (output[1] + 1) / 2  (kept in [0, 1])
//   reset() clears the averages; the next step behaves as a first call.
export function createBrain({ decay = 0.95 } = {}) {
  let leftAvg = 0;
  let rightAvg = 0;
  let primed = false; // whether at least one reading has been seen

  function step(leftReading, rightReading) {
    // 1. Update the running average per sensor. On the very first call the
    //    average is initialised to the first reading, so both inputs are 0.
    if (!primed) {
      leftAvg = leftReading;
      rightAvg = rightReading;
      primed = true;
    } else {
      leftAvg = decay * leftAvg + (1 - decay) * leftReading;
      rightAvg = decay * rightAvg + (1 - decay) * rightReading;
    }

    // 2. Two distinct sensory signals. spatial captures the lateral imbalance
    //    (left minus right, amplified by SPATIAL_GAIN) and drives the
    //    weathervane; temporal captures the change in overall level (mean
    //    reading minus mean running average) and drives the pirouette. They are
    //    no longer the same channel, so the near-zero smooth-gradient signal no
    //    longer blinds steering.
    const spatial = (leftReading - rightReading) * SPATIAL_GAIN;
    const temporal = (leftReading + rightReading) / 2 - (leftAvg + rightAvg) / 2;
    const inputs = [spatial, temporal];

    // 3a. Hidden layer (tanh).
    const hidden = WEIGHTS.hidden.weights.map((row, i) => {
      const pre = row.reduce((sum, w, j) => sum + w * inputs[j], 0) + WEIGHTS.hidden.biases[i];
      return Math.tanh(pre);
    });

    // 3b. Output layer (tanh).
    const outputs = WEIGHTS.output.weights.map((row, k) => {
      const pre = row.reduce((sum, w, i) => sum + w * hidden[i], 0) + WEIGHTS.output.biases[k];
      return Math.tanh(pre);
    });

    // 4. turn is the raw output 0; speed is output 1 rescaled to [0, 1];
    //    temporal is surfaced so the body can drive its pirouette from it.
    return {
      turn: outputs[0],
      speed: (outputs[1] + 1) / 2,
      temporal,
    };
  }

  function reset() {
    leftAvg = 0;
    rightAvg = 0;
    primed = false;
  }

  return { step, reset };
}
