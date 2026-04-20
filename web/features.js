// Faithful JS port of omokai/features.py.
// Plane order: [own, opp, last, color], dtype=float32, layout NCHW.

export function encodeFeatures(states) {
  if (states.length === 0) throw new Error('states must not be empty');
  const N = states[0].boardSize;
  const planeSize = N * N;
  const channels = 4;
  const out = new Float32Array(states.length * channels * planeSize);

  for (let b = 0; b < states.length; b++) {
    const s = states[b];
    const base = b * channels * planeSize;
    const ownBase = base + 0 * planeSize;
    const oppBase = base + 1 * planeSize;
    const lastBase = base + 2 * planeSize;
    const colorBase = base + 3 * planeSize;
    const toPlay = s.toPlay;
    const colorVal = toPlay === 1 ? 1.0 : 0.0;
    const board = s.board;

    for (let i = 0; i < planeSize; i++) {
      const v = board[i];
      out[ownBase + i] = v === toPlay ? 1.0 : 0.0;
      out[oppBase + i] = v === -toPlay ? 1.0 : 0.0;
      out[colorBase + i] = colorVal;
    }
    if (s.lastAction !== null && s.lastAction !== undefined && s.lastAction >= 0) {
      out[lastBase + s.lastAction] = 1.0;
    }
  }
  return out;
}
