// Wraps onnxruntime-web Session to provide a `.evaluate(states)` API
// matching the Python ModelEvaluator (returns softmaxed policy + value).

import { encodeFeatures } from './features.js';

const ORT = window.ort;

function softmaxRow(arr, offset, size) {
  let max = -Infinity;
  for (let i = 0; i < size; i++) if (arr[offset + i] > max) max = arr[offset + i];
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const e = Math.exp(arr[offset + i] - max);
    arr[offset + i] = e;
    sum += e;
  }
  for (let i = 0; i < size; i++) arr[offset + i] /= sum;
}

export class OnnxEvaluator {
  constructor(session, boardSize) {
    this.session = session;
    this.boardSize = boardSize;
    this.actionSize = boardSize * boardSize;
  }

  static async load(url, boardSize, options = {}) {
    const session = await ORT.InferenceSession.create(url, {
      executionProviders: options.executionProviders || ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new OnnxEvaluator(session, boardSize);
  }

  async evaluate(states) {
    const batch = states.length;
    const features = encodeFeatures(states);
    const tensor = new ORT.Tensor('float32', features, [batch, 4, this.boardSize, this.boardSize]);
    const out = await this.session.run({ input: tensor });
    const logits = out.policy_logits.data; // Float32Array length = batch*action_size
    const values = out.value.data; // Float32Array length = batch
    // Apply softmax in-place per row.
    const policy = new Float32Array(logits.length);
    policy.set(logits);
    for (let b = 0; b < batch; b++) softmaxRow(policy, b * this.actionSize, this.actionSize);

    // Slice into per-batch arrays for ergonomic consumption.
    const policyOut = [];
    const valueOut = [];
    for (let b = 0; b < batch; b++) {
      policyOut.push(policy.subarray(b * this.actionSize, (b + 1) * this.actionSize));
      valueOut.push(values[b]);
    }
    return { policy: policyOut, value: valueOut };
  }
}
