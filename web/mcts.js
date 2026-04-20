// JS port of omokai/mcts.py — sequential single-leaf variant suitable for browser.
// Async because the evaluator (onnxruntime-web) returns a Promise.

import { encodeFeatures } from './features.js';

class TreeNode {
  constructor(toPlay, prior = 0.0) {
    this.toPlay = toPlay;
    this.prior = prior;
    this.visitCount = 0;
    this.valueSum = 0.0;
    this.children = new Map();
    this.expanded = false;
  }

  value() {
    return this.visitCount === 0 ? 0.0 : this.valueSum / this.visitCount;
  }
}

export class MCTS {
  constructor({ cPuct = 1.6, evaluator }) {
    this.cPuct = cPuct;
    this.evaluator = evaluator;
  }

  // Run `numSimulations` simulations from `state`. Optionally pass the previous
  // root from a prior call to reuse the subtree.
  async run(state, numSimulations, { reuseRoot = null, onProgress = null } = {}) {
    const root = reuseRoot && reuseRoot.toPlay === state.toPlay ? reuseRoot : new TreeNode(state.toPlay);

    if (!root.expanded && !state.terminal) {
      const { policy, value } = await this.evaluator.evaluate([state]);
      this._expand(root, state, policy[0]);
      root.rootValue = value[0];
    }

    for (let sim = 0; sim < numSimulations; sim++) {
      let node = root;
      const path = [node];
      const sim_state = state.clone();

      while (node.expanded && node.children.size > 0 && !sim_state.terminal) {
        const [action, child] = this._selectChild(node);
        sim_state.applyAction(action);
        node = child;
        path.push(node);
      }

      let leafValue;
      if (sim_state.terminal) {
        leafValue = sim_state.outcomeForPlayer(sim_state.toPlay);
      } else {
        const { policy, value } = await this.evaluator.evaluate([sim_state]);
        this._expand(node, sim_state, policy[0]);
        leafValue = value[0];
      }
      this._backup(path, leafValue);

      if (onProgress && (sim + 1) % 16 === 0) onProgress(sim + 1, numSimulations);
    }

    return this._chooseAction(root, state);
  }

  _selectChild(node) {
    const sqrtVisits = Math.sqrt(Math.max(1, node.visitCount));
    let bestAction = -1;
    let bestScore = -Infinity;
    let bestChild = null;
    for (const [action, child] of node.children) {
      const q = -child.value();
      const u = (this.cPuct * child.prior * sqrtVisits) / (1 + child.visitCount);
      const score = q + u;
      if (score > bestScore) {
        bestScore = score;
        bestAction = action;
        bestChild = child;
      }
    }
    return [bestAction, bestChild];
  }

  _expand(node, state, priors) {
    const legal = state.legalIndices();
    let total = 0;
    for (const a of legal) total += priors[a];
    node.children.clear();
    if (total <= 0 || !isFinite(total)) {
      const uniform = 1.0 / legal.length;
      for (const a of legal) {
        node.children.set(a, new TreeNode(-state.toPlay, uniform));
      }
    } else {
      for (const a of legal) {
        node.children.set(a, new TreeNode(-state.toPlay, priors[a] / total));
      }
    }
    node.expanded = true;
  }

  _backup(path, value) {
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) {
      const node = path[i];
      node.visitCount += 1;
      node.valueSum += v;
      v = -v;
    }
  }

  _chooseAction(root, state) {
    const counts = new Float32Array(state.actionSize);
    let total = 0;
    let best = -1;
    let bestCount = -1;
    for (const [action, child] of root.children) {
      counts[action] = child.visitCount;
      total += child.visitCount;
      if (child.visitCount > bestCount) {
        bestCount = child.visitCount;
        best = action;
      }
    }
    if (total === 0) {
      // fallback to legal-uniform
      const legal = state.legalIndices();
      best = legal[Math.floor(Math.random() * legal.length)];
      for (const a of legal) counts[a] = 1.0 / legal.length;
    } else {
      for (let i = 0; i < counts.length; i++) counts[i] /= total;
    }
    return {
      action: best,
      visitPolicy: counts,
      rootValue: root.value(),
      nextRoot: state.terminal ? null : root.children.get(best) || null,
    };
  }
}
