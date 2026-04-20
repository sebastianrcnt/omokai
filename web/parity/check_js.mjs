// Run with: node web/parity/check_js.mjs
// Reads expected.json (from Python) and replays moves via the JS port,
// checking that feature planes match exactly.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import { GameState } from '../board.js';
import { encodeFeatures } from '../features.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const cases = JSON.parse(readFileSync(resolve(__dirname, 'expected.json'), 'utf8'));

function rebuildFromBoard(boardArr, toPlay, lastAction) {
  // We can't replay arbitrary boards through applyAction since color order matters.
  // Instead, set state directly to mirror Python's encoder behavior.
  const N = Math.sqrt(boardArr.length);
  const s = new GameState({ boardSize: N });
  for (let i = 0; i < boardArr.length; i++) s.board[i] = boardArr[i];
  s.toPlay = toPlay;
  s.lastAction = lastAction < 0 ? null : lastAction;
  // moveCount/winner/terminal don't affect features.
  return s;
}

let allOk = true;
for (const c of cases) {
  const s = rebuildFromBoard(c.board, c.to_play, c.last_action);
  const planes = encodeFeatures([s]);
  const expected = c.planes;
  let maxDiff = 0;
  let firstMismatch = -1;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(planes[i] - expected[i]);
    if (d > maxDiff) maxDiff = d;
    if (d > 1e-6 && firstMismatch < 0) firstMismatch = i;
  }
  const ok = maxDiff < 1e-6;
  console.log(`${ok ? 'OK ' : 'FAIL'}  ${c.name.padEnd(20)}  maxDiff=${maxDiff.toExponential(2)}  size=${expected.length}`);
  if (!ok) {
    allOk = false;
    console.log(`     first mismatch at index ${firstMismatch}: js=${planes[firstMismatch]} expected=${expected[firstMismatch]}`);
  }
}
process.exit(allOk ? 0 : 1);
