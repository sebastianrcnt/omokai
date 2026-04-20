import { GameState } from './board.js';
import { OnnxEvaluator } from './evaluator.js';
import { MCTS } from './mcts.js';

const ORT = window.ort;
// Tell ORT where to fetch its WASM artifacts.
ORT.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/';
ORT.env.wasm.numThreads = 1; // simple, no SAB requirements

const els = {
  board: document.getElementById('board'),
  ckpt: document.getElementById('checkpoint'),
  color: document.getElementById('player-color'),
  sims: document.getElementById('sims'),
  simsValue: document.getElementById('sims-value'),
  newGame: document.getElementById('new-game'),
  undo: document.getElementById('undo'),
  status: document.getElementById('status-line'),
  toPlay: document.getElementById('to-play'),
  lastMove: document.getElementById('last-move'),
  aiValue: document.getElementById('ai-value'),
  aiProgress: document.getElementById('ai-progress'),
  ckptInfo: document.getElementById('ckpt-info'),
};

const COLS = 'ABCDEFGHJKLMNOP'; // skip "I" by Go convention; only first N used

const state = {
  manifest: null,
  evaluator: null,
  mcts: null,
  game: null,
  history: [], // moves played, for undo
  boardSize: 9,
  exactlyFive: false,
  humanPlayer: 1,
  busy: false,
  aiSubtree: null,
};

async function init() {
  await loadManifest();
  bindUI();
  await loadCheckpoint(getRecommendedFile());
  newGame();
}

async function loadManifest() {
  const res = await fetch('models/manifest.json');
  state.manifest = await res.json();
  els.ckpt.innerHTML = '';
  for (const c of state.manifest.checkpoints) {
    const opt = document.createElement('option');
    opt.value = c.file;
    const tag = c.recommended ? '  ⭐' : '';
    opt.textContent = `${c.name}${tag}`;
    opt.dataset.boardSize = c.board_size;
    opt.dataset.exactlyFive = c.exactly_five;
    els.ckpt.appendChild(opt);
  }
  // default selection
  const recommended = state.manifest.checkpoints.find(c => c.recommended);
  if (recommended) els.ckpt.value = recommended.file;
}

function getRecommendedFile() {
  const recommended = state.manifest.checkpoints.find(c => c.recommended);
  return recommended ? recommended.file : state.manifest.checkpoints[0].file;
}

function getCheckpointMeta(file) {
  return state.manifest.checkpoints.find(c => c.file === file);
}

async function loadCheckpoint(file) {
  setStatus(`모델 로딩: ${file} …`);
  setBusy(true);
  const meta = getCheckpointMeta(file);
  state.boardSize = meta.board_size;
  state.exactlyFive = meta.exactly_five;
  els.ckptInfo.textContent = JSON.stringify(meta, null, 2);
  state.evaluator = await OnnxEvaluator.load(`models/${file}`, state.boardSize);
  state.mcts = new MCTS({ cPuct: 1.6, evaluator: state.evaluator });
  setBusy(false);
  setStatus(`${file} 로딩 완료`);
}

function bindUI() {
  els.sims.addEventListener('input', () => {
    els.simsValue.textContent = els.sims.value;
  });
  els.simsValue.textContent = els.sims.value;

  els.ckpt.addEventListener('change', async (e) => {
    await loadCheckpoint(e.target.value);
    newGame();
  });

  els.color.addEventListener('change', () => {
    state.humanPlayer = parseInt(els.color.value, 10);
    newGame();
  });

  els.newGame.addEventListener('click', newGame);
  els.undo.addEventListener('click', undo);

  els.board.addEventListener('click', onBoardClick);
}

function newGame() {
  state.game = new GameState({ boardSize: state.boardSize, exactlyFive: state.exactlyFive });
  state.history = [];
  state.aiSubtree = null;
  state.humanPlayer = parseInt(els.color.value, 10);
  setStatus(state.game.toPlay === state.humanPlayer ? '당신 차례입니다.' : 'AI가 두는 중…');
  redraw();
  if (state.game.toPlay !== state.humanPlayer) aiMove();
}

function undo() {
  if (state.busy) return;
  // Pop until it's the human's turn again, but at least one move.
  if (state.history.length === 0) return;
  state.history.pop(); // remove last (could be AI or human)
  if (state.history.length > 0 && state.game) {
    // also pop one more if the last popped was AI (so human can replay)
    // simpler: rebuild from history
  }
  rebuildFromHistory();
  redraw();
}

function rebuildFromHistory() {
  state.game = new GameState({ boardSize: state.boardSize, exactlyFive: state.exactlyFive });
  state.aiSubtree = null;
  for (const a of state.history) state.game.applyAction(a);
  setStatus(state.game.terminal ? terminalMessage() : (state.game.toPlay === state.humanPlayer ? '당신 차례입니다.' : 'AI가 두는 중…'));
  if (!state.game.terminal && state.game.toPlay !== state.humanPlayer) aiMove();
}

function onBoardClick(e) {
  if (state.busy || !state.game || state.game.terminal) return;
  if (state.game.toPlay !== state.humanPlayer) return;
  const rect = els.board.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (els.board.width / rect.width);
  const y = (e.clientY - rect.top) * (els.board.height / rect.height);
  const cell = pixelToCell(x, y);
  if (cell === null) return;
  const action = cell.row * state.boardSize + cell.col;
  if (state.game.board[action] !== 0) return;
  playMove(action);
  if (!state.game.terminal) aiMove();
}

function playMove(action) {
  state.game.applyAction(action);
  state.history.push(action);
  // we played, so AI's previous subtree no longer matches; drop it
  state.aiSubtree = null;
  redraw();
  if (state.game.terminal) {
    setStatus(terminalMessage());
  }
}

async function aiMove() {
  if (state.game.terminal) return;
  setBusy(true);
  setStatus('AI 탐색 중…');
  const sims = parseInt(els.sims.value, 10);
  els.aiProgress.textContent = `0 / ${sims}`;
  const t0 = performance.now();
  const result = await state.mcts.run(state.game, sims, {
    reuseRoot: state.aiSubtree,
    onProgress: (done, total) => { els.aiProgress.textContent = `${done} / ${total}`; },
  });
  const dt = performance.now() - t0;
  state.game.applyAction(result.action);
  state.history.push(result.action);
  state.aiSubtree = result.nextRoot;
  els.aiValue.textContent = `${result.rootValue.toFixed(3)}  (${dt.toFixed(0)}ms)`;
  els.aiProgress.textContent = `${sims} / ${sims}`;
  redraw();
  if (state.game.terminal) {
    setStatus(terminalMessage());
  } else {
    setStatus('당신 차례입니다.');
  }
  setBusy(false);
}

function terminalMessage() {
  if (state.game.winner === 0) return '무승부.';
  const human = state.game.winner === state.humanPlayer;
  return human ? '당신 승리! 🎉' : 'AI 승리.';
}

function setBusy(b) {
  state.busy = b;
  els.newGame.disabled = false; // always allow new game
  els.undo.disabled = b;
  els.ckpt.disabled = b;
  els.color.disabled = b;
}

function setStatus(msg) {
  els.status.textContent = msg;
  if (state.game) {
    els.toPlay.textContent = state.game.toPlay === 1 ? '흑' : '백';
    els.lastMove.textContent = state.game.lastAction === null ? '-' : actionToLabel(state.game.lastAction);
  }
}

function actionToLabel(action) {
  const N = state.boardSize;
  const row = Math.floor(action / N);
  const col = action % N;
  return `${COLS[col]}${N - row}`;
}

// ---- Rendering ----

function cellMetrics() {
  const W = els.board.width;
  const N = state.boardSize;
  const margin = W / (N + 1);
  const step = (W - 2 * margin) / (N - 1);
  return { margin, step };
}

function pixelToCell(x, y) {
  const { margin, step } = cellMetrics();
  const N = state.boardSize;
  const col = Math.round((x - margin) / step);
  const row = Math.round((y - margin) / step);
  if (row < 0 || row >= N || col < 0 || col >= N) return null;
  return { row, col };
}

function redraw() {
  const ctx = els.board.getContext('2d');
  const W = els.board.width;
  const N = state.boardSize;
  const { margin, step } = cellMetrics();

  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--board').trim() || '#d9b271';
  ctx.fillRect(0, 0, W, W);

  ctx.strokeStyle = '#2a2118';
  ctx.lineWidth = 1.2;
  for (let i = 0; i < N; i++) {
    const p = margin + i * step;
    ctx.beginPath();
    ctx.moveTo(margin, p);
    ctx.lineTo(W - margin, p);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(p, margin);
    ctx.lineTo(p, W - margin);
    ctx.stroke();
  }

  // coordinate labels
  ctx.fillStyle = '#3a2f22';
  ctx.font = `${Math.max(10, step * 0.28)}px ui-monospace, monospace`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < N; i++) {
    ctx.fillText(COLS[i], margin + i * step, margin / 2);
    ctx.fillText(String(N - i), margin / 2, margin + i * step);
  }

  // star points (center for small boards, plus corners for 9x9)
  ctx.fillStyle = '#2a2118';
  const stars = N === 9 ? [[2, 2], [2, 6], [6, 2], [6, 6], [4, 4]] : N === 15 ? [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]] : [[Math.floor(N / 2), Math.floor(N / 2)]];
  for (const [r, c] of stars) {
    ctx.beginPath();
    ctx.arc(margin + c * step, margin + r * step, Math.max(2, step * 0.06), 0, Math.PI * 2);
    ctx.fill();
  }

  // stones
  if (!state.game) return;
  const stoneR = step * 0.42;
  for (let i = 0; i < state.game.actionSize; i++) {
    const v = state.game.board[i];
    if (v === 0) continue;
    const r = Math.floor(i / N);
    const c = i % N;
    const cx = margin + c * step;
    const cy = margin + r * step;
    const grad = ctx.createRadialGradient(cx - stoneR * 0.3, cy - stoneR * 0.3, stoneR * 0.1, cx, cy, stoneR);
    if (v === 1) {
      grad.addColorStop(0, '#555');
      grad.addColorStop(1, '#000');
    } else {
      grad.addColorStop(0, '#fff');
      grad.addColorStop(1, '#cfc8b8');
    }
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, stoneR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = v === 1 ? '#000' : '#888';
    ctx.lineWidth = 0.8;
    ctx.stroke();
  }

  // last-move marker
  if (state.game.lastAction !== null) {
    const r = Math.floor(state.game.lastAction / N);
    const c = state.game.lastAction % N;
    const cx = margin + c * step;
    const cy = margin + r * step;
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, stoneR * 0.35, 0, Math.PI * 2);
    ctx.stroke();
  }
}

init().catch(err => {
  console.error(err);
  setStatus('초기화 실패: ' + err.message);
});
