// Faithful JS port of omokai/board.py.

export class GameState {
  constructor({ boardSize = 9, exactlyFive = false } = {}) {
    this.boardSize = boardSize;
    this.exactlyFive = exactlyFive;
    this.actionSize = boardSize * boardSize;
    this.board = new Int8Array(this.actionSize);
    this.toPlay = 1;
    this.moveCount = 0;
    this.lastAction = null;
    this.winner = 0;
    this.terminal = false;
  }

  clone() {
    const s = Object.create(GameState.prototype);
    s.boardSize = this.boardSize;
    s.exactlyFive = this.exactlyFive;
    s.actionSize = this.actionSize;
    s.board = new Int8Array(this.board);
    s.toPlay = this.toPlay;
    s.moveCount = this.moveCount;
    s.lastAction = this.lastAction;
    s.winner = this.winner;
    s.terminal = this.terminal;
    return s;
  }

  legalMoves() {
    const out = new Uint8Array(this.actionSize);
    if (this.terminal) return out;
    for (let i = 0; i < this.actionSize; i++) {
      out[i] = this.board[i] === 0 ? 1 : 0;
    }
    return out;
  }

  legalIndices() {
    const out = [];
    if (this.terminal) return out;
    for (let i = 0; i < this.actionSize; i++) {
      if (this.board[i] === 0) out.push(i);
    }
    return out;
  }

  applyAction(action) {
    if (this.terminal) throw new Error('cannot play on a terminal position');
    const N = this.boardSize;
    const row = Math.floor(action / N);
    const col = action % N;
    if (this.board[action] !== 0) throw new Error(`illegal move at ${row},${col}`);
    const player = this.toPlay;
    this.board[action] = player;
    this.lastAction = action;
    this.moveCount += 1;
    this.toPlay = -player;
    if (this._isWinningMove(row, col, player)) {
      this.winner = player;
      this.terminal = true;
    } else if (this.moveCount === this.actionSize) {
      this.winner = 0;
      this.terminal = true;
    }
  }

  outcomeForPlayer(player) {
    if (!this.terminal) throw new Error('game is not terminal');
    if (this.winner === 0) return 0.0;
    return this.winner === player ? 1.0 : -1.0;
  }

  _isWinningMove(row, col, player) {
    const dirs = [[1, 0], [0, 1], [1, 1], [1, -1]];
    for (const [dr, dc] of dirs) {
      const count = 1 + this._countDir(row, col, dr, dc, player) + this._countDir(row, col, -dr, -dc, player);
      if (this.exactlyFive) {
        if (count === 5) return true;
      } else {
        if (count >= 5) return true;
      }
    }
    return false;
  }

  _countDir(row, col, dr, dc, player) {
    const N = this.boardSize;
    let total = 0;
    let r = row + dr, c = col + dc;
    while (r >= 0 && r < N && c >= 0 && c < N && this.board[r * N + c] === player) {
      total += 1;
      r += dr;
      c += dc;
    }
    return total;
  }
}
