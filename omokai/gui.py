from __future__ import annotations

import argparse
import random
from pathlib import Path

import pygame

from .board import GameState
from .checkpoint import list_checkpoints, load_checkpoint
from .config import load_config
from .device import resolve_device
from .evaluator import ModelEvaluator
from .mcts import MCTS
from .network import PolicyValueNet
from .openings import sample_balanced_openings


class OmokGUI:
    def __init__(
        self,
        checkpoint: str | None,
        config_path: str,
        device_name: str,
        simulations: int,
        human_color: int = -1,
        seed: int = 0,
    ) -> None:
        self.device = resolve_device(device_name)
        self.simulations = simulations
        self.checkpoint_source = checkpoint
        self.checkpoints = self._discover_checkpoints(checkpoint)
        self.checkpoint_index = len(self.checkpoints) - 1 if self.checkpoints else 0
        self.config = load_config(config_path)
        self.model = PolicyValueNet(self.config.rules.board_size, self.config.network).to(self.device)
        self.metadata: dict[str, object] = {}
        if self.checkpoints:
            self._load_checkpoint(self.checkpoints[self.checkpoint_index])
        self.state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
        self.human_color = 1 if int(human_color) >= 1 else -1
        self.seed = int(seed)
        self.current_opening: list[int] = []

        pygame.init()
        pygame.display.set_caption("OmokAI GUI")
        self.screen = pygame.display.set_mode((1080, 860))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Noto Sans CJK KR,Malgun Gothic,DejaVu Sans", 22)
        self.small_font = pygame.font.SysFont("Noto Sans CJK KR,Malgun Gothic,DejaVu Sans", 18)
        self.margin = 50
        self.board_px = 760
        self.cell = self.board_px / (self.config.rules.board_size - 1)
        self.board_origin = (self.margin, self.margin)

    def _discover_checkpoints(self, checkpoint: str | None) -> list[Path]:
        if checkpoint:
            target = Path(checkpoint)
            if target.is_dir():
                return list_checkpoints(target)
            return [target]
        for default_dir in ("checkpoints/rocm_unlimited", "checkpoints/rocm_24h", "checkpoints/overnight_cpu", "checkpoints/smoke"):
            found = list_checkpoints(default_dir)
            if found:
                return found
        return []

    def _load_checkpoint(self, path: Path) -> None:
        try:
            model, config, metadata = load_checkpoint(path, device="cpu")
        except Exception as exc:
            print(f"[gui] failed to load {path}: {exc}")
            return
        self.config = config
        self.model = model.to(self.device)
        self.model.eval()
        self.metadata = metadata
        if path in self.checkpoints:
            self.checkpoint_index = self.checkpoints.index(path)
        self.state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
        self.current_opening = []

    def _refresh_checkpoints(self) -> None:
        previous = self.checkpoints[self.checkpoint_index] if self.checkpoints else None
        self.checkpoints = self._discover_checkpoints(self.checkpoint_source)
        if not self.checkpoints:
            self.checkpoint_index = 0
            return
        if previous and previous in self.checkpoints:
            self.checkpoint_index = self.checkpoints.index(previous)
        else:
            self.checkpoint_index = len(self.checkpoints) - 1
        self._load_checkpoint(self.checkpoints[self.checkpoint_index])

    def _reset_state(self, apply_opening: bool) -> None:
        self.state = GameState(self.config.rules.board_size, self.config.rules.exactly_five)
        self.current_opening = []
        if not apply_opening:
            return
        rng = random.Random(self.seed)
        openings = sample_balanced_openings(self.config.rules.board_size, 32, rng)
        if not openings:
            return
        opening = openings[0]
        for action in opening:
            if self.state.terminal:
                break
            legal = self.state.legal_moves()
            if not legal[action]:
                break
            self.state.apply_action(action)
            self.current_opening.append(action)

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if not self.state.terminal and self.state.to_play != self.human_color:
                self._ai_move()

            self._render()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_r:
            self._reset_state(apply_opening=bool(self.current_opening))
        elif key == pygame.K_s:
            self.human_color *= -1
            self._reset_state(apply_opening=False)
        elif key == pygame.K_m and not self.state.terminal:
            self._ai_move()
        elif key == pygame.K_n and self.checkpoints:
            self.checkpoint_index = (self.checkpoint_index + 1) % len(self.checkpoints)
            self._load_checkpoint(self.checkpoints[self.checkpoint_index])
        elif key == pygame.K_p and self.checkpoints:
            self.checkpoint_index = (self.checkpoint_index - 1) % len(self.checkpoints)
            self._load_checkpoint(self.checkpoints[self.checkpoint_index])
        elif key == pygame.K_l:
            self._refresh_checkpoints()
        elif key == pygame.K_o:
            self._reset_state(apply_opening=True)
        elif key == pygame.K_LEFTBRACKET:
            self.seed = max(0, self.seed - 1)
        elif key == pygame.K_RIGHTBRACKET:
            self.seed += 1
        return True

    def _handle_click(self, pos: tuple[int, int]) -> None:
        if self.state.terminal or self.state.to_play != self.human_color:
            return
        action = self._pos_to_action(pos)
        if action is None or not self.state.legal_moves()[action]:
            return
        self.state.apply_action(action)

    def _ai_move(self) -> None:
        evaluator = ModelEvaluator(model=self.model, device=self.device, use_amp=self.config.use_amp)
        search = MCTS(
            c_puct=self.config.selfplay.c_puct,
            dirichlet_alpha=self.config.selfplay.dirichlet_alpha,
            dirichlet_epsilon=0.0,
            evaluator=evaluator,
        )
        result = search.search_batch(
            [self.state],
            self.simulations,
            [0.0],
            add_noise=False,
            leaves_per_batch=self.config.selfplay.leaves_per_batch,
            virtual_loss=self.config.selfplay.virtual_loss,
        )[0]
        self.state.apply_action(result.action)

    def _render(self) -> None:
        self.screen.fill((244, 224, 179))
        self._draw_board()
        self._draw_sidebar()

    def _draw_board(self) -> None:
        ox, oy = self.board_origin
        size = self.config.rules.board_size
        board_end_x = ox + self.board_px
        board_end_y = oy + self.board_px

        for i in range(size):
            x = int(ox + i * self.cell)
            y = int(oy + i * self.cell)
            pygame.draw.line(self.screen, (70, 52, 33), (ox, y), (board_end_x, y), 1)
            pygame.draw.line(self.screen, (70, 52, 33), (x, oy), (x, board_end_y), 1)

        for row in range(size):
            for col in range(size):
                stone = self.state.board[row, col]
                if stone == 0:
                    continue
                x = int(ox + col * self.cell)
                y = int(oy + row * self.cell)
                color = (24, 24, 24) if stone == 1 else (245, 245, 245)
                outline = (20, 20, 20) if stone == -1 else color
                pygame.draw.circle(self.screen, color, (x, y), 16)
                pygame.draw.circle(self.screen, outline, (x, y), 16, 1)

        if self.state.last_action is not None:
            row, col = divmod(self.state.last_action, size)
            x = int(ox + col * self.cell)
            y = int(oy + row * self.cell)
            pygame.draw.circle(self.screen, (215, 65, 65), (x, y), 5)

    def _draw_sidebar(self) -> None:
        x = 850
        lines = [
            "Checkpoint",
            self.checkpoints[self.checkpoint_index].name if self.checkpoints else "random-init",
            "",
            f"Human: {'Black' if self.human_color == 1 else 'White'}",
            f"Turn: {'Black' if self.state.to_play == 1 else 'White'}",
            f"Moves: {self.state.move_count}",
            f"Simulations: {self.simulations}",
            "",
        ]
        if self.state.terminal:
            if self.state.winner == 0:
                lines.append("Result: Draw")
            else:
                lines.append(f"Result: {'Black' if self.state.winner == 1 else 'White'} wins")
        else:
            lines.append("Result: In progress")

        lines.extend(
            [
                "",
                f"Seed: {self.seed}",
                f"Opening: {len(self.current_opening)} plies" if self.current_opening else "Opening: none",
                "",
                "Controls",
                "LMB: move",
                "R: reset (keep opening)",
                "O: apply opening w/ seed",
                "[ / ]: seed -/+",
                "S: swap side",
                "N/P: next/prev ckpt",
                "L: reload ckpt list",
                "M: force AI move",
                "Esc: quit",
            ]
        )

        if self.metadata:
            lines.extend(
                [
                    "",
                    f"Iteration: {self.metadata.get('iteration', '-')}",
                    f"Elapsed h: {round(float(self.metadata.get('elapsed_hours', 0.0)), 2)}",
                    f"Warmup: {self.metadata.get('warmup', False)}",
                ]
            )

        for index, text in enumerate(lines):
            font = self.font if index == 0 or text in {"Controls"} else self.small_font
            surface = font.render(text, True, (28, 28, 28))
            self.screen.blit(surface, (x, 60 + index * 28))

    def _pos_to_action(self, pos: tuple[int, int]) -> int | None:
        ox, oy = self.board_origin
        x, y = pos
        size = self.config.rules.board_size
        if x < ox - 18 or y < oy - 18 or x > ox + self.board_px + 18 or y > oy + self.board_px + 18:
            return None
        col = round((x - ox) / self.cell)
        row = round((y - oy) / self.cell)
        if not (0 <= row < size and 0 <= col < size):
            return None
        return row * size + col


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play against a saved OmokAI checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/rocm_24h.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--human-color", type=str, default="white", choices=["black", "white"])
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    human_color = 1 if args.human_color == "black" else -1
    app = OmokGUI(
        checkpoint=args.checkpoint,
        config_path=args.config,
        device_name=args.device,
        simulations=args.simulations,
        human_color=human_color,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
