from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .board import GameState
from .device import amp_context
from .network import PolicyValueNet


@dataclass(slots=True)
class TreeNode:
    to_play: int
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "TreeNode"] = field(default_factory=dict)
    expanded: bool = False

    def value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


@dataclass(slots=True)
class SearchResult:
    action: int
    visit_policy: np.ndarray
    root_value: float


class MCTS:
    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_amp = use_amp

    @torch.no_grad()
    def search_batch(
        self,
        states: list[GameState],
        num_simulations: int,
        temperature: list[float],
        add_noise: bool,
    ) -> list[SearchResult]:
        roots = [TreeNode(to_play=state.to_play) for state in states]
        priors, values = self._evaluate(states)
        for idx, state in enumerate(states):
            self._expand(roots[idx], state, priors[idx])
            if add_noise and not state.terminal:
                self._apply_root_noise(roots[idx])
        root_values = values.tolist()

        for _ in range(num_simulations):
            pending_states: list[GameState] = []
            pending_nodes: list[TreeNode] = []
            pending_paths: list[list[TreeNode]] = []

            for root, root_state in zip(roots, states, strict=True):
                if root_state.terminal:
                    continue
                state = root_state.clone()
                node = root
                path = [node]
                while node.expanded and node.children and not state.terminal:
                    action, node = self._select_child(node)
                    state.apply_action(action)
                    path.append(node)
                if state.terminal:
                    self._backup(path, state.outcome_for_player(state.to_play))
                    continue
                pending_states.append(state)
                pending_nodes.append(node)
                pending_paths.append(path)

            if not pending_states:
                continue

            batch_priors, batch_values = self._evaluate(pending_states)
            for state, node, path, prior, value in zip(
                pending_states, pending_nodes, pending_paths, batch_priors, batch_values, strict=True
            ):
                self._expand(node, state, prior)
                self._backup(path, float(value))

        results: list[SearchResult] = []
        for root, state, root_value, temp in zip(roots, states, root_values, temperature, strict=True):
            counts = np.zeros(state.action_size, dtype=np.float32)
            for action, child in root.children.items():
                counts[action] = float(child.visit_count)
            if counts.sum() == 0:
                legal = state.legal_moves().astype(np.float32)
                counts = legal / legal.sum()
            else:
                counts /= counts.sum()
            action = sample_action_from_policy(counts, temp)
            results.append(SearchResult(action=action, visit_policy=counts, root_value=float(root_value)))
        return results

    def _select_child(self, node: TreeNode) -> tuple[int, TreeNode]:
        sqrt_visits = np.sqrt(max(1, node.visit_count))
        best_action = -1
        best_score = -float("inf")
        best_child: TreeNode | None = None
        for action, child in node.children.items():
            q = -child.value()
            u = self.c_puct * child.prior * sqrt_visits / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None:
            raise RuntimeError("tree node has no child to select")
        return best_action, best_child

    def _expand(self, node: TreeNode, state: GameState, priors: np.ndarray) -> None:
        legal = state.legal_moves()
        masked_priors = np.zeros_like(priors, dtype=np.float32)
        masked_priors[legal] = priors[legal]
        total = float(masked_priors.sum())
        if total <= 0.0:
            masked_priors[legal] = 1.0 / legal.sum()
        else:
            masked_priors /= total
        node.children = {
            action: TreeNode(to_play=-state.to_play, prior=float(masked_priors[action]))
            for action in np.flatnonzero(legal)
        }
        node.expanded = True

    def _backup(self, path: list[TreeNode], value: float) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _apply_root_noise(self, root: TreeNode) -> None:
        if not root.children:
            return
        actions = list(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise, strict=True):
            child = root.children[action]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(n)

    def _evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = np.stack([state.feature_planes() for state in states], axis=0)
        tensor = torch.from_numpy(features).to(self.device, non_blocking=True)
        self.model.eval()
        with amp_context(self.device, self.use_amp):
            logits, values = self.model(tensor)
        priors = torch.softmax(logits, dim=1).float().cpu().numpy()
        return priors, values.float().cpu().numpy()


def sample_action_from_policy(policy: np.ndarray, temperature: float) -> int:
    if temperature <= 1.0e-6:
        return int(np.argmax(policy))
    adjusted = np.power(np.maximum(policy, 1.0e-8), 1.0 / temperature)
    adjusted /= adjusted.sum()
    return int(np.random.choice(np.arange(policy.size), p=adjusted))
