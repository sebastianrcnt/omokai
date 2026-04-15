from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .board import GameState
from .evaluator import Evaluator


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
    next_root: TreeNode | None = None


class MCTS:
    def __init__(
        self,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        evaluator: Evaluator,
    ) -> None:
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.evaluator = evaluator

    def search_batch(
        self,
        states: list[GameState],
        num_simulations: int,
        temperature: list[float],
        add_noise: bool,
        roots: list[TreeNode | None] | None = None,
        leaves_per_batch: int = 1,
        virtual_loss: float = 0.0,
    ) -> list[SearchResult]:
        if roots is None:
            roots = [None] * len(states)
        if len(roots) != len(states):
            raise ValueError("roots and states must have the same length")
        leaves_per_batch = max(1, int(leaves_per_batch))
        vl = max(0.0, float(virtual_loss))

        active_roots = [root if root is not None else TreeNode(to_play=state.to_play) for state, root in zip(states, roots, strict=True)]
        init_indices: list[int] = []
        init_states: list[GameState] = []
        root_values = np.zeros(len(states), dtype=np.float32)

        for idx, (state, root) in enumerate(zip(states, active_roots, strict=True)):
            if state.terminal:
                continue
            if not root.expanded:
                init_indices.append(idx)
                init_states.append(state)
            elif root.visit_count > 0:
                root_values[idx] = float(root.value())

        if init_states:
            priors, values = self._evaluate(init_states)
            for offset, idx in enumerate(init_indices):
                state = states[idx]
                root = active_roots[idx]
                self._expand(root, state, priors[offset])
                root_values[idx] = float(values[offset])

        for state, root in zip(states, active_roots, strict=True):
            if add_noise and not state.terminal:
                self._apply_root_noise(root)

        sims_done = 0
        while sims_done < num_simulations:
            leaves_this_round = min(leaves_per_batch, num_simulations - sims_done)
            pending_states: list[GameState] = []
            pending_nodes: list[TreeNode] = []
            pending_paths: list[list[TreeNode]] = []

            for root, root_state in zip(active_roots, states, strict=True):
                if root_state.terminal:
                    continue
                for _ in range(leaves_this_round):
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
                    if vl > 0.0 and leaves_this_round > 1:
                        self._apply_virtual_loss(path, vl)
                    pending_states.append(state)
                    pending_nodes.append(node)
                    pending_paths.append(path)

            sims_done += leaves_this_round
            if not pending_states:
                continue

            batch_priors, batch_values = self._evaluate(pending_states)
            for state, node, path, prior, value in zip(
                pending_states, pending_nodes, pending_paths, batch_priors, batch_values, strict=True
            ):
                if vl > 0.0 and leaves_this_round > 1:
                    self._revert_virtual_loss(path, vl)
                self._expand(node, state, prior)
                self._backup(path, float(value))

        results: list[SearchResult] = []
        for root, state, root_value, temp in zip(active_roots, states, root_values, temperature, strict=True):
            counts = np.zeros(state.action_size, dtype=np.float32)
            for action, child in root.children.items():
                counts[action] = float(child.visit_count)
            if counts.sum() == 0:
                legal = state.legal_moves().astype(np.float32)
                counts = legal / legal.sum()
            else:
                counts /= counts.sum()
            action = sample_action_from_policy(counts, temp)
            next_root = None if state.terminal else root.children.get(action)
            results.append(SearchResult(action=action, visit_policy=counts, root_value=float(root_value), next_root=next_root))
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

    def _apply_virtual_loss(self, path: list[TreeNode], strength: float) -> None:
        for idx, node in enumerate(path):
            node.visit_count += 1
            if idx > 0:
                node.value_sum += strength

    def _revert_virtual_loss(self, path: list[TreeNode], strength: float) -> None:
        for idx, node in enumerate(path):
            node.visit_count -= 1
            if idx > 0:
                node.value_sum -= strength

    def _apply_root_noise(self, root: TreeNode) -> None:
        if not root.children:
            return
        actions = list(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for action, n in zip(actions, noise, strict=True):
            child = root.children[action]
            child.prior = (1.0 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(n)

    def _evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluator.evaluate(states)


def sample_action_from_policy(policy: np.ndarray, temperature: float) -> int:
    if temperature <= 1.0e-6:
        return int(np.argmax(policy))
    adjusted = np.power(np.maximum(policy, 1.0e-8), 1.0 / temperature)
    adjusted /= adjusted.sum()
    return int(np.random.choice(np.arange(policy.size), p=adjusted))
