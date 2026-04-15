from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .board import GameState
from .device import amp_context
from .features import states_to_feature_planes
from .network import PolicyValueNet


class Evaluator:
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class ModelEvaluator(Evaluator):
    def __init__(self, model: PolicyValueNet, device: torch.device, use_amp: bool) -> None:
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        features = states_to_feature_planes(states)
        return self.evaluate_features(features)

    @torch.no_grad()
    def evaluate_features(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(np.ascontiguousarray(features)).to(self.device, non_blocking=True)
        self.model.eval()
        with amp_context(self.device, self.use_amp):
            logits, values = self.model(tensor)
        priors = torch.softmax(logits, dim=1).float().cpu().numpy()
        return priors, values.float().cpu().numpy()


@dataclass(slots=True)
class _EvalRequest:
    features: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    priors: np.ndarray | None = None
    values: np.ndarray | None = None
    error: BaseException | None = None


class BatchedEvaluator(Evaluator):
    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        use_amp: bool,
        max_batch_size: int,
        wait_ms: float,
    ) -> None:
        self.local = ModelEvaluator(model=model, device=device, use_amp=use_amp)
        self.max_batch_size = max(1, int(max_batch_size))
        self.wait_seconds = max(0.0, float(wait_ms) / 1000.0)
        self.requests: queue.Queue[_EvalRequest | None] = queue.Queue()
        self.worker = threading.Thread(target=self._run, name="batched-evaluator", daemon=True)
        self.worker.start()

    def evaluate(self, states: list[GameState]) -> tuple[np.ndarray, np.ndarray]:
        request = _EvalRequest(features=states_to_feature_planes(states))
        self.requests.put(request)
        request.done.wait()
        if request.error is not None:
            raise RuntimeError("batched evaluator request failed") from request.error
        if request.priors is None or request.values is None:
            raise RuntimeError("batched evaluator returned no result")
        return request.priors, request.values

    def close(self) -> None:
        self.requests.put(None)
        self.worker.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            request = self.requests.get()
            if request is None:
                return

            batch = [request]
            total = request.features.shape[0]
            deadline = time.monotonic() + self.wait_seconds
            while total < self.max_batch_size:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout <= 0.0:
                    break
                try:
                    pending = self.requests.get(timeout=timeout)
                except queue.Empty:
                    break
                if pending is None:
                    self.requests.put(None)
                    break
                batch.append(pending)
                total += pending.features.shape[0]

            try:
                features = np.concatenate([item.features for item in batch], axis=0)
                priors, values = self.local.evaluate_features(features)
                offset = 0
                for item in batch:
                    count = item.features.shape[0]
                    item.priors = priors[offset : offset + count]
                    item.values = values[offset : offset + count]
                    item.done.set()
                    offset += count
            except BaseException as exc:
                for item in batch:
                    item.error = exc
                    item.done.set()
