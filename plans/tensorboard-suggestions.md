# TensorBoard 통합 계획 (Codex 실행용)

이 문서는 OmokAI 학습 파이프라인에 TensorBoard를 추가하기 위한 구체적인 구현 계획이다. 기존 JSONL 로깅은 그대로 유지하고 TensorBoard는 **보조 시각화 계층**으로 붙인다. 모든 경로는 저장소 루트 `/home/coolguy/dev/omokai` 기준이다.

---

## 0. 설계 원칙

- JSONL(`metrics.jsonl`, `training.debug.log`, `training.debug.jsonl`)은 **그대로 유지**한다. 크래시 내성과 grep 분석에 필수.
- TensorBoard는 `config.tensorboard.enabled`가 True일 때만 작동한다. 기본값은 `True`지만 CPU smoke 테스트에서는 off할 수 있어야 한다.
- SummaryWriter는 **`Trainer` 인스턴스 당 하나**. 멀티 프로세스 writer는 만들지 않는다.
- 모든 스칼라는 `self.iteration`을 global step으로 쓴다. 학습 step(update 카운터)이 필요한 메트릭만 `self.total_updates`를 쓴다.
- Codex는 이 계획만 보고 구현해야 하므로, 각 섹션은 **파일 경로 + 라인 기준 + 구체 코드 패치**로 구성됐다.

---

## 1. 의존성 추가

`pyproject.toml`에 `tensorboard`를 런타임 의존성으로 추가한다. `torch.utils.tensorboard.SummaryWriter`는 torch 번들이지만 내부적으로 `tensorboard` 패키지가 없으면 임포트에 실패한다.

```toml
dependencies = [
  ...
  "tensorboard>=2.15",
]
```

확인 명령:
```bash
python3 -c "from torch.utils.tensorboard import SummaryWriter; print('ok')"
```

---

## 2. Config 확장: `omokai/config.py`

새로운 dataclass `TensorBoardConfig`를 추가하고 `RunConfig`에 필드를 건다.

**Edit target**: `omokai/config.py`

```python
@dataclass(slots=True)
class TensorBoardConfig:
    enabled: bool = True
    log_subdir: str = "tb"            # checkpoint_dir/<log_subdir>
    histogram_interval: int = 10      # N iteration마다 histogram flush
    flush_secs: int = 30
```

`RunConfig`에 필드 추가 (필드 순서는 checkpoint 다음):

```python
checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
```

`_to_dataclass`에도 파싱 추가:

```python
tensorboard=TensorBoardConfig(**cfg.get("tensorboard", {})),
```

**확인**: `configs/rocm_unlimited.yaml`, `configs/rocm_24h.yaml`, `configs/rocm_4h.yaml`에 선택적으로 `tensorboard:` 블록 추가 가능. 기본값으로 충분하므로 YAML 변경은 **선택 사항**. `configs/quick_cpu.yaml`에는 `tensorboard:\n  enabled: false`를 추가해 smoke 테스트에서 writer를 안 띄우도록 한다.

---

## 3. Trainer 초기화: `omokai/train.py` line 68 `__init__`

**목표**: `self.tb_writer`를 만들고 `close()`를 안전하게 보장한다.

### 3a. import 추가

파일 상단 import 섹션 (`import torch` 부근):

```python
from torch.utils.tensorboard import SummaryWriter
```

### 3b. `__init__` 하단 (line 73 `self.log_file = ...` 직후)에 writer 초기화

```python
self.log_file = self.checkpoint_dir / "metrics.jsonl"
self.progress_file = self.checkpoint_dir / "runtime_progress.json"
self.debug_log_file = self.checkpoint_dir / "training.debug.log"
self.debug_jsonl_file = self.checkpoint_dir / "training.debug.jsonl"
self._configure_debug_logging()

# NEW: TensorBoard writer (None if disabled)
self.tb_writer: SummaryWriter | None = None
if self.config.tensorboard.enabled:
    tb_dir = self.checkpoint_dir / self.config.tensorboard.log_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    self.tb_writer = SummaryWriter(
        log_dir=str(tb_dir),
        flush_secs=self.config.tensorboard.flush_secs,
    )
```

### 3c. writer close 보장

`run()` 메서드(line 145) 바깥, `main()`의 `try/except/finally`에서 닫거나, `Trainer`에 `close()` 메서드를 추가하고 `main()`에서 호출한다. **권장 방식**: `main()`의 `try` 블록을 `try/finally`로 감싸고 `finally`에 `trainer.close()`를 호출.

```python
# Trainer class 하단에 추가
def close(self) -> None:
    if self.tb_writer is not None:
        self.tb_writer.flush()
        self.tb_writer.close()
        self.tb_writer = None
```

`main()` (line ~1277) 수정:

```python
try:
    trainer.run()
except BaseException:
    log_event(...)
    trainer.debug_logger.exception(...)
    raise
finally:
    trainer.close()
```

### 3d. 헬퍼 메서드 추가

반복되는 None 체크를 줄이기 위해 `Trainer`에 작은 헬퍼를 추가한다 (`_log` 메서드 근처 line 1262):

```python
def _tb_scalar(self, tag: str, value: float, step: int | None = None) -> None:
    if self.tb_writer is None:
        return
    if step is None:
        step = self.iteration
    try:
        self.tb_writer.add_scalar(tag, float(value), step)
    except Exception:
        pass  # TB failure must never break training

def _tb_histogram(self, tag: str, values, step: int | None = None) -> None:
    if self.tb_writer is None:
        return
    if step is None:
        step = self.iteration
    if self.iteration % max(1, self.config.tensorboard.histogram_interval) != 0:
        return
    try:
        self.tb_writer.add_histogram(tag, values, step)
    except Exception:
        pass
```

`values`에 torch Tensor나 numpy array를 넣을 수 있다.

---

## 4. 필수 메트릭 계측

아래 표에 나온 태그를 **모두** 구현한다. 태그 네이밍은 `category/name` 슬래시 표기를 유지한다 (TB가 이걸로 그룹을 만든다).

### 4a. `train_model` (line 812) — 학습 루프 스칼라

루프 **끝부분**, line 889 `self.total_updates += updates_done` 이후 `stats` 딕셔너리 만든 다음에 추가:

```python
# NEW: TensorBoard scalars (per-iteration aggregates)
self._tb_scalar("loss/policy", stats["policy_loss"])
self._tb_scalar("loss/value", stats["value_loss"])
self._tb_scalar("loss/total", stats["train_loss"])
self._tb_scalar("train/lr", stats["learning_rate"])
self._tb_scalar("train/updates_done", updates_done)
self._tb_scalar("train/updates_per_sec",
                0.0 if duration_seconds <= 0 else updates_done / duration_seconds)
self._tb_scalar("perf/train_sec", duration_seconds)
```

### 4b. `train_model` — 배치 단위 추가 메트릭 (루프 내부)

line 852~854 근처 loss 계산부에 **배치 통계**를 뽑는다. 매 update마다 TB에 쓰면 너무 많으니 `updates_done % 16 == 0`일 때만 기록 (기존 progress 로그와 같은 주기).

```python
with amp_context(self.device, self.config.use_amp):
    logits, value = model(states)
    log_probs = torch.log_softmax(logits, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_value)
    loss = policy_weight * policy_loss + value_weight * value_loss

# 배치 진단 지표 (step = self.total_updates + updates_done)
if (updates_done + 1) % 16 == 0:
    with torch.no_grad():
        # predicted policy distribution entropy
        probs = torch.softmax(logits, dim=1)
        policy_entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1).mean()
        # target policy (MCTS visits) entropy — 검색이 얼마나 결단력 있는지
        target_entropy = -(target_policy * torch.log(target_policy.clamp_min(1e-12))).sum(dim=1).mean()
        # value head 크기 분포
        value_abs_mean = value.abs().mean()
        value_target_abs_mean = target_value.abs().mean()

    step = self.total_updates + updates_done + 1
    self._tb_scalar("policy/entropy", float(policy_entropy.item()), step=step)
    self._tb_scalar("policy/target_entropy", float(target_entropy.item()), step=step)
    self._tb_scalar("value/abs_mean", float(value_abs_mean.item()), step=step)
    self._tb_scalar("value/target_abs_mean", float(value_target_abs_mean.item()), step=step)
```

### 4c. `train_model` — grad norm

`nn.utils.clip_grad_norm_`는 clip 전 grad norm을 반환한다. 그걸 기록한다.

line 858 치환:

```python
if self.config.optimization.grad_clip > 0:
    self.scaler.unscale_(self.optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), self.config.optimization.grad_clip)
    if (updates_done + 1) % 16 == 0:
        self._tb_scalar(
            "train/grad_norm",
            float(grad_norm),
            step=self.total_updates + updates_done + 1,
        )
```

### 4d. `generate_selfplay` (line 654) — self-play 집계

이 메서드는 iteration 하나분 self-play 통계를 `dict`로 반환한다 (line 800 근처). 반환 직전에 TB 기록 블록을 추가한다. **정확한 키는 기존 dict가 이미 쓰는 이름과 같아야** 한다 (line 799~810 참고).

`stats = {...}` 뒤, `return stats` 전:

```python
total_games = max(1, total_played)
self._tb_scalar("selfplay/games", total_played)
self._tb_scalar("selfplay/avg_moves", avg_moves)
self._tb_scalar("selfplay/black_win_rate", counters["black_wins"] / total_games)
self._tb_scalar("selfplay/white_win_rate", counters["white_wins"] / total_games)
self._tb_scalar("selfplay/draw_rate", counters["draws"] / total_games)
self._tb_scalar("selfplay/candidate_games", counters["candidate_games"])
self._tb_scalar("selfplay/best_games", counters["best_games"])
self._tb_scalar("replay/samples", len(self.replay))
self._tb_scalar("replay/games_seen", self.replay.games_seen)
```

추가로 **self-play 소요 시간**. `generate_selfplay`에 `started_at = time.monotonic()`이 있는지 확인하고, 없으면 메서드 최상단에 추가한 뒤 마지막에:

```python
self._tb_scalar("perf/selfplay_sec", time.monotonic() - started_at)
```

### 4e. `evaluate_candidate` (line 920) — arena 결과

arena 결과를 리턴하기 직전에 추가한다. arena 결과 딕셔너리 키가 `arena_candidate_win_rate`, `arena_candidate_black_win_rate`, `arena_candidate_white_win_rate`, `accepted` 등으로 이미 존재함 (line 922~ 근처 확인).

```python
# arena stats dict를 만들고 리턴하기 전에
self._tb_scalar("arena/games", arena_result.games)
self._tb_scalar("arena/win_rate", arena_result.candidate_win_rate)
self._tb_scalar(
    "arena/black_win_rate",
    0.0 if arena_result.games == 0 else 2 * arena_result.candidate_black_wins / arena_result.games,
)
self._tb_scalar(
    "arena/white_win_rate",
    0.0 if arena_result.games == 0 else 2 * arena_result.candidate_white_wins / arena_result.games,
)
self._tb_scalar("arena/promoted", 1.0 if accepted else 0.0)
self._tb_scalar("perf/arena_sec", arena_duration_seconds)
```

**주의**: `arena_result`와 `accepted`, `arena_duration_seconds`의 정확한 변수명은 `evaluate_candidate` 내부 현재 구현을 읽고 맞춘다. `accept_win_rate` 임계값도 `arena/threshold` 태그로 추가해두면 유용하다.

### 4f. `run()` (line 145) — iteration 레벨 시간/시뮬레이션

iteration 루프(line 170) 각 iteration 끝에 추가. 현재 self-play/train/arena 직렬 호출 구조이므로 각 구간 시간을 `time.monotonic()` 차로 뽑아 기록한다.

```python
self._tb_scalar("schedule/simulations", simulations)
self._tb_scalar("perf/iteration_sec", time.monotonic() - iteration_started_at)
self._tb_scalar("run/elapsed_hours", self._elapsed_hours())
```

`_elapsed_hours()` 헬퍼가 이미 존재하면 그대로 쓴다. 없으면 `(time.monotonic() - self.start_time + self.elapsed_seconds_offset) / 3600.0`.

---

## 5. Nice-to-have: histogram

`histogram_interval` 마다 (기본 10 iteration) 한 번씩만 기록한다. `train_model` 끝에 한 번:

```python
with torch.no_grad():
    # 마지막 배치의 logits/value 분포 사용
    self._tb_histogram("dist/policy_entropy_batch", policy_entropy.detach().cpu())
    self._tb_histogram("dist/value_pred", value.detach().cpu().flatten())
    self._tb_histogram("dist/value_target", target_value.detach().cpu().flatten())
```

파라미터 히스토그램 (가중치, grad):

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        self._tb_histogram(f"weights/{name}", param.detach().cpu())
        self._tb_histogram(f"grads/{name}", param.grad.detach().cpu())
```

**비용 경고**: SE-ResNet 6 blocks × 64ch도 파라미터 텐서가 200+개다. histogram_interval이 너무 짧으면 TB 파일이 금방 커진다. 기본 10 유지.

---

## 6. 테스트 & 검증

### 6a. smoke 테스트

```bash
rm -rf checkpoints/quick_cpu
python3 -m omokai.train --config configs/quick_cpu.yaml
```

- 에러 없이 2~3 iteration 완주하는지 확인.
- `tensorboard.enabled: false` 로 설정했을 때 `checkpoints/quick_cpu/tb/` 가 **생성되지 않아야** 한다.
- `true`로 바꾸면 디렉터리가 생기고 `events.out.tfevents.*` 파일이 생겨야 한다.

### 6b. TB 서버 기동 확인

```bash
tensorboard --logdir checkpoints/rocm_unlimited/tb --port 6006
```

브라우저에서 `http://localhost:6006` 접속 → Scalars 탭에 `loss/`, `arena/`, `selfplay/`, `perf/` 그룹이 보이는지 확인.

### 6c. 크래시 내성

- `Ctrl+C`로 중단해도 writer가 `close()`되어 TB 파일이 손상 없이 읽히는지 확인.
- 재시작 후 (resume 경로) 같은 `tb/` 디렉터리에 append되어 연속된 iteration이 그려지는지 확인. **Trainer가 같은 `log_dir`을 재사용하면 TB는 자동으로 이어 붙인다**. 별도 처리 불필요.

---

## 7. 커밋 전 체크리스트

- [ ] `omokai/config.py`에 `TensorBoardConfig` 추가, `RunConfig`에 필드 추가, `_to_dataclass` 파싱 추가
- [ ] `omokai/train.py` import에 `SummaryWriter` 추가
- [ ] `Trainer.__init__`에 writer 초기화
- [ ] `Trainer.close()` 추가, `main()` `finally`에서 호출
- [ ] `Trainer._tb_scalar`, `Trainer._tb_histogram` 헬퍼 추가
- [ ] `train_model`: loss/lr/updates/grad_norm/policy_entropy/value_abs_mean/perf 스칼라
- [ ] `generate_selfplay`: selfplay/replay/perf 스칼라
- [ ] `evaluate_candidate`: arena 스칼라
- [ ] `run()`: schedule/perf/run 스칼라
- [ ] `configs/quick_cpu.yaml`에 `tensorboard.enabled: false` 추가
- [ ] `pyproject.toml`에 `tensorboard>=2.15` 추가
- [ ] smoke 테스트 통과
- [ ] TB 서버 접속해 그래프 확인
- [ ] README에 "TensorBoard" 섹션 한 줄 추가: `tensorboard --logdir checkpoints/<exp>/tb`

---

## 8. 의도적으로 제외한 것 (나중에)

- **PR curve / embedding / graph**: 오목에는 부적합.
- **Multi-run 비교 뷰**: 각 run이 다른 `checkpoint_dir`에 쓰므로 `tensorboard --logdir checkpoints/` 로 parent를 넘기면 자동 비교된다. 별도 코드 불필요.
- **MCTS 내부 통계 (Q 분포, visit 편향)**: 계측 포인트가 hot path에 너무 깊어 throughput을 깎는다. 별건으로 논의.
- **GPU/CPU 사용률**: `pynvml`/ROCm tooling이 별도 필요. TB에 넣기보다 `nvidia-smi`/`rocm-smi` 외부 모니터링이 낫다.

---

## 9. 참고 파일 위치

- `omokai/train.py` (1330 lines)
  - 67: `class Trainer`
  - 68: `def __init__`
  - 73: `self.log_file = ...` — TB writer 초기화 삽입 지점
  - 145: `def run`
  - 170: iteration while 루프
  - 654: `def generate_selfplay`
  - 812: `def train_model`
  - 920: `def evaluate_candidate`
  - 1262: `def _log` — 헬퍼 추가 지점
  - 1277: `def main` — `try/finally` 변경 지점
- `omokai/config.py` — `TensorBoardConfig` 신설, `RunConfig` 확장
- `configs/quick_cpu.yaml` — TB off 스위치

끝.
