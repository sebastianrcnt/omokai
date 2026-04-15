# Overnight Report

작성 시각: `2026-04-15 02:41:50 KST (+0900)`

## 현재 상태

- 오목 학습 프로젝트를 빈 저장소에서 새로 구성했다.
- self-play RL 학습 루프, 체크포인트 저장, arena 평가, Pygame GUI를 구현했다.
- CPU smoke 런은 실제로 통과했다.
- 밤샘 학습은 현재 `CPU fallback`으로 실행 중이다.

## 구현 완료 항목

- `omokai/train.py`: self-play + MCTS + 학습 + arena + 체크포인트
- `omokai/gui.py`: 체크포인트 로드 후 직접 대국 가능
- `configs/smoke.yaml`: 초소형 검증 설정
- `configs/overnight_cpu.yaml`: 밤샘 CPU 설정
- `configs/rocm_24h.yaml`: GPU용 24시간 기본 설정
- `scripts/setup_rocm_wsl.sh`: AMD ROCm/WSL 점검 및 설치 안내

## 검증 결과

Smoke run:

- 명령: `python3 -m omokai.train --config configs/smoke.yaml`
- 결과: 2 iteration 정상 완료
- 생성된 체크포인트:
  - `checkpoints/smoke/best.pt`
  - `checkpoints/smoke/latest.pt`
  - `checkpoints/smoke/iter_0001.pt`
  - `checkpoints/smoke/iter_0002.pt`

Checkpoint load 검증:

- `checkpoints/smoke/best.pt` 로드 성공
- board size 15 확인

GUI 검증:

- `python3 -m omokai.gui --help` 정상 동작

## GPU / ROCm 상태

확인된 하드웨어:

- Windows 쪽 GPU: `AMD Radeon RX 9070 XT`

확인된 문제:

- AMD 공식 `torch 2.9.1 + rocm 7.2.1` wheel은 로컬 `.venv-rocm`에 설치 완료
- 하지만 현재 WSL 세션에는 ROCm WSL 런타임이 빠져 있어서 import 단계에서 멈춘다
- 대표 누락 라이브러리:
  - `libroctx64.so.4`
  - `libMIOpen.so.1`
  - `librocsolver.so.0`
  - `libamdhip64.so.7`

원인:

- 이 환경은 `sudo` 비대화식 실행이 불가능해서 AMD WSL runtime 패키지를 시스템에 설치하지 못했다
- `scripts/setup_rocm_wsl.sh`에 공식 설치 경로를 적어 두었다

## 실행 중인 밤샘 학습

- PID: `6719`
- 명령:
  - `python3 -u -m omokai.train --config configs/overnight_cpu.yaml`
- 로그:
  - `logs/overnight_cpu.log`
- 체크포인트 출력 디렉터리:
  - `checkpoints/overnight_cpu`

현재 로그 맨 앞:

```text
{"event": "startup", "device": "cpu", "cuda_available": false, "rocm_build": false, "torch_version": "2.9.0+cu128"}
```

확인된 최신 진행:

```text
{"iteration": 1, "elapsed_hours": 0.0208, "simulations": 16, "accepted": false, "arena_score": 0.5, "arena_wins": 4, "arena_losses": 4, "arena_draws": 0, "selfplay_games": 12, "selfplay_black_wins": 10, "selfplay_white_wins": 2, "selfplay_draws": 0, "selfplay_avg_moves": 54.83, "replay_samples": 658, "replay_games": 12, "train_loss": 5.775665, "policy_loss": 5.431391, "value_loss": 0.344274, "learning_rate": 0.0001}
```

## 아침에 바로 볼 것

학습 로그:

```bash
tail -n 50 logs/overnight_cpu.log
```

체크포인트 목록:

```bash
ls checkpoints/overnight_cpu
```

GUI로 모델 대국:

```bash
python3 -m omokai.gui --checkpoint checkpoints/overnight_cpu/best.pt
```

GPU 점검 재시도:

```bash
bash scripts/setup_rocm_wsl.sh
```

## 메모

- 현재 시스템 Python 3.11 환경에는 CPU/CUDA build의 torch가 남아 있다
- ROCm 전용 환경은 `.venv-rocm`에 따로 만들어 두었다
- GPU 런타임만 갖춰지면 이 프로젝트 코드는 `torch.cuda` 경로로 AMD GPU를 그대로 사용하도록 되어 있다
