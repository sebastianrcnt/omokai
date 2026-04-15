# OmokAI

오목용 self-play RL 프로젝트다. 빈 저장소에서 바로 학습과 플레이가 가능하도록, 학습 엔진과 체크포인트 GUI를 함께 포함한다. 현재 기본 학습 프로파일은 bootstrap 안정성을 위해 9x9 설정을 사용한다.

## 선택한 학습 방식

검토한 후보:

- AlphaZero
- Gumbel AlphaZero
- MuZero
- Gumbel MuZero
- MiniZero의 progressive simulation
- KataGo 계열의 실전 최적화

최종 선택:

- 기본 알고리즘은 `AlphaZero 스타일 self-play + PUCT MCTS`
- 여기에 `SE-ResNet`, `8방향 대칭 증강`, `mixed precision`, `progressive simulation`, `arena gating`을 적용

이 선택을 한 이유:

- 오목은 규칙 모델이 완전히 알려져 있어서, MuZero의 세계모델 학습 비용이 상대적으로 불리하다.
- 24시간 예산에서는 검색 품질과 self-play throughput이 중요하다.
- MiniZero 비교 실험과 Gumbel/MuZero 계열 결과를 보면, 보드게임에서는 시뮬레이션 예산과 검색 설계가 매우 중요하고, 게임 특성에 따라 AlphaZero가 더 효율적일 수 있다.
- KataGo처럼 규칙이 알려진 게임에서는 강한 검색 + 좋은 표현학습이 가장 실전적이다.

즉, "가장 최신 논문 이름"을 그대로 쓰는 대신, `현재 제약에서 가장 강하게 학습될 가능성이 높은 조합`으로 설계했다.

## 포함 기능

- 가변 보드 크기 오목 환경
- 배치 self-play MCTS
- `virtual loss` 기반 leaves-per-batch MCTS로 단일 게임에서도 GPU 추론 배치를 키움
- `discounted value target` — 빨리 이긴 게임이 늦게 이긴 게임보다 강한 신호를 가지도록 γ^remaining 감가
- `recency-weighted replay` — 오래된 self-play 샘플 비중을 줄여 정책 업데이트가 과거 분포에 끌려가지 않도록 함
- 정책/가치 손실 가중치 및 선형 temperature decay 설정
- 정책/가치 헤드가 달린 SE-ResNet
- `best` 대 `candidate` arena 승급 평가
- 중단/재개 가능한 trainer state 저장
- 체크포인트 저장 및 로드
- Pygame GUI로 체크포인트와 직접 대국 (학습 중에도 최신 체크포인트 리로드 가능)
- 오프닝 시드 선택으로 다양한 시작 국면 재현
- ROCm/WSL 설치 가이드 및 자동 점검 스크립트

## 설치

`torch`는 CPU/ROCm 환경별 설치가 달라서 기본 의존성에 고정하지 않았다.

먼저 기본 패키지를 설치한다.

```bash
python3 -m pip install -e .
```

AMD ROCm on WSL 환경이면 아래 스크립트로 설치 가이드를 확인한다.

```bash
bash scripts/setup_rocm_wsl.sh
```

## 빠른 시작

CPU 스모크 테스트:

```bash
python3 -m omokai.train --config configs/quick_cpu.yaml
```

GPU가 정상일 때 장시간 학습:

```bash
python3 -m omokai.train --config configs/rocm_24h.yaml
```

CPU fallback 밤샘 학습:

```bash
python3 -m omokai.train --config configs/overnight_cpu.yaml
```

GUI 플레이:

```bash
python3 -m omokai.gui --checkpoint checkpoints/rocm_unlimited/best.pt
```

## GUI 조작

- 좌클릭: 착수
- `R`: 현재 시드로 재시작
- `S`: 흑/백 전환
- `N` / `P`: 체크포인트 디렉터리 내 다음/이전 모델 로드
- `L`: 체크포인트 디렉터리 재스캔 (학습 중 새로 저장된 체크포인트 반영)
- `M`: AI 즉시 한 수 두기
- `O`: 현재 시드로 오프닝 자동 배치
- `[` / `]`: 오프닝 시드 감소/증가

학습 중에도 `--checkpoint checkpoints/rocm_unlimited` 로 GUI를 실행하면 `L` 키로 최신 iter/latest/best 체크포인트를 즉시 불러와 대국할 수 있다.

## AMD GPU 메모

현재 이 작업 환경은 WSL2이며, Windows 쪽에서 `AMD Radeon RX 9070 XT`가 확인됐다. 다만 이 WSL 세션에는 ROCm용 `torch`가 아직 설치되지 않아, 지금은 CPU fallback으로 동작한다.

AMD 공식 문서 기준으로 `RX 9070 XT`는 WSL에서 ROCm 7.2.1 지원 대상이며, PyTorch 2.9.1이 공식 production support 조합이다. 세부 명령은 `scripts/setup_rocm_wsl.sh`에 정리했다.

## 한계

- 현재 기본 실험 규칙은 `freestyle gomoku`다.
- 금수 규칙은 구현하지 않았다.
- `trainer_state.pt` 재개는 replay buffer와 optimizer state까지 함께 저장한다.
