# omokai web demo

학습된 모델을 브라우저에서 그대로 돌립니다. ONNX Runtime Web (WASM 백엔드) +
JS로 포팅한 MCTS를 사용합니다. 친구한테 링크/zip만 주면 바로 둘 수 있습니다.

## 빌드 (모델 변환)

리포 루트에서:

```bash
.venv-rocm/bin/python web/export_onnx.py --best-iteration 115
```

`checkpoints/rocm_24h/*.pt` 전체를 `web/models/*.onnx` 로 변환하고
`web/models/manifest.json` 을 갱신합니다. 다른 디렉터리를 쓰려면
`--ckpt-dir` / `--out-dir` 플래그로 지정하세요.

## 로컬 실행

```bash
.venv-rocm/bin/python web/serve.py --port 8080
# 또는 그냥
python3 -m http.server -d web 8080
```

브라우저로 `http://localhost:8080` 접속.

## 배포 (정적 호스팅)

`web/` 디렉터리 통째로 어디든 올리면 됩니다:

- **GitHub Pages**: `web/` 내용을 `gh-pages` 브랜치 또는 `docs/` 폴더로 푸시
- **Netlify / Vercel / Cloudflare Pages**: 빌드 명령 없이 `web/` 폴더만 publish
- **S3 / 호스팅 버킷**: `web/` 통째로 업로드, `index.html` 을 인덱스로

ONNX 파일이 정적 자원이라 CDN 친화적입니다 (모델 19개 합쳐서 ~36MB).
브라우저는 선택한 체크포인트 1개만 다운로드하므로 첫 로딩은 ~2MB 정도입니다.

## 파일 구조

```
web/
  index.html        # UI
  styles.css
  app.js            # 메인 컨트롤러 (보드 렌더링, 입력, 게임 상태 관리)
  board.js          # GameState (omokai/board.py 포팅)
  features.js       # 입력 plane 인코딩 (omokai/features.py 포팅)
  mcts.js           # MCTS (omokai/mcts.py 포팅, async)
  evaluator.js      # onnxruntime-web 래퍼
  export_onnx.py    # PyTorch → ONNX 변환 스크립트
  serve.py          # 로컬 정적 서버
  models/
    manifest.json
    best.onnx, iter_XXXX.onnx, latest.onnx ...
```

## 컨트롤

- **체크포인트**: 드롭다운에서 모델 선택. ⭐ 표시는 학습이 best로 채택한 가중치.
- **내 색**: 흑(선) / 백(후)
- **AI 시뮬레이션**: 16~800 (높을수록 강하고 느림). 9×9면 200 정도면 충분히 대국 가능.
- **새 게임 / 한 수 무르기**

## 한계

- WASM 단일 스레드. WebGPU 백엔드를 켜면 더 빠르지만 호환성 이슈 있음.
- 학습이 131 iteration에서 멈춘 미성숙 상태이므로 가끔 어색한 수가 나옵니다.
- 기본은 AlphaZero식 정책망 + MCTS만 사용. 4-3, 5목 같은 전술 룰 하드코딩 없음.
