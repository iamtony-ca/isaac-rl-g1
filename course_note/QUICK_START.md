# G1 Quick Start — PPO 보행 전체 파이프라인 (CLI)

> **목적**: 이 파일 하나만 보고 Unitree G1 velocity 보행을 **학습 → 모니터링 → 재생/평가 → 정책 export**까지 처음부터 끝까지 실습한다.
> **두 트랙 제공**: 🖥️ **GUI**(창이 떠서 눈으로 확인) / ⚡ **Headless**(창 없이 빠르게 학습). 실전 권장 = **학습은 Headless(빠름), 재생은 GUI(관찰)**.
> 상세 이론은 각 단계에서 링크: 개념 [[DAY3_ppo_rsl_rl]] · 튜닝 [[PPO_TUNING]] · 모니터링 [[TENSORBOARD]] · export/sim2real [[DAY6_ros2_sim2real]].

---

## 0. 공통 준비 (딱 한 번)

```bash
cd /isaac-sim/IsaacLab          # ★ 모든 명령은 이 디렉토리에서. 로그도 여기 ./logs/rsl_rl/ 에 쌓인다.
```

- **실행 래퍼**: `./isaaclab.sh -p <스크립트>` = Isaac Sim 내장 파이썬으로 실행(그냥 `python` 아님).
- **G1 태스크 ID** (`--task`에 넣는 값):

| 태스크 ID | 용도 | 특징 |
|---|---|---|
| `Isaac-Velocity-Flat-G1-v0` | 평지 보행 (**첫 실습 추천**) | obs 123, 빠름(~1500 iter) |
| `Isaac-Velocity-Rough-G1-v0` | 거친 지형 | obs 310(height-scan 포함), 느림(~3000 iter) |
| `Isaac-Velocity-Flat-G1-Play-v0` | 재생/평가 전용 | env 적고 랜덤화 축소된 경량판 |
| `Isaac-Velocity-Rough-G1-Play-v0` | 재생/평가 전용 | 〃 |

- **전체 파이프라인 6단계**:
  `①환경 확인 → ②학습 → ③모니터링(TensorBoard) → ④재생/평가 → ⑤정책 export → (선택)⑥이어학습`

> 아래 각 단계마다 🖥️GUI / ⚡Headless 명령을 나란히 둔다. **`--headless` 플래그의 유무가 유일한 핵심 차이**다(창을 띄우느냐).

---

## ① 환경 스모크 테스트 — "일단 뜨는지" 확인 (선택, 1분)

학습 전에 env가 정상 로드되는지 랜덤 정책으로 확인. 등록된 태스크 목록도 볼 수 있다.

```bash
# 등록된 G1 태스크 확인
./isaaclab.sh -p scripts/environments/list_envs.py | grep G1
```

🖥️ **GUI** — 창에서 G1들이 랜덤하게 버둥대는 걸 눈으로 확인:
```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Velocity-Flat-G1-v0 --num_envs 16
```

⚡ **Headless** — 렌더 없이 텐서 shape·에러만 빠르게 검증:
```bash
./isaaclab.sh -p scripts/environments/random_agent.py \
    --task Isaac-Velocity-Flat-G1-v0 --num_envs 16 --headless
```
- 통과 기준: 에러 없이 스텝이 돌면 OK. (아직 학습 아님 — 로봇은 못 걷는 게 정상)

---

## ② 학습 (Training) — PPO로 걷는 정책 만들기

핵심 단계. `train.py`가 [수집→GAE→PPO 업데이트]를 `max_iterations`번 반복한다([[DAY3_ppo_rsl_rl]]).

⚡ **Headless (실전 권장 — 이걸로 학습)**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless \
    --num_envs 4096 --max_iterations 1500
```

🖥️ **GUI (학습 과정을 눈으로 — 느림, 학습 자체엔 비추천)**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 \
    --num_envs 512 --max_iterations 1500
```
- GUI는 렌더링 때문에 **훨씬 느리고** GPU 메모리도 먹으니 `--num_envs`를 줄인다(512 등). 실제 학습은 Headless로, GUI는 "어떻게 생겼나" 구경용.

**자주 쓰는 플래그**:
| 플래그 | 뜻 |
|---|---|
| `--headless` | 창 없이(빠름) |
| `--num_envs N` | 병렬 로봇 수(배치 크기 좌우, [[PPO_TUNING]] §1) |
| `--max_iterations N` | 총 학습 라운드 |
| `--seed N` | 재현용 시드 |
| `--video` | 학습 중 주기적 영상 녹화(headless에서도 됨) |

**PPO 튜닝을 CLI로** (config 파일 안 고치고 hydra override):
```bash
# 예: 엔트로피↑ + adaptive KL 목표↑ (의미는 [[PPO_TUNING]])
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 1500 \
    agent.algorithm.entropy_coef=0.02 agent.algorithm.desired_kl=0.02 \
    env.rewards.track_lin_vel_xy_exp.weight=1.5
```
- `agent.*` = PPO 하이퍼파라미터, `env.rewards.*` = 보상 가중치([[DAY4_reward_shaping]]).

**결과 저장 위치** (자동):
```
logs/rsl_rl/g1_flat/<타임스탬프>/
    ├─ model_<iter>.pt        # 체크포인트(주기적)
    ├─ events.out.tfevents... # TensorBoard 로그
    └─ params/                # 사용된 config 스냅샷
```
(rough는 `g1_rough/`. experiment_name으로 폴더가 갈린다.)

---

## ③ 모니터링 — TensorBoard로 학습 상태 보기

학습을 돌리면서 **다른 터미널**에서:
```bash
cd /isaac-sim/IsaacLab
tensorboard --logdir logs/rsl_rl/g1_flat
# 브라우저에서 http://localhost:6006
```
- **가장 먼저 볼 지표**: `Train/mean_episode_length`(=eplen, 오래 버틸수록 잘 걷는 것 — reward 숫자보다 이걸 봐라, [[DAY3]]).
- 그 외 `Policy/mean_noise_std`(탐험량), `Loss/value_function`(critic 학습), `Loss/learning_rate`(adaptive가 출렁이는 게 정상). 지표 해석 전체는 [[TENSORBOARD]].

---

## ④ 재생 / 평가 (Play) — 학습된 정책으로 걷는 것 보기

`play.py`는 **최신 체크포인트를 자동 로드**해 정책을 실행한다(학습과 달리 탐험 없이 결정론적).

🖥️ **GUI (강력 추천 — 걷는 걸 눈으로 확인)**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16
```
- `--real-time` 붙이면 실제 속도로 재생(관찰하기 좋음):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 --real-time
```

⚡ **Headless (영상 파일로만 저장 — 원격/서버용)**:
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 --headless --video --video_length 400
# 영상: logs/rsl_rl/g1_flat/<run>/videos/ 에 저장
```

**특정 체크포인트 지정** (기본은 최신 자동):
```bash
    --checkpoint /isaac-sim/IsaacLab/logs/rsl_rl/g1_flat/<타임스탬프>/model_1499.pt
# 또는 특정 run 폴더 선택: agent.load_run=<타임스탬프> agent.load_checkpoint=model_1499.pt
```

---

## ⑤ 정책 Export — 배포용 파일 뽑기 (sim2real 입구)

`play.py`는 실행하면서 정책을 **자동으로 export**한다(별도 명령 불필요):
```
logs/rsl_rl/g1_flat/<run>/exported/
    ├─ policy.pt      # TorchScript (결정론적 추론용, critic·PPO 다 버림)
    └─ policy.onnx    # ONNX (obs→actions MLP, ~수백KB)
```
- 즉 **④를 한 번 돌리면 ⑤가 같이 나온다.** 이 `.onnx`/`.pt`가 실제 로봇/ROS2로 넘기는 산출물([[DAY6_ros2_sim2real]]).
- 검증(G1 flat 실측): onnx = `obs[1,123] → actions[1,37]`, Gemm+Elu MLP.

---

## ⑥ (선택) 이어서 학습 (Resume) — 중단점부터 재개

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 3000 \
    --resume agent.load_run=<타임스탬프> agent.load_checkpoint=model_1499.pt
```
- "방향은 맞는데 덜 됐다" 싶으면 max_iterations 늘려 이어서(대부분 "안 됨"은 "덜 됨", [[PPO_TUNING]] §10-5).

---

## 🔁 전체 파이프라인 한 번에 (복붙용 최소 레시피)

⚡ **Headless 트랙 (학습 → 재생/export)**:
```bash
cd /isaac-sim/IsaacLab
# 1) 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 1500
# 2) (다른 터미널) 모니터링
tensorboard --logdir logs/rsl_rl/g1_flat
# 3) 재생 + export (영상 저장)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 --headless --video
# → logs/rsl_rl/g1_flat/<run>/exported/policy.onnx 생성됨
```

🖥️ **GUI 트랙 (눈으로 보며)**:
```bash
cd /isaac-sim/IsaacLab
# 1) 학습(창 뜸, env 줄여서)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --num_envs 512 --max_iterations 1500
# 2) 재생(걷는 것 실시간 관찰) — export도 자동
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 --real-time
```

---

## ⚠️ 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| GUI가 안 뜸 / 느림 | 원격이면 Headless+`--video` 사용. GUI는 `--num_envs` 낮춰라 |
| 메모리 부족(OOM) | `--num_envs` 줄이기(4096→2048…) |
| play가 "checkpoint 없음" | ②를 먼저 완료, 또는 `--checkpoint`로 경로 명시 |
| 학습이 안 걷음 | "덜 됨"일 확률 큼 → max_iterations↑. 그다음 [[PPO_TUNING]] §10 진단표 |
| 로그 폴더 못 찾음 | 반드시 `cd /isaac-sim/IsaacLab`에서 실행(로그가 상대경로 `./logs`) |

> 다음 단계: 튜닝 실험은 [[PPO_TUNING]], 거친 지형은 태스크를 `Rough`로 바꾸고 [[DAY5_sensors]], 4족 Go2는 [[go2/README]].
