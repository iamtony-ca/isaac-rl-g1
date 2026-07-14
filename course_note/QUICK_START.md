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
# 예(문법 데모일 뿐): 아래는 3개를 동시에 바꾼 것 — 실제 튜닝은 "한 번에 하나씩" (PPO_TUNING §12).
# 명시한 토큰만 기본값을 덮어쓰고, 안 적은 값은 전부 아래 "기본값" 그대로 사용된다.
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 1500 \
    agent.algorithm.entropy_coef=0.02 agent.algorithm.desired_kl=0.02 \
    env.rewards.track_lin_vel_xy_exp.weight=1.5
```
- `agent.*` = PPO 하이퍼파라미터, `env.rewards.*` = 보상 가중치([[DAY4_reward_shaping]]).

<details>
<summary><b>📋 기본값(default) 파라미터 전체 — 클릭해서 펼치기</b> (아무것도 override 안 하면 이 값으로 학습)</summary>

실측 출처: `config/g1/agents/rsl_rl_ppo_cfg.py` (`G1FlatPPORunnerCfg` / `G1RoughPPORunnerCfg`). **override 경로에 그대로 대입하면 튜닝**할 수 있다.

| 파라미터 | hydra override 경로 | **Flat 기본** | **Rough 기본** | 의미/절 |
|---|---|---|---|---|
| 병렬 로봇 수 | `--num_envs` (또는 `env.scene.num_envs`) | 4096 | 4096 | 배치 크기 [[PPO_TUNING]] §1 |
| rollout 길이 | `agent.num_steps_per_env` | 24 | 24 | 수집 스텝 §1 |
| 총 학습 반복 | `--max_iterations` (또는 `agent.max_iterations`) | 1500 | 3000 | 학습량 §1 |
| 체크포인트 주기 | `agent.save_interval` | 50 | 50 | 저장 간격 |
| **[정책망]** | `agent.policy.*` | | | §2 |
| 초기 탐험 σ | `agent.policy.init_noise_std` | 1.0 | 1.0 | 출발 탐험량 §2 |
| actor 망 | `agent.policy.actor_hidden_dims` | [256,128,128] | [512,256,128] | 표현력 §2 |
| critic 망 | `agent.policy.critic_hidden_dims` | [256,128,128] | [512,256,128] | 〃 |
| 활성함수 | `agent.policy.activation` | elu | elu | 보통 고정 |
| 관찰 정규화(actor) | `agent.policy.actor_obs_normalization` | False | False | §2 |
| 관찰 정규화(critic) | `agent.policy.critic_obs_normalization` | False | False | §2 |
| **[PPO 규칙]** | `agent.algorithm.*` | | | §3 |
| 신뢰영역 clip | `agent.algorithm.clip_param` | 0.2 | 0.2 | 거의 고정 §3-1 |
| 탐험 유지 | `agent.algorithm.entropy_coef` | 0.008 | 0.008 | **종종 튜닝** §3-2 |
| lr 시작값 | `agent.algorithm.learning_rate` | 1.0e-3 | 1.0e-3 | adaptive라 시작값일 뿐 §3-3 |
| lr 스케줄 | `agent.algorithm.schedule` | adaptive | adaptive | §3-3 |
| KL 목표 | `agent.algorithm.desired_kl` | 0.01 | 0.01 | **진짜 lr 손잡이** §3-3 |
| 경험 재사용 | `agent.algorithm.num_learning_epochs` | 5 | 5 | 가끔 §4 |
| 미니배치 수 | `agent.algorithm.num_mini_batches` | 4 | 4 | §4 |
| 할인율 | `agent.algorithm.gamma` | 0.99 | 0.99 | 시야 §5 |
| GAE λ | `agent.algorithm.lam` | 0.95 | 0.95 | bias/variance §5 |
| critic 비중 | `agent.algorithm.value_loss_coef` | 1.0 | 1.0 | 거의 고정 §3-4 |
| value clip | `agent.algorithm.use_clipped_value_loss` | True | True | §3-4 |
| grad 클립 | `agent.algorithm.max_grad_norm` | 1.0 | 1.0 | 안전장치 §3-5 |

> **Flat vs Rough 차이는 3개뿐**: `max_iterations`(1500 vs 3000), `actor/critic_hidden_dims`(작음 vs 큼). 나머지 PPO 값은 **완전히 동일**. (Flat은 `G1FlatPPORunnerCfg`가 `G1RoughPPORunnerCfg`를 상속해 이 셋만 덮어씀)
> **환경/물리 기본값**: control dt 0.02s(50Hz), 물리 dt 0.005s(decimation 4), obs Flat 123 / Rough 310, action 37. 보상 항 기본 가중치는 `velocity_env_cfg.py` + `config/g1/{flat,rough}_env_cfg.py` 참고([[DAY4_reward_shaping]] 부록 A: 상속 3단계).

</details>

**결과 저장 위치** (자동):
```
logs/rsl_rl/g1_flat/<타임스탬프>/
    ├─ model_<iter>.pt        # 체크포인트(주기적)
    ├─ events.out.tfevents... # TensorBoard 로그
    └─ params/                # 사용된 config 스냅샷
```
(rough는 `g1_rough/`. experiment_name으로 폴더가 갈린다.)

<details>
<summary><b>📊 실제 실행 예시 (100 iter) + 결과 해석 — 클릭해서 펼치기</b> (위 튜닝 CLI를 RTX 5090에서 돌린 실측)</summary>

돌린 명령 (검증용으로 `--max_iterations 100`, 튜닝 override 3개):
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 100 \
    agent.algorithm.entropy_coef=0.02 agent.algorithm.desired_kl=0.02 \
    env.rewards.track_lin_vel_xy_exp.weight=1.5
```

마지막(iter 99/100) 출력 — **정상 완주 시 이렇게 생겼다**:
```
                      Learning iteration 99/100
       Computation: 117699 steps/s (collection: 0.784s, learning 0.051s)
             Mean action noise std: 1.62
          Mean value_function loss: 0.0069
               Mean surrogate loss: -0.0264
                       Mean reward: -5.97
               Mean episode length: 67.15
  Episode_Reward/track_lin_vel_xy_exp: 0.0303      ← 실제 태스크 보상(+)
  Episode_Reward/termination_penalty:  -0.2000     ← 지금 제일 큰 벌점(넘어짐)
  Metrics/base_velocity/error_vel_xy:   0.1092     ← 전진속도 오차 ~0.1 m/s
  Metrics/base_velocity/error_vel_yaw:  0.5612     ← 회전은 아직 못 따라감
    Episode_Termination/base_contact:   1.0000     ← 지금은 다 넘어져서 끝남
        Episode_Termination/time_out:   0.0000
                   Total timesteps: 9830400  (=4096×24×100)
                     Training time: 87.56 seconds
```

**이 숫자 읽는 법** (자세한 이론은 [[PPO_TUNING]] · [[TENSORBOARD]]):

| 지표 | 값 | 의미 |
|---|---|---|
| **Mean episode length** | **67.15** | ★ 학습 성공의 진짜 신호. 11→67로 상승 중(100 iter라 아직 초반, 완보는 ~1500 iter) |
| Mean reward | −5.97 | 음수는 **정상** — 벌점 합이 커서. **이 숫자로 판단하지 말 것**([[DAY3]]) |
| Mean action noise std | 1.62 | 초기 1.0보다 **올라감** = `entropy_coef=0.02`(기본 0.008↑)의 직접 증거(탐험 강화) |
| base_contact / time_out | 1.0 / 0.0 | 지금은 거의 다 넘어짐. 학습되면 base_contact↓·time_out↑로 뒤집힘 |
| error_vel_xy | 0.109 | 전진속도는 꽤 따라감. yaw(0.56)는 아직 |
| Computation | 117k steps/s | collection 0.784s ≫ learning 0.051s → **시뮬이 병목**(그래서 num_envs↑ = 주로 collection↑) |

> **한 줄**: `eplen이 오르고`, `error_vel_xy가 줄고`, 에러 없이 iter가 끝났으면 **정상**. reward 숫자·초반 base_contact=1.0에 겁먹지 말 것. 여기서 `--max_iterations 1500`으로 늘리면 실제로 걷는 정책이 된다.

</details>

---

## ③ 모니터링 — TensorBoard로 학습 상태 보기

학습을 돌리면서 **다른 터미널**에서:
```bash
cd /isaac-sim/IsaacLab
# ✅ 이 환경에서 되는 유일한 방법 — isaaclab 파이썬으로 실행(필요한 의존성 PYTHONPATH가 세팅됨)
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/g1_flat
# 브라우저에서 http://localhost:6006  (원격이면 --bind_all 또는 ssh -L 6006:localhost:6006)
```
> ⚠️ `tensorboard`(시스템 PATH)나 `/isaac-sim/kit/python/bin/tensorboard`(bin 직접)는 **안 됨**:
> 전자는 `command not found`, 후자는 `ModuleNotFoundError: markupsafe`(kit 파이썬 site-packages만 봐서). 반드시 `./isaaclab.sh -p -m tensorboard.main` 형태로.
> `TensorFlow installation not found` 경고는 정상(무시). 별칭: `alias tensorboard='/isaac-sim/IsaacLab/isaaclab.sh -p -m tensorboard.main'`
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
# 예: 특정 런의 중간 체크포인트(model_50)를 --checkpoint 로 직접 지정
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 16 \
    --checkpoint /isaac-sim/IsaacLab/logs/rsl_rl/g1_flat/2026-07-15_00-46-58/model_50.pt
# 또는 특정 run 폴더 선택: agent.load_run=<타임스탬프> agent.load_checkpoint=model_1499.pt
```
> 어떤 체크포인트가 로드됐는지는 실행 로그의 `[INFO]: Loading model checkpoint from: ...` 줄로 확인.

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

## ⑥ (선택) 이어서 학습 (Resume) — 처음부터 안 하고 체크포인트에서 재개

불만족스러운 결과를 **처음부터 다시 할 필요 없이**, 특정 체크포인트에서 이어서 학습할 수 있다. `runner.load()`가 **정책+critic 가중치·optimizer 상태·iteration 번호까지 전부 복원**해 끊김 없이 이어간다.

```bash
# 예: 방금 100-iter 런(model_99)에서 이어서 1400회 더 → iter 99→1499
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 1400 \
    --resume --load_run 2026-07-15_00-46-58 --checkpoint model_99.pt
```
> ⚠️ **함정(실측): `agent.resume=true` (hydra)로는 resume 안 됨 — 반드시 `--resume` 플래그.**
> `cli_args.py`에서 `--resume`가 `store_true, default=False`라, 플래그를 안 주면 `agent_cfg.resume`을 **False로 덮어써** hydra override(`agent.resume=true`)를 무효화한다. (`--load_run`/`--checkpoint`는 기본 None이라 hydra로 줘도 살아남지만, resume 자체가 false면 로드가 안 일어남)
> **resume 확인 3신호**: ① `[INFO]: Loading model checkpoint from:` 로그, ② 첫 iteration이 **99/199**처럼 0이 아닌 값에서 시작, ③ `params/agent.yaml`에 `resume: true`. (셋 다 아니고 model_0부터 저장되면 fresh로 돈 것)
> train의 `--checkpoint`는 **파일명**(`model_99.pt`), play의 `--checkpoint`는 **전체경로** — 서로 다름.

**★ 꼭 주의 — `--max_iterations`는 "추가" 횟수지 총량이 아님** (실측: `tot_iter = start_iter + num_learning_iterations`)
- model_99(=iter 99)에서 resume + `max_iterations 1400` → **99 → 1499**까지 (1400회 더 돎).
- "총 1500 채우고 싶다" = 이미 100 했으니 **추가 1400**을 넣는 것.
- 로그는 **새 타임스탬프 폴더**에 쌓임(옛 가중치를 불러와 새 폴더에 저장). 원본 런은 보존됨.

**언제 resume vs 언제 처음부터?**

| 상황 | 권장 |
|---|---|
| 방향 맞는데 **덜 학습됨**(eplen 상승 중) | ✅ **resume** — 그냥 더 돌리기 |
| 하이퍼파라미터 **살짝** 조정(warm-start) | ✅ resume 가능(가중치 유지+새 설정 적용) |
| **튜닝 A/B 공정 비교** | ❌ 처음부터(같은 출발점이어야, [[PPO_TUNING]] §12) |
| **reward/observation 크게 변경** | ❌ 처음부터(critic이 옛 reward 기준이라 꼬임) |
| 정책이 **나쁜 곳에 붕괴/조기수렴** | ❌ 처음부터([[PPO_TUNING]] §10-2) |

> 판단 한 줄: **"같은 목표로 더 오래"면 resume, "설정 바꿔 다른 결과 보려는 실험"이면 처음부터.** 대부분의 "안 됨"은 사실 "덜 됨"이라 resume가 답인 경우가 많다([[PPO_TUNING]] §10-5).

<details>
<summary><b>📊 resume 실제 검증 결과 (model_99 + 100 iter) + 해석 — 클릭해서 펼치기</b></summary>

위 `--resume` 명령을 검증용으로 100회만 이어서 돌린 실측(첫 런 `2026-07-15_00-46-58`의 model_99, eplen 67에서 이어받음):
```
                      Learning iteration 198/199      ← ★ 99부터 시작 = resume 성공 (fresh면 99/100)
             Mean action noise std: 1.23              ← 복원된 1.62에서 하강(수렴 중)
                       Mean reward: -9.38             ← 더 음수지만 함정(아래)
               Mean episode length: 970.49            ← 67→970, 거의 완주!
  Episode_Reward/track_lin_vel_xy_exp: 0.5993         ← 0.03→0.60, 속도 실제 추종
      Episode_Termination/time_out:   0.8944          ← 89%가 끝까지 버팀
  Episode_Termination/base_contact:   0.1056          ← 넘어져 끝나는 비율 급감
  Metrics/base_velocity/error_vel_yaw: 4.5433         ← 회전 추종은 아직(199 iter라 당연)
```

| 지표 | 이어받기 전 | 지금(198) | 의미 |
|---|---|---|---|
| eplen | ~67 | **970** | 거의 최대 에피소드 생존 |
| time_out / base_contact | 0.0 / 1.0 | **0.89 / 0.11** | termination이 넘어짐→완주로 **뒤집힘 = 걷는다** |
| track_lin_vel_xy_exp | 0.03 | **0.60** | 속도 명령 추종(가만히 있으면 0) |
| noise_std | 1.62 | 1.23 | 복원된 σ가 하강 = 수렴 진행 |

**해석 포인트**:
- **resume 확인**: 첫 iteration이 `99/199`(0 아님)로 시작 → optimizer·iteration까지 복원돼 끊김 없이 이어짐.
- **★ reward −6→−9.38 함정(또!)**: 걷기 좋아졌는데 reward는 더 음수. 로봇이 970스텝 활발히 움직이니 `action_rate`(−0.63) 등 페널티가 긴 에피소드에 누적된 것. **eplen·time_out이 진실, reward 숫자 아님**([[DAY3]]).
- warm-start(model_99) + 낮은 기본 entropy(0.008)로 100 iter만에 급수렴. yaw 추종은 본 학습(1500)까지 가야 다듬어짐.

</details>

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
