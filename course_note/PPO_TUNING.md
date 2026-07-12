# PPO 심화 + rsl_rl 하이퍼파라미터 튜닝 실전

> PPO의 각 개념을 **실제 config 값**과 1:1로 엮고, "이 증상이면 이 값을 이렇게"까지 정리한다.
> 값 출처(실측): `config/g1/agents/rsl_rl_ppo_cfg.py` (G1RoughPPORunnerCfg / G1FlatPPORunnerCfg).
> 선행: [[DAY0_foundations]](비유), [[DAY3_ppo_rsl_rl]](개념), 모니터링 [[TENSORBOARD]].

---

## ⚠️ 먼저 알아야 할 실전 진실

**Isaac Lab 보행에서 튜닝의 90%는 reward([[DAY4_reward_shaping]])이고, PPO 하이퍼파라미터는 대부분 기본값을 그대로 쓴다.**
- rsl_rl 기본값은 legged locomotion에 잘 맞춰져 있고, 가장 까다로운 **learning rate는 adaptive로 자동 조절**된다.
- 그래서 "학습이 안 된다"의 첫 원인은 보통 PPO가 아니라 **reward 설계·관절 이름·게인**이다.
- 이 노트의 목적: PPO 값을 **이해**하고, 정말 필요할 때 **어디를 건드릴지** 아는 것. (마구 돌리는 곳 아님)

세 종류의 config가 있다:
| 클래스 | 무엇 | 파일 |
|---|---|---|
| `RslRlOnPolicyRunnerCfg` | 학습 루프(수집량·반복) | rsl_rl_ppo_cfg.py 최상단 |
| `RslRlPpoActorCriticCfg` (`policy`) | 신경망 구조·탐험 | `policy=` 블록 |
| `RslRlPpoAlgorithmCfg` (`algorithm`) | PPO 업데이트 규칙 | `algorithm=` 블록 |

---

## 0-B. PPO 손잡이 개념 사전 (비유로)

수식 없이 "이게 대체 뭐냐"부터. 자세한 튜닝은 아래 각 절.

- **`clip_param` (0.2) = 한 번에 바꾸는 폭의 하드 한계.**
  advantage가 +면 그 행동 확률을 왕창 올리고 싶지만, 적은 경험으로 과하게 밀면 학습이 망가진다. 그래서 새 정책이 그 행동을 할 확률을 옛 정책 대비 `ratio`로 보고 **[0.8, 1.2] 밖은 잘라낸다(±20%)**.
  *비유: 시험 한 번 잘 봤다고 공부법을 3배로 늘리지 말고 최대 20%만 조정.* → **딱딱한 벽.** 거의 0.2 고정.

- **`learning_rate` (1e-3) = 보폭.**
  그래디언트가 "어느 방향으로 뇌를 고칠까"를 주면, lr은 **그 방향으로 얼마나 크게 한 걸음** 내딛나.
  *비유: 안개 낀 산에서 내리막으로 내려가기 — 보폭이 크면 골짜기를 건너뛰어 튕기고(발산), 작으면 기어간다.*
  ⚠️ rsl_rl은 `schedule="adaptive"`라 이건 **시작값일 뿐, 자동으로 오르내린다** → 내가 직접 정하는 값이 아님.

- **KL (KL divergence) = "이번에 뇌가 얼마나 바뀌었나"를 재는 자.**
  업데이트 전/후 정책(행동분포)이 얼마나 달라졌는지의 수치. KL≈0=거의 안 바뀜, 큼=많이 바뀜.
  PPO의 "많이 바꾸지 마" 원칙에서 **실제 측정 도구**.

- **`desired_kl` (0.01) = 그 변화량의 목표치 = 내가 실제로 만지는 손잡이.**
  rsl_rl이 매 업데이트 후 KL을 재서 `desired_kl`보다 크면 **lr 자동 감소**, 작으면 **증가** → lr을 이 목표에 맞춘다.
  즉 **desired_kl↑ → lr↑(공격적), desired_kl↓ → lr↓(보수적).** (실측: [[PPO_TUNING]] desired_kl 부록 — 0.002면 lr 바닥, 0.05면 lr 80배)
  *clip은 행동별 하드 컷, KL+adaptive lr은 업데이트 전체 보폭을 무르게 조절 = 두 겹 안전장치.*

- **`entropy_coef` (0.008) = 탐험 유지(호기심) 강도.** 결과가 `Policy/mean_noise_std`(탐험량 σ)로 나타난다.
  낮으면 σ↓(수렴), 높으면 σ↑(계속 흔듦). (실측: 부록)

> 모니터링 지표(eplen·noise_std·reward)의 뜻은 [[TENSORBOARD]] "지표 개념 풀이".

---

## 1. 데이터 수집 — 배치 크기 (Runner)

| config | G1 값 | 개념 |
|---|---|---|
| `num_steps_per_env` | 24 | 한 env를 몇 스텝 굴려 경험을 모으나(rollout 길이) |
| `--num_envs` (CLI) | 4096(본)/512(실험) | 병렬 로봇 수 |
| `max_iterations` | 1500(flat)/3000(rough) | 수집+학습 라운드 총 횟수 |

- **배치 크기 = num_envs × num_steps_per_env** (4096×24 = 98,304 경험/iteration).
- 이 경험으로 그래디언트(뇌 수정 방향)를 추정 → **배치 클수록 방향이 정확·안정, 대신 iteration당 느림.**

**튜닝**
- 학습이 **지글지글 불안정** → num_envs↑ 또는 num_steps_per_env↑ (배치↑ = 안정↑).
- **너무 느림/메모리 부족** → num_envs↓. 단 너무 작으면(≤256) 노이즈 커짐.
- `num_steps_per_env`는 보행에서 **24~48**이 흔함. 길수록 긴 시야의 신용할당에 유리하나 메모리↑.
- **가장 먼저 할 것**: 학습이 방향은 맞는데 덜 됐으면 → **max_iterations를 늘려 더 오래** 돌린다(웬만한 "안 됨"은 사실 "덜 됨").

---

## 2. 신경망 & 탐험 (policy = ActorCritic)

| config | G1 값 | 개념 |
|---|---|---|
| `actor_hidden_dims` | [512,256,128] rough / [256,128,128] flat | actor(뇌) 크기 = 표현력 |
| `critic_hidden_dims` | 위와 동일 | critic(코치) 크기 |
| `activation` | "elu" | 비선형 함수(보통 안 건드림) |
| `init_noise_std` | 1.0 | **초기 탐험량 σ**(주사위 퍼짐) |
| `actor_obs_normalization` | False | 관찰 정규화 사용 여부 |

- **왜 rough가 flat보다 망이 큰가**: rough는 observation이 310차원(height-scan 포함, [[DAY5_sensors]])이라 더 복잡한 함수를 배워야 함 → 표현력↑ 필요.

**튜닝**
- 태스크가 **복잡(비전·거친지형)** 인데 학습이 정체 → 망 크기↑ ([512,256,128] 등).
- **초반에 탐험을 안 함**(금방 한 행동으로 굳음) → `init_noise_std`↑ (예 1.0→1.5).
- 관찰 스케일이 항목마다 제각각이라 불안정 → `actor_obs_normalization=True`(단, 배포 시 normalizer 포함 필요, [[DAY6_ros2_sim2real]]).
- **주의**: 망을 키우면 배포 추론도 무거워지고 sim-to-real 시 과적합 위험. 필요할 때만.

---

## 3. PPO 업데이트의 심장 (algorithm)

### 3-1. `clip_param` = 0.2 — 신뢰영역(한 번에 얼마나 바꿀까)
- 행동 확률을 옛 정책 대비 **±20% 넘게 못 바꾸게 잘라냄** = PPO 안정의 핵심.
- **튜닝**: 거의 안 건드림. 학습이 **너무 공격적으로 붕괴**하면 0.2→0.1(더 보수적), **너무 굼뜨면** 0.2→0.3(위험). 보통 0.2 고정.

### 3-2. `entropy_coef` = 0.008 — 탐험 유지(호기심)
- 엔트로피(무작위성)에 주는 보너스. **탐험이 꺼지는 걸 늦춘다.**
- **너무 낮으면**: 정책이 **일찍 굳어** 나쁜 동작에 수렴(DAY4의 "얼어붙은 로봇"과 사촌).
- **너무 높으면**: 계속 흔들려 **수렴을 안 함**(걸음이 안 정리됨).
- **튜닝**(실제로 만지는 몇 안 되는 값): 
  - `Policy/mean_noise_std`가 **너무 빨리 0으로** 떨어지고 성능 정체 → entropy_coef↑ (0.008→0.02).
  - 끝까지 걸음이 **덜덜 떨리고 안 정리** → entropy_coef↓ (0.008→0.003).

### 3-3. `learning_rate` = 1e-3 + `schedule="adaptive"` + `desired_kl` = 0.01
- lr = 한 번에 뇌를 얼마나 크게 수정하나(보폭).
- **adaptive**: 매 업데이트 후 **KL divergence**(뇌가 얼마나 바뀌었나)를 재서, `desired_kl`(0.01)보다 크면 lr **자동 감소**, 작으면 **증가**. → **lr을 손으로 안 맞춰도 됨.**
- **핵심 튜닝 손잡이는 lr이 아니라 `desired_kl`**:
  - 학습을 **더 공격적·빠르게** → desired_kl↑ (0.01→0.02). 불안정 위험.
  - **더 안정적·느리게** → desired_kl↓ (0.01→0.005).
- 관찰: `Loss/learning_rate` 그래프가 오르내리는 게 정상(adaptive가 일하는 것).

### 3-4. `value_loss_coef` = 1.0 / `use_clipped_value_loss` = True
- 코치(critic) 학습을 전체 손실에서 얼마나 비중 두나. 보통 1.0 고정.
- **튜닝**: `Loss/value_function`이 **안 줄고 계속 큼**(코치가 상황을 못 맞춤) → 관찰 정규화 켜기, 망 키우기 검토. coef 자체는 잘 안 건드림.

### 3-5. `max_grad_norm` = 1.0 — 그래디언트 클리핑(폭발 방지)
- 수정 방향이 갑자기 튀면 잘라 발산을 막음. 안전장치. **거의 안 건드림.**

---

## 4. 모은 경험을 몇 번 우려먹나 (algorithm)

| config | G1 값 | 개념 |
|---|---|---|
| `num_learning_epochs` | 5 | 같은 rollout을 몇 번 반복 학습 |
| `num_mini_batches` | 4 | 한 번에 다 말고 몇 조각으로 나눠 |

- 한 iteration의 그래디언트 스텝 = epochs × mini_batches = **5 × 4 = 20회**.
- **epochs↑**: 경험을 더 짜냄(샘플 효율↑) but **그 rollout에 과적합** → KL이 튀고 불안정.
- **튜닝**: 학습이 **불안정하고 KL이 자꾸 큼** → num_learning_epochs↓ (5→3). 데이터가 아까우면(수집 비쌈) ↑도 가능하나 5가 무난.

---

## 5. 신용 할당 — 시야 (algorithm)

| config | G1 값 | 개념 |
|---|---|---|
| `gamma` (γ) | 0.99 | 할인율 = **미래를 얼마나 멀리 보나** |
| `lam` (λ) | 0.95 | GAE의 bias-variance 절충 |

- **gamma 0.99** ≈ 유효 시야 약 100스텝(=2초, 50Hz). 높을수록 먼 미래 중시(예지력↑) but 학습 어려움·노이즈↑.
- **lam 0.95**: advantage 추정을 얼마나 부드럽게. 낮으면 편향↑·분산↓, 높으면 반대.
- **튜닝**: 웬만하면 **고정**. 아주 긴 호흡의 태스크가 아니면 0.99/0.95가 정석.

---

## 6. 실전 튜닝 순서 (증상 → 손잡이)

**0순위 — PPO 건드리기 전에**: reward([[DAY4_reward_shaping]]), 관절/링크 이름, 게인([[DAY2_physics_actuators]])부터 의심. 대부분 여기.

**증상별 진단표** (TensorBoard 신호 → 손잡이):
| 증상 | TB 신호 | 먼저 시도 |
|---|---|---|
| 방향은 맞는데 덜 됨 | `mean_episode_length` 완만 상승 중 | **max_iterations↑ (더 오래)** |
| 초반 탐험을 안 함, 금방 굳음 | `Policy/mean_noise_std` 급강하 | `entropy_coef`↑, `init_noise_std`↑ |
| 끝까지 덜덜 떨림, 수렴 안 함 | 보상 곡선 진동 | `entropy_coef`↓ |
| 불안정·붕괴 | 보상 급락, `Loss/*` 튐 | `desired_kl`↓, `num_learning_epochs`↓ |
| 너무 느리게 배움 | 곡선 완만 | `desired_kl`↑ (조심), 배치↑ |
| 지글지글 노이즈 | 곡선 거칢 | **배치↑**(num_envs 또는 num_steps↑) |
| 코치가 상황 못 맞춤 | `Loss/value_function` 안 줄음 | 관찰 정규화, 망↑ |
| 복잡 태스크 정체 | rough/vision에서 정체 | 망 크기↑ |

**한 번에 하나씩** 바꾸고 재학습해서 비교 — reward 튜닝과 같은 황금률(DAY4).

---

## 7. 한눈 요약 — 개념 ↔ config

| PPO 개념 | config (G1 값) | 만질 일 |
|---|---|---|
| 탐험(주사위 퍼짐) | `init_noise_std` 1.0 | 가끔 |
| 탐험 유지(호기심) | `entropy_coef` 0.008 | **종종** |
| 한 번에 바꾸는 폭(신뢰영역) | `clip_param` 0.2 | 거의 안 |
| 보폭 자동조절 | `learning_rate` 1e-3 + `desired_kl` 0.01 | **desired_kl로** |
| 코치 학습 | `value_loss_coef` 1.0 | 거의 안 |
| 발산 방지 | `max_grad_norm` 1.0 | 안 |
| 경험 재사용 | `num_learning_epochs` 5 / `num_mini_batches` 4 | 가끔 |
| 시야(신용할당) | `gamma` 0.99 / `lam` 0.95 | 거의 안 |
| 배치·학습량 | `num_steps_per_env` 24 / `num_envs` / `max_iterations` | **자주(양·속도)** |
| 표현력 | `actor/critic_hidden_dims` | 태스크 난이도 따라 |

**실전 요령**: 자주 만지는 건 **max_iterations·num_envs(양)** 와 **entropy_coef·desired_kl(학습 성향)**. 나머지는 기본값이 정답인 경우가 대부분.

---

## 실험 명령 (hydra로 config 안 고치고 바로)
```bash
cd /isaac-sim/IsaacLab
# 예: 엔트로피↑ + 더 오래 + adaptive KL 목표↑
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 4096 --max_iterations 2000 \
  agent.algorithm.entropy_coef=0.02 agent.algorithm.desired_kl=0.02
```
- reward는 `env.rewards.*`, PPO는 `agent.algorithm.*` / `agent.policy.*` / `agent.*` 로 override.

---

## 부록 — 직접 돌린 실험: `entropy_coef`가 `mean_noise_std`를 조종한다

동일 조건(flat G1, seed 1, 300 iter, num_envs 512)에서 **entropy_coef만** 0.0 vs 0.05로:

| iter | ent=0.0 · noise_std | ent=0.05 · noise_std | ent=0.0 · eplen | ent=0.05 · eplen |
|---:|---:|---:|---:|---:|
| 0 | 1.00 | 1.00 | 13 | 13 |
| 100 | 0.91 ↓ | 1.21 ↑ | 46 | 35 |
| 200 | 0.84 ↓ | 1.43 ↑ | 60 | 39 |
| 299 | **0.79** ↓ | **1.64** ↑ | **116** | **38** |

**관찰**
- `entropy_coef=0.0`: 탐험량 σ가 **1.00 → 0.79로 감소**(정책이 한 동작으로 좁혀감=수렴). eplen 13→**116**로 잘 걷게 됨.
- `entropy_coef=0.05`: 엔트로피 보너스가 σ를 **1.00 → 1.64로 오히려 키움**(계속 흔들기 강제). eplen 13→**38에서 정체** — 너무 흔들려 걸음을 못 정리함.
- reward 절대값은 둘 다 ~−5.6로 비슷 → 3일차 교훈대로 **eplen(116 vs 38)이 진짜 신호.**

**교훈 (균형이 핵심)**
- `entropy_coef`는 **탐험 유지 강도** 다이얼이고, `mean_noise_std`가 그 결과다: 낮으면 σ↓(수렴), 높으면 σ↑(발산적 탐험).
- ⚠️ **"그럼 0이 최고?" 아니다.** 이 짧은 flat 실험에선 일찍 수렴이 유리했을 뿐, entropy=0은 어려운 태스크에서 **나쁜 지역최적에 조기 수렴(premature convergence)** 위험이 크다. 그래서 기본값이 작은 양수(**0.008**)인 것 = 수렴과 탐험의 **중간**.
- 실전: `Policy/mean_noise_std`가 너무 빨리 떨어지고 성능 정체 → entropy_coef↑ / 끝까지 안 정리되고 σ가 높게 유지 → entropy_coef↓. (로그: `logs/exp_ent_low.log`, `logs/exp_ent_high.log`)

---

## 부록 — 직접 돌린 실험: `desired_kl`이 `learning_rate`를 자동으로 몬다

동일 조건(flat G1, seed 1, 300 iter)에서 **desired_kl만** 0.002 vs 0.05. lr은 stdout에 없어 TB 이벤트파일(`Loss/learning_rate`)에서 추출:

| iter | lr (kl=0.002) | lr (kl=0.05) | eplen (kl=0.002) | eplen (kl=0.05) |
|---:|---:|---:|---:|---:|
| 0 | 1.0e-5 | 2.0e-4 | 13 | 13 |
| 100 | 1.0e-5 | 1.1e-4 | 42 | 63 |
| 200 | 1.0e-5 | 8.6e-4 | 45 | 173 |
| 299 | **1.0e-5** (바닥에 붙음) | **~1e-4 ~ 8.6e-4** (높게 진동) | **45** (정체) | **475** (빠른 학습) |

**메커니즘 (KL 온도조절기 실증)**
- rsl_rl은 매 업데이트 후 실제 KL을 재서 lr을 ×/÷ 하며 `desired_kl`에 맞춘다(lr 범위 대략 [1e-5, 1e-2]).
- **desired_kl=0.002**("아주 조금만 바꿔"): 작은 스텝도 KL이 0.002를 넘어서 → 스케줄러가 lr을 계속 줄여 **최소값 1e-5에 붙어버림** → 정책이 거의 안 변함(noise_std 1.00→0.98) → **학습이 기어감(eplen 45 정체).**
- **desired_kl=0.05**("크게 바꿔도 돼"): KL이 여유 있어 lr을 키움(최대 8.6e-4, ~80배) → 빠른 학습(**eplen 475**, ≈9.5초 버팀).

**교훈**
- **손잡이는 lr이 아니라 `desired_kl`**이다 — lr은 그걸 맞추려 자동으로 따라온다(같은 방향: desired_kl↑ → lr↑).
- ⚠️ 이 쉬운 flat 태스크에선 공격적(0.05)이 유리했지만, **rough·비전 등 어려운 태스크에선 큰 lr이 붕괴(collapse)를 부를 수 있다.** desired_kl=0.002는 반대로 너무 소심해 lr이 바닥에 붙어 기어감. 기본값 **0.01이 안전한 중간.**
- (로그: `logs/exp_kl_low.log`, `logs/exp_kl_high.log`)
