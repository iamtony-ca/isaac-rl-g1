# 3일차 심화 — 강화학습 기초 + PPO + rsl_rl

> 목표: 1·2일차의 **observation/action/물리**가 **PPO**로 어떻게 "걷는 정책"이 되는지, 그리고 rsl_rl의 실제 하이퍼파라미터가 무엇을 뜻하는지 이해한다.
> 실측 근거: `config/g1/agents/rsl_rl_ppo_cfg.py`, `flat_env_cfg.py`, 그리고 **이 환경에서 직접 돌린 50-iter 학습 로그**(RTX 5090).

---

## 1. MDP로 다시 보는 G1 보행 (1일차 복습 + 수치)

강화학습 = **MDP**(Markov Decision Process) 풀기: 상태 s에서 행동 a를 하면 보상 r과 다음 상태 s'.

이 환경의 실제 차원(Flat G1):
| 요소 | 내용 | 차원 |
|---|---|---|
| **observation s** | base_lin_vel(3)+base_ang_vel(3)+projected_gravity(3)+velocity_commands(3)+joint_pos(37)+joint_vel(37)+last_action(37) | **123** |
| **action a** | 37개 관절 목표각 offset (×0.5, default 기준) | **37** |
| **reward r** | tracking(+) − 각종 페널티(−) 가중합 | 스칼라 |
| **transition** | PhysX가 4물리스텝(=control dt 0.02s) 굴린 결과 | — |

> Flat은 height_scan이 빠져서 123차원. Rough는 height_scan 격자가 더해져 더 큼. 정책 신경망의 입력 크기가 곧 이 숫자.

---

## 2. PPO를 5단계로 (개념 → 이 config의 값)

### 단계 1. 정책(policy) = 확률적 신경망
- 정책 π(a|s) = 상태를 넣으면 **각 관절 목표각의 평균 μ와 표준편차 σ**를 내는 가우시안.
- rsl_rl 실측: `actor_hidden_dims=[512,256,128]`(rough) / `[256,128,128]`(flat), `activation="elu"`.
- `init_noise_std=1.0`: 초기 σ. 처음엔 크게 흔들며(탐험) 점점 줄어듦. σ가 곧 탐험량.

### 단계 2. Actor-Critic
- **Actor**(정책): 행동을 낸다.
- **Critic**(가치 V(s)): "이 상태가 앞으로 얼마나 좋은가"를 추정 → advantage 계산에 씀.
- 실측: critic도 별도 MLP `[512,256,128]`. `value_loss_coef=1.0`, `use_clipped_value_loss=True`.

### 단계 3. Advantage & GAE
- **Advantage A = (실제로 받은 리턴) − (critic이 예측한 V)**. "예상보다 좋았나?"
- **GAE(λ)** 로 이 A를 부드럽게 추정: `gamma=0.99`(미래 할인), `lam=0.95`(bias-variance 절충).
  - gamma 0.99 → 약 100스텝(=2초) 앞까지 내다봄.

### 단계 4. PPO clip — "한 번에 너무 크게 바꾸지 마"
- 새 정책/옛 정책 확률비 `ratio = π_new/π_old`.
- 목적함수를 `ratio`가 `1±clip_param` 밖으로 나가면 잘라(clip) 과도한 업데이트를 막음.
- 실측: `clip_param=0.2` (표준값). → 정책이 업데이트당 ±20% 넘게 안 바뀜 = **안정성의 핵심**.
- `entropy_coef=0.008`: 엔트로피 보너스 → 너무 빨리 한 행동으로 굳지 않게 탐험 유지.

### 단계 5. Adaptive learning rate (rsl_rl 특징)
- 실측: `learning_rate=1e-3`, `schedule="adaptive"`, `desired_kl=0.01`.
- 매 업데이트 후 옛/새 정책의 **KL divergence**를 측정해:
  - KL이 목표(0.01)보다 크면(너무 많이 바뀜) → lr **자동 감소**
  - 작으면 → lr **자동 증가**
- **의미**: lr을 손으로 안 맞춰도 rsl_rl이 알아서 안정 유지. 초보에게 rsl_rl이 편한 이유 중 하나.
- `max_grad_norm=1.0`: 그래디언트 클리핑(발산 방지).

---

## 3. On-policy 학습 루프 = "iteration" 한 번의 정체

rsl_rl 실측: `num_steps_per_env=24`, `num_learning_epochs=5`, `num_mini_batches=4`.

한 **iteration**이 하는 일:
```
1) 수집(rollout): 현재 정책으로 모든 env를 24스텝 굴려 데이터 모음
     데이터 양 = num_envs × 24
     (기본 4096 env → 98,304 transitions / 내 실험 512 env → 12,288)
2) 학습(update): 그 데이터를 4개 미니배치로 나눠, 5 epoch 반복하며 PPO 업데이트
     → iteration당 그래디언트 스텝 = 4 × 5 = 20회
3) 수집한 데이터는 버림(on-policy!) → 다음 iteration은 새 정책으로 다시 수집
```
- **on-policy = 현재 정책 데이터만 사용** → 그래서 대량 병렬 env로 매번 새로 모아야 함. GPU 시뮬(수천 env)이 필수인 이유.
- `max_iterations`: flat 1500 / rough 3000 (실전 학습량).

---

## 4. 실제 학습을 돌려봤다 (이 환경, 50-iter 스모크 테스트)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 512 --max_iterations 50
```
결과: **50 iteration을 약 15초에 완주**(RTX 5090, ~20,000 steps/s).

| iteration | Mean reward | Mean episode length |
|---|---|---|
| 0 | **−0.38** | 11.25 |
| 5 | −5.85 | 36.74 |
| 10 | −5.91 | 39.25 |
| 20 | −5.80 | 38.64 |
| 35 | −5.76 | 41.49 |
| 49 | **−5.64** | **42.45** |

### ⚠️ 함정: "reward가 −0.38 → −5.64로 나빠졌는데 학습이 된 거라고?"
**된 거 맞다.** 초보가 반드시 넘어야 할 착시:
- iter 0엔 로봇이 **즉시 넘어져** 에피소드가 11스텝밖에 안 됨 → 페널티 쌓일 시간이 없어 총합 reward가 0에 가까움(−0.38).
- 학습이 진행되며 **버티는 시간이 42스텝으로 늘어남** → 매 스텝 걷히는 자잘한 페널티(action_rate, dof_acc 등)가 더 오래 누적 → 총 reward는 더 음수로.
- 즉 이 구간의 진짜 신호는 **reward 절대값이 아니라 `episode length`(11→42)와 termination(넘어짐) 감소**.
- 50 iter은 정상 학습량(1500)의 3%에 불과한 **동작 확인용**. 실제로는 수백 iter 뒤 tracking 보상이 페널티를 압도하며 reward가 양수로 올라선다.

**교훈**: 학습 판단은 한 숫자(mean reward)가 아니라 **여러 지표를 함께** 봐야 한다 → tensorboard.

### 보상 항 분해도 로그에 찍힌다 (4일차 예고)
로그 끝의 `Episode_Reward/*`가 항별 기여를 보여줌:
```
track_lin_vel_xy_exp: +0.0067   (아직 작음 — 이걸 키우는 게 학습 목표)
termination_penalty : −0.2000   (여전히 잘 넘어짐)
action_rate_l2      : −0.0150
Metrics/base_velocity/error_vel_xy: 0.094   (속도 추종 오차)
Episode_Termination/base_contact : 1.0      (아직 100% 넘어짐)
```
→ 4일차에서 이 항들의 weight를 조정하는 게 곧 "reward 설계".

---

## 5. 학습 상태 보는 법 (반드시 익힐 것)

```bash
# 텐서보드로 reward/episode length/loss/lr 곡선 보기
cd /isaac-sim/IsaacLab
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/g1_flat
```
꼭 볼 곡선:
- **Train/mean_episode_length** ↑ (초반 핵심 신호)
- **Episode_Reward/track_lin_vel_xy_exp** ↑ (목표에 다가가는지)
- **Episode_Termination/base_contact** ↓ (안 넘어지게 되는지)
- **Loss/learning_rate**: adaptive schedule이 어떻게 움직이는지
- **Loss/value_function**, **Loss/surrogate**(PPO)

체크포인트 저장 위치: `logs/rsl_rl/g1_flat/<날짜>/model_*.pt` (`save_interval=50`).

---

## 6. Standing vs Walking (3일차→4일차 다리)
- **Standing(균형)**: velocity command를 0으로 주면 "제자리 유지"에 가까움. tracking 보상이 "0 속도 유지"를 요구.
- **Walking(보행)**: command range를 실측처럼 `lin_vel_x (0,1.0)`, `lin_vel_y (−0.5,0.5)`, `ang_vel_z (−1,1)`로 주면 다양한 속도 추종을 학습.
- 둘은 **같은 태스크, 다른 command 분포**일 뿐. 4일차는 여기에 reward 항 설계를 더한다.

---

## 7. 3일차 자기점검

1. Flat G1의 observation은 왜 123차원인가? 각 항을 나열할 수 있는가?
2. 정책 π(a|s)가 내는 것은 무엇인가? `init_noise_std`는 무엇을 통제하나?
3. Advantage와 GAE(gamma/lam)를 한 줄로 설명할 수 있는가?
4. `clip_param=0.2`가 막는 것은? PPO가 "안정적"인 이유는?
5. rsl_rl의 `adaptive` lr + `desired_kl`은 어떻게 동작하나? 왜 편한가?
6. on-policy라서 수천 개 병렬 env가 필요한 이유는?
7. 1 iteration = ? (수집 num_envs×24 → 4미니배치×5epoch=20 그래디언트 스텝)
8. **내 실험에서 reward가 −0.38→−5.64로 내려갔는데 왜 학습이 잘 되고 있는 것인가?**

---

## 다음
- 4일차: 로그에서 본 `Episode_Reward/*` 항들의 **weight를 설계·튜닝**해 걸음걸이를 바꾸기 (강좌 하이라이트)
