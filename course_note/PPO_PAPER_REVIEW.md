# PPO 논문 리뷰 × rsl_rl 소스코드 대조

> 목적: PPO를 **논문 원리(수식)** 수준에서 이해하고, 그 각 개념이 **rsl_rl `algorithms/ppo.py` 실제 라인**과 **G1/Go2 config 값**에 어떻게 나타나는지 1:1로 박제한다.
> 그래야 튜닝할 때 "이 값이 논문의 무엇을 건드리는지" 알고 만진다.
> 선행: [[DAY3_ppo_rsl_rl]](개념), [[PPO_TUNING]](값↔증상), [[TENSORBOARD]](모니터링). 심화 후속: 이 노트가 그 둘의 "이론 바닥".
> 실측 근거: `kit/python/.../rsl_rl/algorithms/ppo.py`, `rsl_rl/storage/rollout_storage.py`, `config/g1/agents/rsl_rl_ppo_cfg.py`.

---

## 0. 논문 계보 — PPO는 혼자 있는 알고리즘이 아니다

rsl_rl의 PPO 한 덩어리는 사실 **네 갈래의 논문**이 합쳐진 것이다. 하나만 읽으면 코드의 절반이 "왜 있는지" 안 보인다.

| # | 논문 | 이 노트에서 주는 것 | rsl_rl 위치 |
|---|---|---|---|
| ① | **TRPO** — Schulman et al. 2015, [1502.05477](https://arxiv.org/abs/1502.05477) | *왜* 정책을 조금씩만 바꿔야 하나 (trust region의 논리) | clip이 근사하는 대상 |
| ② | **PPO** — Schulman et al. 2017, [1707.06347](https://arxiv.org/abs/1707.06347) | clipped surrogate objective (핵심) | `ppo.py:296-302` |
| ③ | **GAE** — Schulman et al. 2015, [1506.02438](https://arxiv.org/abs/1506.02438) | advantage 추정, `gamma`·`lam`의 진짜 의미 | `rollout_storage.py:127-149` |
| ④ | **"The 37 Implementation Details of PPO"** — Huang et al. 2022 (ICLR blog) | *논문엔 없는데 코드엔 있는* 것들 (value clip, adv 정규화, grad clip, timeout bootstrap) | `ppo.py` 곳곳 |

> **가장 중요한 메타-교훈**: 아래 5절에서 보듯, rsl_rl은 원논문 PPO와도 **한 군데에서 결정적으로 다르다**(KL로 clip이 아니라 *learning rate*를 조절). 논문만 요약하면 이걸 놓치고 엉뚱한 손잡이를 돌린다.

---

## 1. 한 사이클 큰 그림 (코드 흐름으로)

on-policy actor-critic PPO의 1 iteration = **수집 → advantage 계산 → 여러 번 업데이트 → 버림**.

```
[수집 num_steps_per_env=24 스텝 × num_envs=4096]
  act():        a~π_old(·|s), V(s), logπ_old(a|s) 저장         ppo.py:129
  env.step():   r, done 저장, timeout이면 V로 부트스트랩       ppo.py:142-167
[compute_returns()]  GAE로 advantage Â_t, return R_t 계산       ppo.py:171-176
[update()]  같은 데이터로 epochs=5 × mini_batches=4 = 20번:     ppo.py:178
  - ratio = exp(logπ_new − logπ_old)                            ppo.py:297
  - clipped surrogate loss                                       ppo.py:298-302
  - clipped value loss                                           ppo.py:305-311
  - KL 재서 learning_rate 자동 조절 (rsl_rl 고유!)               ppo.py:260-294
  - loss.backward(), grad clip, optimizer.step()                 ppo.py:367-381
[storage.clear()]  데이터 폐기 → 다음 iteration 새로 수집        ppo.py:409
```

**on-policy의 핵심 제약**: 데이터는 *현재 정책 근처*에서만 유효하다. 그래서 같은 배치로 20번 업데이트하는 동안 정책이 너무 멀어지면 데이터가 거짓말이 된다 → 이걸 막는 게 PPO의 전부다.

---

## 2. ① TRPO — "왜 조금씩만 바꾸나"의 원류

### 정책 경사(policy gradient)의 근본 문제
정책을 좋게 만들려면 이런 **surrogate(대리) 목적함수**를 올리고 싶다:

$$L(\theta) = \mathbb{E}_t\!\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}\,\hat{A}_t\right] = \mathbb{E}_t\big[r_t(\theta)\,\hat{A}_t\big]$$

- `ratio` $r_t(\theta) = \pi_\theta/\pi_{\theta_{old}}$ = "새 정책이 이 행동을 옛 정책보다 몇 배 더/덜 하려는가"(**importance sampling** 보정).
- $\hat{A}_t$(advantage) > 0 이면 그 행동 확률을 올리고(→ratio↑), < 0 이면 내린다.

**문제**: 이 근사는 $\pi_\theta$가 $\pi_{\theta_{old}}$ **근처일 때만** 맞다. ratio를 무한정 키우면 목적함수는 커 보이지만 실제 정책은 망가진다(데이터가 옛 정책 것이라).

### TRPO의 답: 신뢰 구간(trust region)
"목적함수를 올리되, 새 정책이 옛 정책에서 **KL divergence로 δ 이상 벗어나지 마라**"는 **하드 제약**을 건다:

$$\max_\theta L(\theta)\quad \text{s.t.}\quad \mathbb{E}_t\big[\mathrm{KL}(\pi_{\theta_{old}}\,\|\,\pi_\theta)\big] \le \delta$$

- 효과는 확실하지만 **2차 미분(Fisher 행렬)·conjugate gradient**가 필요해 무겁고 구현이 복잡하다.
- → PPO는 "이 trust region 효과를 **1차 미분만으로, clip 한 줄로** 근사하자"가 동기.

> **KL이 왜 중요한지 여기서 결정됨**: KL은 "정책이 한 번에 얼마나 변했나"의 자다. TRPO는 KL을 제약으로, PPO는 clip으로 간접 통제, **rsl_rl은 KL을 measured 값으로 삼아 learning rate를 조절**(5절). 세 방식 모두 뿌리는 이 trust region 아이디어다.

---

## 3. ② PPO — clip으로 trust region을 흉내내기

원논문의 두 변형 중 rsl_rl은 **clipped 변형**을 쓴다(KL-penalty 변형 아님).

### Clipped Surrogate Objective (논문 식 7)

$$L^{CLIP}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta),\,1-\epsilon,\,1+\epsilon)\,\hat{A}_t\big)\Big]$$

- $\epsilon$ = `clip_param` = **0.2** → ratio를 [0.8, 1.2]로 자른다.
- `min`의 역할: **개선은 막지 않되, 과한 개선의 유혹만 제거**한다.
  - $\hat{A}>0$(좋은 행동): ratio가 1.2를 넘어도 목적함수는 1.2에서 평평 → "더 밀어도 보상 없음" → 안 민다.
  - $\hat{A}<0$(나쁜 행동): ratio가 0.8 밑으로 가도 평평 → 과하게 억누르지 않음.
  - **비대칭 트릭**: `min`이라서 "정책이 이미 너무 멀어진 방향"은 clip으로 자르지만, "되돌아오는 방향"은 안 자른다.

### rsl_rl 실제 코드 (`ppo.py:296-302`)

```python
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))   # r_t
surrogate = -torch.squeeze(advantages_batch) * ratio                                     # -r·Â
surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1-clip_param, 1+clip_param)   # -clip(r)·Â
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()                          # max(부호반전) = min
```

- log-space에서 `exp(logπ_new − logπ_old)`로 ratio 계산 → 수치 안정(곱셈 대신 뺄셈).
- 부호가 반대인 이유: 논문은 목적함수를 **최대화**, 코드는 loss를 **최소화** → `min`이 `max(-·)`로 뒤집힘. 결과는 동일.
- **`clip_param`은 사실상 고정값(0.2)**이다. 이건 "허용 변화폭"이라 reward나 로봇이 바뀌어도 잘 안 건드린다. → [[PPO_TUNING]] "딱딱한 벽" 참고.

---

## 4. ③ GAE — advantage를 어떻게 추정하나 (`gamma`·`lam`의 정체)

surrogate에 들어가는 $\hat{A}_t$를 어떻게 구하느냐가 학습 안정성의 절반이다. rsl_rl은 **GAE(Generalized Advantage Estimation)**를 쓴다.

### TD 오차와 GAE 식
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \qquad(\text{1-스텝 advantage})$$
$$\hat{A}^{GAE}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l} \;=\; \delta_t + \gamma\lambda\,\hat{A}^{GAE}_{t+1}$$

- **`gamma` (γ=0.99) = 미래 할인**. 얼마나 먼 미래 보상까지 신경 쓰나. 0.99 → 유효 지평 ~100스텝(=2초, control dt 0.02s). 보행처럼 즉각적 태스크에 적당.
- **`lam` (λ=0.95) = bias↔variance 손잡이**.
  - λ→0: $\hat{A}_t≈\delta_t$ (1스텝, **저분산·고편향** — V를 많이 믿음).
  - λ→1: Monte-Carlo return에 가까움 (**고분산·저편향** — 실제 보상열을 많이 믿음).
  - 0.95 = 살짝 V 쪽으로 기운 균형. critic이 부정확한 초반에 폭주를 막아줌.

### rsl_rl 실제 코드 (`rollout_storage.py:127-149`) — 위 식을 뒤에서 앞으로 한 줄씩

```python
for step in reversed(range(num_transitions_per_env)):        # 뒤→앞 (재귀식이라)
    next_values = last_values if step==last else values[step+1]
    next_is_not_terminal = 1.0 - dones[step].float()         # 종료 스텝이면 미래 끊음
    delta = rewards[step] + next_is_not_terminal*gamma*next_values - values[step]   # δ_t
    advantage = delta + next_is_not_terminal*gamma*lam*advantage                    # Â_t = δ + γλÂ_{t+1}
    returns[step] = advantage + values[step]                 # R_t = Â_t + V(s_t)  (critic 학습 타깃)
advantages = returns - values                                # 최종 Â
advantages = (advantages - mean) / (std + 1e-8)              # ★ 정규화 (논문엔 없음, 5-B절)
```

- `next_is_not_terminal`: episode가 끝난 스텝에서는 $V(s_{t+1})$을 0으로 끊어 "죽은 뒤 미래"를 새지 않게 한다.
- **`returns`(R_t)가 곧 critic의 회귀 타깃**이다 → value loss(5-B)가 이걸 맞추도록 학습.

---

## 5. ★ rsl_rl의 결정적 괴리 — KL로 clip이 아니라 *learning rate*를 조절

여기가 **이 노트에서 가장 중요한 부분**이다. 논문만 읽으면 100% 놓친다.

### 원논문 PPO의 adaptive 변형
PPO 논문에는 clip 대신 쓰는 **KL-penalty 변형**이 있는데, 거기서 adaptive는 이렇게 동작한다:
- 목적함수 $= L^{surrogate} - \beta\,\mathrm{KL}$
- KL이 목표보다 크면 **penalty 계수 β를 키우고**, 작으면 줄인다. (learning rate는 안 건드림)

### rsl_rl이 실제 하는 것 (`ppo.py:260-294`)
rsl_rl은 **clip 변형을 쓰면서**, KL을 재서 **learning rate 자체**를 조절한다:

```python
if desired_kl is not None and schedule == "adaptive":
    kl = 두 가우시안의 해석적 KL (아래)
    kl_mean = kl.mean()
    if   kl_mean > desired_kl * 2.0:              # 너무 많이 변함
        learning_rate = max(1e-5, lr / 1.5)      #   → 보폭 줄임
    elif kl_mean < desired_kl / 2.0 and kl_mean>0:  # 너무 안 변함
        learning_rate = min(1e-2, lr * 1.5)      #   → 보폭 키움
    for g in optimizer.param_groups: g["lr"] = learning_rate
```

**튜닝에 직결되는 함의**:
1. **`learning_rate=1e-3`은 *시작값*일 뿐**이다. 매 미니배치 KL을 보고 [1e-5, 1e-2] 범위에서 1.5배씩 자동으로 오르내린다.
2. **진짜 손잡이는 `desired_kl`(=0.01)**이다. 이게 "한 번에 이 정도만 변해라"의 목표치.
   - `desired_kl`↑ → lr이 잘 안 깎여서 공격적 학습(빠르지만 불안정 위험).
   - `desired_kl`↓ → lr이 자주 깎여서 보수적 학습(안정적이지만 느림).
3. TensorBoard `Loss/learning_rate`가 **출렁이는 게 정상**이다. Go2 kl 실험(logs/go2_exp_kl_*)에서 kl=0.002→lr 1e-5 바닥, kl=0.05→최대 5e-3까지 오른 게 이 코드 때문. → [[go2/GO2_EXPERIMENTS]].

### 왜 clip이 있는데 KL도 재나? (이중 안전장치)
- **clip**(3절) = 매 샘플 ratio의 하드 상한 → 개별 스텝 폭주 차단.
- **KL→lr**(이 절) = 배치 전체 평균 변화량 피드백 → 전역 보폭 조절.
- 둘은 층위가 다르다. clip은 "한 걸음의 최대 폭", lr은 "그 걸음을 얼마나 세게". 그래서 공존한다.

### 가우시안 KL을 해석적으로 계산 (`ppo.py:262-268`)
정책이 대각 가우시안이라 KL을 샘플링 없이 **닫힌 형태**로 잰다:

$$\mathrm{KL}(\pi_{old}\|\pi_{new}) = \sum_i\left[\log\frac{\sigma_{new,i}}{\sigma_{old,i}} + \frac{\sigma_{old,i}^2 + (\mu_{old,i}-\mu_{new,i})^2}{2\sigma_{new,i}^2} - \frac12\right]$$

코드가 이 식 그대로다(`log(sigma/old_sigma) + (old_sigma² + (old_mu-mu)²)/(2·sigma²) - 0.5`). 관절별로 합산 → 평균. 저분산·정확해서 lr 스케줄러의 신뢰할 신호가 된다.

---

## 6. ④ 논문엔 없고 코드엔 있는 디테일들 ("37 details")

원 PPO 논문 의사코드엔 없지만 **모든 실전 구현에 있는** 것들. 이걸 알아야 config를 오해 없이 읽는다.

### 6-A. Clipped Value Loss (`ppo.py:305-311`, `use_clipped_value_loss=True`)
critic도 한 번에 너무 안 변하게 clip한다:
```python
value_clipped = target_values + (value - target_values).clamp(-clip_param, +clip_param)
value_loss = max( (value-returns)², (value_clipped-returns)² ).mean()
```
- actor의 clip과 **같은 `clip_param=0.2`를 재사용**. critic 값이 한 업데이트에서 ±0.2 넘게 튀는 걸 억제.
- `value_loss_coef=1.0`으로 전체 loss에 합쳐짐.

### 6-B. Advantage 정규화 (`rollout_storage.py:148`, `ppo.py:221-223`)
- advantage를 평균 0·표준편차 1로 정규화 → **스케일 불변**. reward 크기가 바뀌어도 gradient 크기가 안정.
- rsl_rl은 두 방식: 배치 전체(기본) 또는 미니배치별(`normalize_advantage_per_mini_batch`). 이래서 [[DAY4_reward_shaping]]에서 reward weight를 바꿔도 학습이 덜 민감했던 것.

### 6-C. Gradient Clipping (`ppo.py:380`, `max_grad_norm=1.0`)
```python
nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)   # grad 전체 norm을 1.0으로 상한
```
- gradient가 폭발해 파라미터가 한 번에 날아가는 걸 막는 마지막 안전벨트. optimizer.step() 직전.

### 6-D. Entropy Bonus (`ppo.py:315`, `entropy_coef`)
```python
loss = surrogate_loss + value_loss_coef*value_loss - entropy_coef*entropy.mean()
```
- 정책 분포의 엔트로피(=탐험량)를 **키우는 쪽**으로 보상(부호 −). 조기 수렴/결정론화 방지.
- G1 `entropy_coef=0.008`(rough), Go2도 유사. **작게 올리면 탐험↑**. [[PPO_TUNING]]·Go2 실험에서 ent=0.05로 키우니 eplen 13으로 붕괴(과탐험) → 이 항의 위력 확인. → [[go2/GO2_EXPERIMENTS]].
- `init_noise_std=1.0`(policy의 초기 σ)과 함께 탐험을 결정. entropy_coef는 "탐험을 유지하는 힘", noise_std는 "출발 탐험량".

### 6-E. Time-out Bootstrapping (`ppo.py:160-164`)
```python
if "time_outs" in extras:
    rewards += gamma * values * time_outs     # 시간초과 종료는 "진짜 실패"가 아님
```
- **중요한 구분**: episode가 넘어져서 끝난 것(termination)과 시간이 다 돼서 끊긴 것(time-out)은 다르다.
- time-out은 "거기서 미래가 없다"가 아니라 "단지 우리가 관찰을 멈췄을 뿐" → $V(s)$로 미래를 **부트스트랩**해 되살린다. 안 하면 "오래 살면 벌받는다"는 잘못된 신호를 학습. 무한지평 태스크(보행)에 필수.

---

## 7. 전체 손실 함수 조립 (`ppo.py:315`)

지금까지의 조각이 한 줄로 합쳐진다:

$$L = \underbrace{L^{CLIP}}_{\text{3절, actor}} + \underbrace{c_v\,L^{VF}}_{\text{6-A, critic}} - \underbrace{c_e\,S[\pi]}_{\text{6-D, 탐험}}$$

```python
loss = surrogate_loss  +  value_loss_coef * value_loss  -  entropy_coef * entropy.mean()
#      ────────────────    ──────────────────────────     ─────────────────────────────
#      정책 개선(clip)       가치함수 회귀(c_v=1.0)          탐험 유지(c_e=0.008)
```
그 뒤: `backward()` → `clip_grad_norm_`(6-C) → `optimizer.step()`(Adam). 이걸 epochs×mini_batches = **5×4 = 20번** 반복하고 데이터를 버린다.

---

## 8. 튜닝으로 되돌아오기 — 각 파라미터의 "논문 출신 성분표"

이 노트의 최종 산출물. **어떤 손잡이가 논문의 무엇을 건드리는지**, 그래서 **얼마나 만져도 되는지**.

| config 파라미터 | 값(G1) | 논문 출신 | 실제로 만지나? | 만지면 생기는 일 |
|---|---|---|---|---|
| `clip_param` | 0.2 | ② PPO 핵심 | ✗ 거의 고정 | 허용 변화폭. 키우면 불안정, 줄이면 학습 느림 |
| `desired_kl` | 0.01 | ①TRPO+rsl_rl | **○ 진짜 lr 손잡이** | ↑공격적/불안정, ↓보수적/느림 (5절) |
| `learning_rate` | 1e-3 | 공통 | △ 시작값일 뿐 | adaptive가 곧 덮어씀. 초기 수 iter만 영향 |
| `entropy_coef` | 0.008 | ④ | **○ 탐험 손잡이** | ↑탐험(과하면 붕괴), ↓조기수렴 (6-D) |
| `gamma` | 0.99 | ③ GAE | △ 태스크 지평 | ↓근시안, ↑장기신용(분산↑) |
| `lam` | 0.95 | ③ GAE | △ bias/variance | ↓V의존(편향), ↑MC의존(분산) |
| `num_learning_epochs` | 5 | 공통 | △ 재사용 횟수 | ↑샘플효율(과적합·KL폭주 위험), ↓안정 |
| `num_mini_batches` | 4 | 공통 | △ | 배치크기와 gradient 노이즈 조절 |
| `value_loss_coef` | 1.0 | ④ | ✗ 거의 고정 | critic vs actor 학습 균형 |
| `max_grad_norm` | 1.0 | ④ | ✗ 안전벨트 | 거의 안 건드림 |
| `init_noise_std` | 1.0 | policy | △ 출발 탐험 | ↑초기 탐험량, entropy_coef와 짝 |

> **결론적 튜닝 철학** ([[PPO_TUNING]]와 일치): Isaac Lab 보행에서 PPO 손잡이 중 실제로 의미있게 만지는 건 **`desired_kl`·`entropy_coef` 둘뿐**이고, 나머지는 legged locomotion에 이미 잘 맞춰진 기본값이다. 튜닝의 90%는 여전히 reward([[DAY4_reward_shaping]]). 이 노트의 값어치는 "왜 그런지"를 논문 수준에서 아는 것 — 그래야 정말 필요할 때 자신 있게 건드린다.

---

## 9. 읽는 순서 추천

1. **먼저 이 노트 1절**(큰 그림) → 2·3절(TRPO 동기 → PPO clip) — 여기까지가 "PPO가 뭐냐"의 핵심.
2. **5절**(rsl_rl의 KL→lr 괴리) — 튜닝하려면 반드시. 원논문 안 읽어도 이건 알아야 함.
3. 여유되면 원논문 순서: **GAE(1506.02438) → PPO(1707.06347)**. TRPO(1502.05477)는 동기만 훑어도 충분(수학 무거움).
4. 실전 디테일은 Huang et al. "37 Implementation Details of PPO" (검색: *ICLR blog track 2022 ppo details*) — 6절이 그 요약.

---

### 부록: 논문↔코드 라인 빠른 색인
- clipped surrogate: `ppo.py:296-302` ← PPO 식 7
- KL→learning_rate: `ppo.py:260-294` ← rsl_rl 고유
- 가우시안 해석적 KL: `ppo.py:262-268`
- clipped value loss: `ppo.py:305-311`
- 전체 loss 조립: `ppo.py:315`
- grad clip: `ppo.py:380`
- timeout bootstrap: `ppo.py:160-164`
- GAE 계산: `rollout_storage.py:127-149`
- config 값: `config/g1/agents/rsl_rl_ppo_cfg.py`
