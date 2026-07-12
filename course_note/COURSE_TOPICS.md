# 강좌 기술스택 심화 — 5대 주제

> PinkLab 강좌 소개의 "강화학습(PPO·Reward Shaping·Hyperparameter Tuning)"과 "핵심 기법(Observation/Action 설계·Sim-to-Real)"을
> 개념 + Isaac Lab 어디에 + 실전 원칙으로 묶은 종합 노트. 이미 깊게 다룬 것은 링크로 연결.

---

## 1. PPO (강화학습 알고리즘)

**한 줄**: 정책(뇌)을 **한 번에 조금씩만**(clip) 안전하게 개선하는 on-policy actor-critic 알고리즘.
- 개념·비유: [[DAY0_foundations]] (주사위·코치·나침반), 수식·Critic 학습: [[DAY0_foundations]] 부록 B
- 상세·하이퍼파라미터: [[DAY3_ppo_rsl_rl]], [[PPO_TUNING]]
- 핵심만: 정책은 확률분포(μ,σ) 출력 → 탐험 → advantage(GAE)로 방향 → **clip으로 ±20% 컷** → critic이 기준선 제공 → on-policy라 매 iter 신선한 데이터(수천 env 병렬).

---

## 2. Reward Shaping (보상 설계)

**한 줄**: "무엇을 하면 +/−"를 항(term)의 가중합으로 조각해 **원하는 행동을 유도**하는 기술.
- 상세·실험(얼어붙은 로봇): [[DAY4_reward_shaping]], weight 위치: DAY4 부록 A

### 설계 원칙 (강좌에서 반복될 핵심)
- **세 부류 균형**: ① Task(+ 목표달성) ② Safety(− 넘어짐) ③ Style(− 부드러움·효율). 셋의 균형이 전부.
- **Dense vs Sparse**: "걸으면 +1, 넘어지면 −1"(sparse)은 신호가 드물어 학습이 안 됨 → 보행은 **매 스텝 촘촘한(dense)** 보상(속도추종 exp 커널 등)이 필수.
- **Exp 커널**: `exp(−error²/std²)` — 목표 근처에서 완만해 미세추종 안정(선형 벌점보다 나음).
- **정규화와 상대성**: advantage가 학습 전 정규화되므로([[DAY0_foundations]] B-3) **weight의 절대크기보다 항들 사이 비율**이 중요.
- **황금률**: 한 번에 한 항만, 조금씩. TensorBoard `Episode_Reward/<항>`으로 항별 기여를 보며 진단([[TENSORBOARD]]).
- **주의**: reward hacking — 로봇이 의도와 다른 편법으로 점수를 딴다(예: 과한 style 페널티 → 안 움직여 벌 회피 = "얼어붙은 로봇").

---

## 3. Hyperparameter Tuning

**한 줄**: 학습이 되게/빠르게/안정되게 **손잡이들을 조절**하는 것. 세 층위가 있다.
- 상세·증상→손잡이 표·실험: [[PPO_TUNING]]

### 무엇을 튜닝하나 (3층)
| 층 | 예 | 비중 |
|---|---|---|
| **Reward** (env) | 각 항 weight | ⭐ 보행에서 **가장 큼(90%)** |
| **PPO** (agent) | entropy_coef, desired_kl, 배치, 망크기, epochs | 대부분 기본값, 몇 개만 |
| **물리/액추에이터** (env.sim) | PD게인, decimation, solver | 로봇이 폭발/불안정할 때 |

### 실전 방법론
- **먼저 의심할 순서**: reward·관절이름·게인 → 그 다음 PPO.
- **자주 만지는 PPO**: `max_iterations`(더 오래), `num_envs`(배치·안정), `entropy_coef`(탐험), `desired_kl`(속도).
- **거의 고정**: clip_param, gamma, lam, max_grad_norm, value_loss_coef.
- **방법**: 한 번에 하나 바꿔 재학습 → TensorBoard로 비교. 우리가 돌린 실험 = 이 방법의 실례([[PPO_TUNING]] 부록).

---

## 4. Observation / Action 설계 (핵심 기법) ★새 깊이

정책의 **입력(무엇을 보여줄까)** 과 **출력(무엇을 시킬까)** 을 정하는 것. 잘못 설계하면 아무리 학습해도 안 된다.

### 4-A. Observation 설계 — "정책에게 뭘 보여줄까"
G1 flat의 실제 구성(123차원)을 예로 원칙을 본다:
| 넣는 것 | G1의 예 | 왜 |
|---|---|---|
| **자기 상태(proprioceptive)** | base 선/각속도, joint pos/vel | 몸을 제어하려면 필수 |
| **자세** | `projected_gravity`(중력벡터) | IMU 없이 "얼마나 기울었나"=균형 핵심 |
| **목표(command)** | velocity_commands | "무엇을 하라"를 알아야 추종 |
| **직전 행동** | last_action | 부드러운 연속 동작·진동 억제에 도움 |
| **환경 인지(exteroceptive)** | (rough) height_scan 187 | 지형을 봐야 계단 대응([[DAY5_sensors]]) |

**설계 원칙**
- **관측 가능성**: 실물에서 **측정 가능한 것만** 넣어라. 시뮬에서만 아는 값(진짜 마찰·질량)을 actor에 넣으면 배포 불가.
- **정규화**: 항목마다 스케일이 다르면(속도 m/s vs 각도 rad) 학습이 흔들림 → `actor_obs_normalization`(현재 False) 또는 항별 스케일.
- **노이즈 주입**: 관찰에 일부러 `Unoise`를 더함 → 실물 센서 오차에 강건(sim-to-real 복선).
- **History(과거 프레임)**: 순간값만으론 속도·추세를 모를 때 여러 프레임을 쌓기도(관성·지연 대응). G1은 last_action으로 최소한만.
- **⭐ Asymmetric Actor-Critic(고급)**: **critic에게만** privileged 정보(진짜 지형·마찰)를 주고 actor는 배포 가능한 관찰만 → critic이 더 정확한 기준선을 배워 학습↑, 배포는 그대로. Isaac Lab은 별도 **critic 관찰 그룹**으로 지원(이 태스크는 actor=critic 동일 그룹).

### 4-B. Action 설계 — "정책에게 뭘 시킬까"
G1: `JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)` — **모든 관절의 목표각 offset**.

**행동 공간 선택지**
| 방식 | 의미 | 특징 |
|---|---|---|
| **관절 위치(기본)** | 목표각 → 온보드 PD가 토크로 | 안정·부드러움. 보행 표준 |
| 관절 토크 | 토크 직접 | 유연하나 학습 어렵고 불안정 |
| 관절 속도 | 목표 속도 | 중간 |

**설계 원칙**
- **왜 위치인가**: PD 컨트롤러가 "저역통과 필터" 역할 → 정책이 대충 목표만 줘도 부드럽게 감. 토크 직접 제어는 매 스텝 완벽해야 해 학습 난이도↑.
- **default_offset**: action=0이면 **기본자세 유지**. 정책은 "기본자세에서 얼마나 벗어날까"만 배우면 됨 → 학습 쉬움([[DAY1_isaac_sim_basics]]).
- **action scale(0.5)**: 정책 출력의 물리적 크기 조절. Kp와 함께 "얼마나 세게 움직일 수 있나" 결정([[DAY2_physics_actuators]]).
- **주파수(50Hz)**: control dt와 일치. 실물 제어 주파수와 맞춰야 sim-to-real.
- **대칭성/스무딩**: action_rate 페널티로 급변 억제(=출력 필터), 좌우 대칭 유도 등.

---

## 5. Sim-to-Real (핵심 기법) ★종합

**한 줄**: 시뮬에서 배운 정책이 **실제 로봇에서도** 동작하게 하는 것. 둘 사이 **reality gap**을 메우는 기술의 총합.
- 각 조각의 출처: [[DAY2_physics_actuators]](랜덤화), [[DAY5_sensors]](센서), [[DAY6_ros2_sim2real]](export·ROS2)

### Reality Gap의 원천과 대비책
| 격차 | 무엇이 다른가 | 대비 기법 |
|---|---|---|
| 물리 | 마찰·질량·CoM이 실물과 다름 | **Domain Randomization**(학습 중 무작위화) |
| 액추에이터 | 토크한계·모터지연·백래시 | effort_limit·PD를 실물에 맞춤, **actuator network** |
| 센서 | 시뮬 height-scan은 이상적, 실물은 추정·노이즈 | 관찰 **noise/clip**, **teacher→student** |
| 지연(latency) | 관찰→행동 사이 실물은 지연 큼 | 지연 랜덤화, action 필터 |
| 관찰 분포 | 스케일·순서 불일치 | export에 **normalizer 포함**, 관찰 파이프라인 검증 |

### 대표 기법 상세
- **Domain Randomization**: 물리 파라미터를 매 에피소드 무작위 → 정책이 "평균 로봇"이 아니라 "다양한 로봇 모두"에 강건해짐. 이 태스크의 `EventCfg`가 마찰·질량·밀기 랜덤화.
- **Observation Noise**: 관찰에 노이즈를 학습 때부터 섞어 실물 센서 오차에 대비.
- **Teacher–Student (privileged learning)**: ① teacher를 privileged 정보(진짜 지형)로 학습(쉬움) → ② student가 배포 가능한 관찰(카메라·proprio)만으로 teacher를 모방 → 어려운 vision 정책을 안정적으로 얻음. 거친 지형·실물 배포의 최신 정석.
- **Policy Export**: privileged critic·PPO 다 버리고 actor만 ONNX/TorchScript로 → 온보드 실시간 추론([[DAY6_ros2_sim2real]]).
- **ROS2 Bridge**: 학습 정책을 `/cmd_vel` 등 실 로봇 인터페이스에 연결.

### 핵심 메시지
> **sim-to-real은 마지막 단계가 아니라 1일차부터 시작된다.** Observation을 "측정 가능한 것만"으로 짜고(4-A), 물리를 랜덤화하고(DR), 관찰에 노이즈를 넣는 그 순간부터 이미 sim-to-real 설계다.

---

## 6. 다섯 주제의 연결 지도
```
Observation/Action 설계  →  무엇을 보고 무엇을 시킬까 (문제 정의)
        ↓
   Reward Shaping        →  무엇이 좋은 행동인가 (목표 정의)
        ↓
       PPO               →  그 목표를 어떻게 학습하나 (알고리즘)
        ↓
 Hyperparameter Tuning   →  학습이 되게 손잡이 조절 (최적화)
        ↓
    Sim-to-Real          →  실물로 넘기기 (배포)
```
> 위에서 아래로 잘못되면 아래가 다 무너진다. **관찰·행동 설계가 틀리면 reward·PPO를 아무리 손봐도 안 됨.**
