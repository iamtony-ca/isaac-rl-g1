# Go2 vs G1 — 한 장 종합 비교 (델타의 전부)

> "무엇이 재사용되고 무엇이 달라지나"를 한눈에. 세부는 각 델타 노트로.
> 값 출처(실측): `unitree.py`(asset), `config/go2/*`(태스크·에이전트), `inspect_go2.py` 출력.

---

## 0. 한 문장
> **프레임워크·PPO·reward 함수 라이브러리는 그대로. 로봇 형태(4족·DC모터·12DOF)에서 오는 것만 바뀐다.**

---

## 1. 하드웨어/형태

| 항목 | G1 (휴머노이드) | **Go2 (4족)** |
|---|---|---|
| 형태 | 2족 직립 + 팔/손/몸통 | **4족 보행** |
| DOF | 23~37 (다리+팔+손가락+허리) | **12** (4다리 × hip/thigh/calf) |
| base 링크 | `torso_link` | **`base`** |
| 발 링크 | `.*_ankle_roll_link` | **`.*_foot`** |
| 무게/키 | 크고 높음(불안정) | 작고 낮음(**안정**) |
| 균형 난이도 | 높음(2점 지지, 넘어지기 쉬움) | 낮음(**4점 지지, 정적 안정**) |

> 4족은 항상 여러 발이 땅에 닿아 **정적으로 안정**하다 → 초기 학습이 G1보다 쉽고 빠르다.
> 형태 실측은 [[GO2_STRUCTURE_ACTUATOR]].

---

## 2. 액추에이터 모델 (핵심 차이)

| 항목 | G1 | **Go2** |
|---|---|---|
| 모델 클래스 | `ImplicitActuatorCfg` | **`DCMotorCfg`** |
| Kp (stiffness) | 100~200 (관절군별) | **25** (전 관절 동일) |
| Kd (damping) | 2~5 | **0.5** |
| effort_limit | ~300 N·m (다리) | **23.5 N·m** |
| 특수 파라미터 | — | **`saturation_effort`, `velocity_limit`** (속도-토크 곡선) |

> **DC-Motor 모델**은 실제 모터의 "속도가 빠를수록 낼 수 있는 토크가 준다"(속도-토크 특성)를 반영한다.
> Implicit PD보다 **실물에 가까운 액추에이터** → sim-to-real에 유리. 심화는 [[GO2_STRUCTURE_ACTUATOR]].

---

## 3. Reward 항 (형태가 reward를 결정한다)

| reward 항 | G1(biped) | **Go2(quadruped)** | 이유 |
|---|---|---|---|
| `feet_air_time` 함수 | `feet_air_time_positive_biped` | **`feet_air_time`(표준)** | 2족 전용 gait 함수 vs 4족 표준 |
| `feet_slide` | ✅ 있음 | ❌ 없음 | 2족은 미끄러지면 넘어짐 → 규제 필요 |
| `joint_deviation_arms/fingers/torso` | ✅ 있음 | ❌ **없음** | Go2엔 **팔·손·몸통이 없다** |
| `undesired_contacts` | None(rough) | **`.*THIGH` 접촉 페널티**(base) / None(rough) | 4족은 허벅지 끌기 방지 |
| `flat_orientation_l2` | -1.0 | **-2.5(flat)** | 몸통 수평 유지 |

> **팔·손이 없으니 그걸 규제하던 reward 항 4~5개가 통째로 사라진다** = Go2 reward가 더 단순.
> 상세·weight는 [[GO2_REWARD]].

---

## 4. Action / 환경 스케일

| 항목 | G1 | **Go2** |
|---|---|---|
| action | `JointPositionAction scale=0.5` | **`scale=0.25`** (rough) |
| 제어 방식 | 목표각 offset → PD | 동일 (관절 위치) |
| terrain 스케일 | 기본 | **작게 축소** (로봇이 작아서: box 높이 0.025~0.1, noise 0.01~0.06) |
| add_base_mass | — | **-1.0~+3.0 kg** on `base` (domain randomization) |
| push_robot | 있음 | **rough에선 None** (외력 밀기 비활성) |

---

## 5. PPO 하이퍼파라미터 (거의 동일, 망 크기만 다름)

| config | G1 flat | **Go2 flat** | G1 rough | **Go2 rough** |
|---|---|---|---|---|
| `max_iterations` | 1500 | **300** | 3000 | **1500** |
| `actor/critic_hidden_dims` | [256,128,128] | **[128,128,128]** | [512,256,128] | **[512,256,128]** |
| `entropy_coef` | 0.008 | **0.01** | 0.008 | **0.01** |
| `num_steps_per_env` | 24 | 24 | 24 | 24 |
| clip/gamma/lam/desired_kl | 0.2/0.99/0.95/0.01 | 동일 | 동일 | 동일 |

> **Go2 flat은 300 iter·[128³] 소형망으로 충분**하다 = 4족 평지 보행이 그만큼 쉽다는 뜻.
> (G1 flat은 1500 iter 필요.) PPO 개념 자체는 [[DAY3_ppo_rsl_rl]]·[[PPO_TUNING]] 그대로.

---

## 6. 재사용 vs 새로 볼 것 (요약)

| 그대로 재사용 (G1 노트로) | Go2에서 새로 보는 것 (이 폴더) |
|---|---|
| PPO/GAE/clip/entropy 이론 [[DAY0_foundations]][[DAY3_ppo_rsl_rl]] | 12-DOF 4족 구조 [[GO2_STRUCTURE_ACTUATOR]] |
| 튜닝 손잡이 개념 [[PPO_TUNING]] | DC-Motor 액추에이터 |
| TensorBoard 읽는 법 [[TENSORBOARD]] | quadruped reward [[GO2_REWARD]] |
| observation/action 설계 원칙 [[COURSE_TOPICS]] | Go2 실험 수치 [[GO2_EXPERIMENTS]] |
| sim-to-real·export·ROS2 [[DAY6_ros2_sim2real]] | height_scan을 `base`에 부착 |

> **결론**: 강좌를 G1으로 한 번 익혔다면, Go2는 "이 6개 델타 표"만 이해하면 바로 넘어간다.
