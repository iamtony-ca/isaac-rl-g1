# 2일차 심화 — 시뮬레이션 환경 구축 (PhysX & 액추에이터)

> 목표: "물리 엔진 파라미터와 액추에이터(PD 게인)가 **학습에 어떤 영향을 주는가**"를 이 환경의 실제 값으로 이해한다.
> 실측 근거: `LocomotionVelocityRoughEnvCfg.__post_init__`, `SimulationCfg/PhysxCfg`, `G1_MINIMAL_CFG` (Isaac Lab 2.3.2).

---

## 1. 시간 구조 — 모든 것의 뼈대 (실측값)

이 환경의 진짜 값:
```python
self.sim.dt          = 0.005   # 물리 스텝 = 5 ms  → 물리 200 Hz
self.decimation      = 4       # 정책은 물리 4스텝마다 1번 행동
self.episode_length_s = 20.0   # 에피소드 20초
self.sim.render_interval = 4   # decimation과 동일 (렌더는 정책 주기로)
```

여기서 파생되는 핵심 수치:
| 값 | 계산 | 의미 |
|---|---|---|
| **물리 주파수** | 1/0.005 | **200 Hz** — 물리 적분·접촉 해석 빈도 |
| **control dt** | 0.005 × 4 | **0.02 s → 정책 50 Hz** |
| **에피소드 스텝 수** | 20.0 / 0.02 | **1000 RL 스텝** / 에피소드 |
| **1 RL step 동안** | — | 물리가 4번 돌고(같은 action 유지) 그 결과가 다음 observation |

### 왜 이 구조인가 (초보가 꼭 알아야 할 것)
- **물리는 빠르게(200Hz), 정책은 느리게(50Hz).** 물리를 촘촘히 풀어야 접촉/안정성이 확보되지만, 정책이 매 물리스텝 결정하면 학습이 불필요하게 어려워지고 실제 로봇 제어 주파수(보통 50~100Hz)와도 안 맞는다.
- **decimation을 바꾸면 문제 자체가 바뀐다**: decimation↑ → 정책이 더 뜸하게 결정 → 각 action이 더 오래 유지 → 반응성↓ 이지만 학습 안정성↑, 계산량↓. decimation↓ → 정밀하지만 학습 어려움.
- **reward/페널티 scale이 이 dt에 묶여 있다**: `action_rate`, `dof_acc`, `dof_torques` 페널티는 "스텝당" 값이라 dt를 바꾸면 상대적 크기가 달라진다. → dt를 바꾸면 reward 재튜닝 필요.

> 🔑 **control dt = sim.dt × decimation** 이 한 줄이 2일차의 핵심. 3·4일차 내내 등장.

---

## 2. PhysX 솔버 — "얼마나 정확히 푸느냐"

### 2.1 솔버 종류 (실측: `solver_type = 1`)
- **TGS(1, Temporal Gauss-Seidel)** ← 기본. 접촉·관절이 많은 다리 로봇에 안정적. (반대는 PGS(0), 더 단순·빠르지만 뻣뻣한 접촉에 약함)

### 2.2 솔버 반복 횟수 (G1 config 실측)
```python
articulation_props = ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,   # 위치 제약 반복
    solver_velocity_iteration_count=4,   # 속도 제약 반복
)
```
- **iteration↑** → 관절·접촉 제약을 더 정확히 만족(관통·미끄러짐↓) but 느림.
- 보행처럼 접촉 많은 태스크는 position 8 정도가 흔한 타협점.
- **self-collision=False**: 휴머노이드는 팔-몸통 등 자기충돌 계산이 비싸서 보통 끔(학습 속도↑). 켜면 더 현실적이나 무거움.

### 2.3 접촉 파라미터 (PhysxCfg 기본값)
- `bounce_threshold_velocity = 0.5` : 이보다 느린 충돌은 안 튕김(발-지면이 계속 튀는 것 방지).
- `friction_correlation_distance = 0.025` : 인접 접촉점을 묶는 거리(성능/안정 타협).
- `gpu_max_rigid_patch_count = 10 * 2**15` : GPU 접촉 버퍼. **너무 작으면 접촉 많은 씬에서 오버플로 에러** → 이 값을 늘려야 함(4096 env 보행에서 자주 만나는 실전 이슈).

---

## 3. 액추에이터 — 관절을 "어떻게 구동하나" (강좌 튜닝의 심장)

### 3.1 Implicit vs Explicit PD (G1은 Implicit)
G1/H1은 `ImplicitActuatorCfg`를 쓴다.
- **Implicit PD**: Kp/Kd를 **PhysX 솔버에 직접 넘겨** 관절 제약으로 함께 푼다. → 매우 안정적, 높은 Kp도 발산 안 함, 빠름. **입문·보행의 기본.**
- **Explicit PD/actuator network**: 파이썬에서 `torque = Kp·(q* − q) − Kd·q̇` 를 직접 계산해 적용. → 커스텀 모터 모델(마찰, 지연, 데이터 기반 actuator net) 가능하지만 불안정·느릴 수 있음. sim-to-real 고급에서 사용.
- 강좌 예습은 Implicit로 충분. "실제 모터 특성을 더 정밀히 = Explicit"만 기억.

### 3.2 PD 게인의 물리적 의미
관절 목표각 `q*`에 대해 (개념식):
```
torque = Kp · (q* − q)  −  Kd · q̇      (effort_limit로 clip)
        └ 위치 오차를 당김 ┘   └ 속도 감쇠(저항) ┘
```
- **Kp(stiffness)↑** : 목표에 강하게/빠르게 붙음(딱딱). 과하면 진동·불안정.
- **Kd(damping)↑** : 움직임에 저항(끈적, 진동 억제). 과하면 굼뜸.
- **effort_limit** : 낼 수 있는 최대 토크(물리 한계). 넘으면 clip → 실제로 목표에 못 미침.

### 3.3 G1의 그룹별 게인 (실측)
| 그룹 | 관절 수 | Kp | Kd | 왜 이렇게? |
|---|---|---|---|---|
| **legs** | 9 | ≈178 | 5 | 체중 지탱·추진 → 강해야 함(hip/knee 150~200) |
| **feet** | 4 | 20 | 2 | 발목은 **부드럽게** → 지면 적응·충격 흡수 |
| **arms** | 24 | 40 | 10 | 보행 중 팔은 자세 유지만 → 중간 강성, 높은 감쇠로 덜 흔들리게 |

- **직관**: "무게를 받치는 관절은 딱딱하게, 지면과 부딪히는 발목은 물렁하게, 균형용 팔은 감쇠 크게." 이 배분 자체가 설계 지식.

### 3.4 action scale과 게인의 관계 (놓치기 쉬운 연결)
1일차에서 본 action:
```python
JointPositionActionCfg(joint_names=[".*"], scale=0.5, use_default_offset=True)
```
- 정책 출력 `a` → 목표각 `q* = default_q + 0.5·a`.
- 그 `q*`를 위 PD가 토크로 바꿈. → **action scale(0.5)과 Kp가 함께 "정책이 얼마나 세게 움직일 수 있나"를 결정.**
- scale↑ 또는 Kp↑ → 더 격렬한 동작 가능(but 불안정). 튜닝 시 둘을 같이 봐야 함.

---

## 4. Domain Randomization — sim-to-real의 시작 (EventCfg 실측)

이 태스크는 학습 중 물리를 무작위로 흔든다(`EventCfg`):
| 이벤트 | 무엇을 랜덤화 | 효과 |
|---|---|---|
| `randomize_rigid_body_material` | 발-지면 **마찰** | 다양한 바닥에서도 걷게 |
| `randomize_rigid_body_mass` / `add_base_mass` | 링크 **질량** | 짐/제작 오차에 강건 |
| `randomize_rigid_body_com` / `base_com` | 무게중심 위치 | 모델 오차 흡수 |
| `apply_external_force_torque` / `push_robot` | 외부 밀기 | 밀쳐도 안 넘어지게(로버스트) |
| `reset_base` / `reset_robot_joints` | 시작 자세·위치 | 다양한 초기조건 |

- **핵심 개념**: 시뮬에서 일부러 "불확실성"을 학습시켜야 실제 로봇(마찰·질량·지연이 다름)에서도 동작 → **domain randomization**. 5·6일차 sim-to-real의 뿌리.
- 참고: G1 rough 설정은 `push_robot=None`, `add_base_mass=None`으로 일부 랜덤화를 **끔**(로봇마다 튜닝 다름). 즉 "무엇을 켜고 끄느냐"도 튜닝 대상.

---

## 5. 실습 — 파라미터를 바꿔 체감하기

> 코드 수정은 태스크 config에서. 아래는 실험 제안(복사본을 만들어 수정 권장).

1. **decimation 실험**: `velocity_env_cfg.py`의 `self.decimation`을 4→8로 바꾸면 정책이 25Hz가 됨. 학습 곡선/걸음이 어떻게 달라지는지.
2. **게인 실험**: `unitree.py`의 `legs` stiffness를 200→100으로 낮추면? (다리가 물러져 무릎이 꺾일 것 — 넘어짐↑)
3. **GPU 버퍼 실험**: `num_envs`를 크게(예 8192) 하다 접촉 버퍼 오버플로 에러가 나면 `gpu_max_rigid_patch_count`를 늘려 해결.
4. **환경 수 vs 속도**: `--num_envs`를 512 / 2048 / 4096로 바꿔 iteration당 시간과 수렴을 비교.

```bash
cd /isaac-sim/IsaacLab
# 환경 수만 바꿔 빠르게 감 잡기 (설정 수정 없이)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 512 --max_iterations 100
```

---

## 6. 2일차 자기점검

1. **control dt**는 얼마이며 어떻게 계산되는가? (0.02s = sim.dt 0.005 × decimation 4)
2. 물리는 200Hz인데 정책은 왜 50Hz로 느리게 도는가?
3. decimation을 키우면 반응성/안정성/계산량은 각각 어떻게 되나?
4. Implicit PD와 Explicit PD의 차이, G1이 Implicit을 쓰는 이유는?
5. legs Kp(≈178)와 feet Kp(20)가 다른 이유를 물리적으로 설명할 수 있는가?
6. action scale(0.5)과 Kp는 왜 함께 봐야 하는가?
7. domain randomization이 왜 sim-to-real에 필수인가? 이 태스크는 무엇을 랜덤화하나?
8. 4096 env 학습에서 접촉 버퍼 오버플로가 나면 어떤 파라미터를 건드리나?

---

## 다음
- 3일차: 지금까지의 **observation/action + 물리**가 **PPO**로 어떻게 정책이 되는가 (GAE·clip·엔트로피, rsl_rl 하이퍼파라미터)
