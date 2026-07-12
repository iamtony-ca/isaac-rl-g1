# 일자별 심화 개념 노트

> `STUDY_GUIDE.md`의 각 일차를 기술적으로 깊게 파고드는 노트.
> 코드 경로는 모두 `/isaac-sim/IsaacLab/` 기준. 실행은 `./isaaclab.sh -p <script>`.

---

## 1일차 — Isaac Sim & 휴머노이드 로봇 기초

### 왜 USD인가
- Omniverse의 모든 씬은 **USD(Universal Scene Description)** 로 표현. Prim(노드) 트리 + 속성(attribute) + 관계(relationship).
- 로봇 = USD 안의 **articulation**: 링크(rigid body) + 조인트(joint)로 이뤄진 트리. 루트에 `ArticulationRootAPI`가 붙음.
- 핵심 스키마: `UsdPhysics`(질량/충돌/조인트), `PhysxSchema`(솔버·마찰 등 PhysX 확장).

### articulation 해부 (휴머노이드 기준)
- **base/pelvis(floating base)**: 지면에 고정되지 않은 루트 → RL에서 로봇 자세·균형의 기준 프레임.
- **joint 종류**: revolute(회전, 대부분의 관절), prismatic(직선), fixed.
- **DOF**: 각 revolute joint = 1 DOF. 휴머노이드 다리 = hip(yaw/roll/pitch) + knee + ankle(pitch/roll).
- **actuator**: joint를 구동하는 모델. Isaac Lab은 보통 **PD 제어** (target position → torque).

### 관찰 포인트 (예습 체크리스트)
- [ ] G1/H1 USD를 GUI Stage 트리에서 열어 링크·조인트 이름 확인
- [ ] `left_hip_pitch_joint` 같은 관절 명명 규칙 파악 (reward/observation에서 정규식 `.*_hip_.*`로 참조됨)
- [ ] base link 위치가 observation의 root state가 되는 흐름 이해

### 실행
```bash
# 자산 확인용 데모 (관절 있는 로봇 스폰)
cd /isaac-sim/IsaacLab
./isaaclab.sh -p scripts/demos/quadrupeds.py    # 또는 humanoids 데모가 있으면 그것
```

---

## 2일차 — 시뮬레이션 환경 구축 (PhysX)

### 시뮬레이션 시간 구조 (가장 헷갈리는 부분)
- `sim dt`: 물리 적분 스텝 (예 1/200 s = 5 ms).
- `decimation`: 정책이 몇 물리스텝마다 한 번 행동하는지 (예 4).
- **control dt = sim dt × decimation** (예 5 ms × 4 = 20 ms → 정책 50 Hz).
- RL의 1 step = control dt 1회. 이 관계가 reward scale, action rate 페널티에 직접 영향.

### 액추에이터 = PD 게인 (튜닝 감각의 핵심)
로컬 `unitree.py`의 G1 설정 예시:
```python
"legs": ImplicitActuatorCfg(
    joint_names_expr=[".*_hip_yaw_joint", ".*_knee_joint", ...],
    stiffness={".*_knee_joint": 200.0, ...},   # Kp
    damping={".*_knee_joint": 5.0, ...},       # Kd
    effort_limit_sim=300,                      # 최대 토크
)
```
- **stiffness(Kp) ↑** → 목표 각도에 강하게 붙음(딱딱). 너무 크면 진동/불안정.
- **damping(Kd) ↑** → 움직임 감쇠(끈적). 너무 크면 반응 느림.
- **effort_limit** → 물리적으로 낼 수 있는 최대 토크. H1 무릎이 크게 잡히는 이유.
- `armature`: 모터 회전 관성 반영(수치 안정성).

### 그 밖의 물리 파라미터
- **solver iteration** (`solver_position_iteration_count`): 크면 정확·안정하나 느림. 접촉 많은 보행은 8~ 정도.
- **friction / restitution**: 발-지면 마찰이 보행 학습 난이도를 좌우. domain randomization 대상.
- **self-collision**: 휴머노이드는 보통 off(성능) 또는 선택적 on.

### 관찰 포인트
- [ ] `SceneCfg`, `ArticulationCfg`, `ActuatorCfg`가 어디서 조립되는지 추적
- [ ] Kp/Kd를 극단값으로 바꾸면 로봇이 어떻게 반응할지 가설 세우기

---

## 3일차 — RL 기초 + PPO + 균형(Standing)

### MDP로 보는 보행 태스크
- **state/observation**: base 선/각속도, 중력벡터(자세), 관절 위치/속도, 이전 action, (rough면) height scan, velocity command.
- **action**: 보통 각 관절의 **목표 위치 offset** (PD 컨트롤러가 토크로 변환). 직접 토크가 아님에 주의.
- **reward**: 여러 항의 가중합 (4일차에서 심화).
- **termination**: 넘어짐(base 높이/기울기), 시간 초과.

### PPO 한 장 요약
- **on-policy**: 현재 정책으로 모은 데이터로만 업데이트 (그래서 대량 병렬 환경으로 데이터를 빨리 모음).
- **actor-critic**: policy(actor) + value(critic) 동시 학습.
- **GAE(λ)**: advantage 추정 (bias-variance 트레이드오프, λ~0.95).
- **clip objective**: 정책이 한 번에 너무 크게 안 바뀌게 ratio를 `1±ε`로 클리핑 (ε~0.2).
- **entropy bonus**: 탐험 유지.
- 주요 하이퍼파라미터: `num_envs`, `num_steps_per_env`(rollout 길이), `learning_rate`, `gamma`, `lam`, `clip_param`, `entropy_coef`, `num_learning_epochs`, `num_mini_batches`.

### rsl_rl에서 이 값들 어디 있나
```
scripts/reinforcement_learning/rsl_rl/            # train.py / play.py
source/isaaclab_tasks/.../locomotion/velocity/config/g1/agents/  # rsl_rl_ppo_cfg.py (하이퍼파라미터)
```
- `rsl_rl_ppo_cfg.py`의 `RslRlPpoActorCriticCfg`(네트워크), `RslRlPpoAlgorithmCfg`(PPO 하이퍼파라미터)를 열어볼 것.

### Standing(균형) 실습
```bash
# 균형/보행 velocity 태스크에서 command를 0으로 두면 제자리 균형에 가까움
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 1024 --max_iterations 200
```
- [ ] reward 곡선이 오르는지 tensorboard로 확인
- [ ] `learning_rate`, `entropy_coef`를 바꿔 학습 속도/안정성 변화 관찰

---

## 4일차 — 보행(Walking) + Reward 설계/튜닝 (강좌 하이라이트)

### velocity locomotion reward 항 해부
Manager-based 보행 태스크의 전형적 reward 구성(개념):
| 보상 항 | 방향 | 역할 |
|---|---|---|
| `track_lin_vel_xy` | + | 명령한 전/횡 속도 추종 (주 목적) |
| `track_ang_vel_z` | + | 명령한 회전 속도 추종 |
| `feet_air_time` | + | 발을 적당히 들어 "걷게" 유도 (질질 끌기 방지) |
| `lin_vel_z` (수직 속도) | − | 위아래 튐 억제 |
| `ang_vel_xy` (roll/pitch) | − | 몸통 흔들림 억제 |
| `joint_torques` / `dof_acc` | − | 에너지·급격한 움직임 페널티 (부드러움) |
| `action_rate` | − | 연속 action 급변 페널티 (떨림 방지) |
| `undesired_contacts` | − | 무릎·몸통 지면 접촉 페널티 |
| `flat_orientation` | − | 몸통 수평 유지 |

### 튜닝 직관 (실습에서 체득할 것)
- tracking 보상 weight ↑ → 명령 잘 따르지만 거칠어질 수 있음.
- smoothness 페널티(torque/action_rate) ↑ → 부드럽지만 소극적/느린 보행.
- `feet_air_time` weight가 걸음걸이(보폭·리듬)를 크게 좌우.
- **reward는 상호작용**함. 한 항만 크게 바꾸면 예상 못한 걸음이 나옴 → 조금씩.

### 코드 위치 & 실습
```
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
  mdp/rewards.py                 # reward 함수 정의
  velocity_env_cfg.py            # RewardsCfg (각 항 + weight)
  config/g1/                     # G1 전용 오버라이드
```
- [ ] `RewardsCfg`에서 `feet_air_time` 또는 `track_lin_vel_xy`의 weight를 1.5~2배로 바꿔 재학습 → 걸음 변화 관찰
- [ ] curriculum(난이도 점증), command range(속도 명령 범위)도 찾아보기

### Termination/Curriculum
- termination이 너무 관대하면 넘어져도 안 끝나 학습이 지저분해짐; 너무 엄격하면 초기에 학습 신호 부족.
- terrain curriculum: 평지→거친 지형으로 점진 상승(rough 태스크).

---

## 5일차 — 센서 기반 RL (Camera, LiDAR, Vision)

### 센서 종류와 RL 연결
| 센서 | Isaac Lab | RL 관찰로의 쓰임 |
|---|---|---|
| Height scan / LiDAR | `RayCaster`, `RayCasterCamera` | 지형 높이맵 → rough terrain 보행 (exteroceptive) |
| Camera | `Camera`(RGB/Depth/Semantic) | vision 기반 정책, 장애물 회피 |
| Contact | `ContactSensor` | 발 접촉/충돌 감지 → reward·termination |
| IMU | base state | 자세/각속도 (proprioceptive) |

### proprioceptive vs exteroceptive
- **proprioceptive**(자기 상태: 관절·IMU): 항상 사용, 저차원, 학습 쉬움.
- **exteroceptive**(외부 환경: height scan/camera): 지형 인지 보행에 필요, 고차원, 학습·연산 부담 ↑.
- 전형: `Isaac-Velocity-Rough-G1-v0`는 base 주변 격자에 height scan을 쏴 observation에 포함.

### Vision RL 주의점
- 카메라는 렌더링 비용 큼 → `num_envs`를 크게 못 씀. CNN encoder 필요.
- 실습 예습: rough 태스크에서 height-scan observation term을 코드에서 찾아 차원(그리드 크기) 확인.

### 관찰 포인트
```
source/isaaclab/isaaclab/sensors/            # RayCaster, Camera, ContactSensor
.../locomotion/velocity/  ObservationsCfg 안의 height_scan term
```
- [ ] height scan 그리드 해상도가 observation 차원에 어떻게 반영되는지 확인
- [ ] sensor noise/`domain randomization`이 sim-to-real에 왜 중요한지 정리

---

## 6일차 — ROS2 연동 + 종합 프로젝트

### 학습 policy를 실제로 쓰는 파이프라인
```
[학습] rsl_rl train → checkpoint(.pt)
   ↓ export
[배포] play.py 가 policy를 ONNX/TorchScript로 export
   ↓
[추론] observation 입력 → action 출력 (경량 런타임)
   ↓
[ROS2] joint command 토픽 발행 / joint state·센서 구독
```
- `play.py` 실행 시 policy가 `exported/policy.onnx` 등으로 저장되는지 확인.

### Isaac Sim ROS2 Bridge
- **OmniGraph(Action Graph)** 노드로 ROS2 토픽 ↔ 시뮬레이션 연결.
- 주요 노드: `ROS2 Publish JointState`, `ROS2 Subscribe JointState/Twist`, clock, TF.
- `ros2 topic list`로 `/joint_states`, `/cmd_vel` 등 확인.

### sim-to-real 핵심 개념 (종합 프로젝트에서 언급될 것)
- **domain randomization**: 마찰·질량·지연·센서 노이즈를 학습 중 무작위화 → 실제 로봇에 강건.
- **observation noise & latency**: 시뮬에서 일부러 노이즈/지연을 넣어 실제와 격차 축소.
- **actuator model gap**: PD 게인·토크 한계를 실제 모터에 맞추기.

### 관찰 포인트
- [ ] ROS2 Bridge extension 활성화 방법 (Isaac Sim Extensions 창)
- [ ] policy export 산출물 위치와 포맷 확인
- [ ] 미니 프로젝트 아이디어: 학습된 G1 보행 정책을 ROS2 `/cmd_vel`로 원격 조종

---

## 부록 — 예습 중 자주 볼 명령 모음

```bash
cd /isaac-sim/IsaacLab

# 태스크 목록
./isaaclab.sh -p scripts/environments/list_envs.py

# 학습 (G1 평지)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 1024 --max_iterations 300

# 재생 (checkpoint 자동 로드, policy export 포함)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 32

# 로그 시각화
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl

# 같은 태스크를 rl_games로 비교 (심화)
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless
```
