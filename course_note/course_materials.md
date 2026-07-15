# course_materials 정독 노트 — PinkLAB 강의 원본 코드 (36강 + g1/)

> **무엇인가**: 강좌 제작사(PinkLAB)가 제공한 **실제 강의 예제 코드**(`/isaac-sim/rl_course_ws/course_materials/`)를 코드까지 전부 정독하고 정리한 노트.
> **왜 중요한가**: 기존 내 노트([[STUDY_GUIDE]]·[[DAILY_NOTES]]·[[QUICK_START]]·go2/)는 **"이미 등록된 G1/Go2 velocity 보행 태스크를 rsl_rl로 돌리고 reward를 튜닝"** 에 최적화돼 있다. 반면 이 강좌 원본은 **시뮬 스크립팅부터 로봇/센서/매니퓰레이션/모바일/RL환경/커스텀로봇/ROS2/휴머노이드 RL까지 직접 만드는** 실습 중심이라 접근법과 범위가 다르다.
> **환경**: Ubuntu 24.04 · Isaac Sim 4.5(isaacsim 5.1.0) · Isaac Lab 2.3.2 · Python 3.10+ · ROS2 Jazzy. 실행은 `./isaaclab.sh -p <예제>.py` (또는 venv python).

---

## 0. 큰 그림 — 6부 구성

이 강좌는 내 노트(G1/Go2 보행 RL 중심)보다 훨씬 넓다. 아래 순서로 **난이도·추상화가 점증**한다.

```
Part 1 (01~04)  시뮬레이션 기초         ← Isaac Sim 스크립팅 골격
Part 2 (05~14)  로봇 제어 & 센서         ← Franka 팔: 제어/센서/FK/IK/Pick&Place
Part 3 (15~19)  모바일 로봇             ← JetBot 차동구동 + 고전 내비게이션(RL 아님)
Part 4 (20~29)  강화학습 환경 ★본론      ← ManagerBasedRLEnv 바닥부터 조립 → PPO 학습
Part 5 (30~34)  커스텀 로봇 통합         ← URDF→USD, Pinky, ROS2 브릿지 + SLAM/Nav2
Part 6 (35~36)  휴머노이드 RL ★         ← G1 Standing + G1 Walking(모션 모방)
```

핵심 통찰 3가지:
1. **20강이 분기점**: 그 전(01~19)은 "수동 제어 루프"(직접 `sim.step()` 돌리며 명령), 그 후(20~)는 "MDP 구성"(`ManagerBasedRLEnv`가 step/reset/reward를 관리).
2. **강좌의 걷기 접근법은 velocity-tracking이 아니라 motion imitation**(36강, RSI + 레퍼런스 모션). 내 기존 실험(Isaac-Velocity-Flat-G1)과 **근본적으로 다르다** → 아래 §갭6에서 코드로 상술.
3. **로봇이 다양**: RL 입문은 Cartpole→ANYmal, 팔은 Franka, 모바일은 JetBot/Pinky, 휴머노이드는 G1. 내 노트는 G1/Go2만 다뤄 폭이 좁았다.

---

## Part 1. 시뮬레이션 기초 (01~04)
로봇·RL 이전에 **Isaac Sim 스크립팅 골격**을 먼저 익힘. → 개념은 [[DAY1_isaac_sim_basics]]·[[DAY2_physics_actuators]].

- **01 launch_sim** — `AppLauncher`→(모듈 import)→`SimulationContext` 구성 → `reset()`→`step()` 루프 → `close()`. 앱 라이프사이클과 "AppLauncher 이후 import" 규칙.
- **02 spawn_primitives** — `sim_utils.CuboidCfg/SphereCfg/ConeCfg...` + `PreviewSurfaceCfg`로 물리 없는 시각 프림을 Xform 아래 배치. 스폰 기본기.
- **03 galileo_experiment** — 사탑에서 질량 다른 두 공 낙하 → 착지 시각을 √(2h/g)와 비교. `MassPropertiesCfg`/`RigidBodyPropertiesCfg`, `omni.ui` Play/Reset 버튼. 강체 물리속성 부여 + View로 pose 읽기.
- **04 simulation_loop** — `RigidBodyMaterialCfg(restitution)` 바운스 차이 관찰. step/reset/dt/에피소드 주기 관리 패턴.

## Part 2. 로봇 제어 및 센서 (05~14)
Franka Panda **팔(manipulator)** 중심 — 제어·센서·운동학·매니퓰레이션. **(내 기존 노트에 전무한 영역)**

- **05 spawn_robot** — `Articulation` + `FRANKA_PANDA_CFG`, `write_joint_state_to_sim`, `set_joint_effort_target`(랜덤 토크). Articulation 상태 API.
- **06 robot_joint_control** — 사인파 위치 명령 + matplotlib 실시간 목표 vs 실제 비교. PD 위치 제어와 추종 오차.
- **07 scene_design** — `@configclass` + `InteractiveSceneCfg`, `{ENV_REGEX_NS}`, `env_origins`로 다중 환경 자동 복제. **선언적 씬 설계 패턴(이후 전부 여기 기반).**
- **08 multi_robot** — 한 씬에 Cartpole+Franka 동시, `env_ids`로 특정 환경만 선택 제어.
- **09 camera_sensor** — `Camera`/`PinholeCameraCfg`, `--enable_cameras` 필수, RGB/Depth 텐서 추출·시각화.
- **10 ray_caster** — `RayCaster` + `GridPatternCfg`(격자)로 계단 높이 스캔. RayCaster는 **단일 삼각메시 타겟** 필요.
- **11 contact_sensor** — 레버에 동일 토크 → `ContactSensor`(`net_forces_w`)로 접촉력 측정, 토크 정의(F=τ/L) 검증. `stiffness=0/damping=0` 순수 토크 액추에이터, 런타임 URDF→USD.
- **12 forward_kinematics** — `data.body_pose_w`로 관절각→EE pose(FK) 관찰, `VisualizationMarkers`로 프레임 표시.
- **13 diff_ik** — `DifferentialIKController`(DLS), `get_jacobians()` + `subtract_frame_transforms`로 목표 pose 추종 IK.
- **14 pick_and_place** — IK 팔 + 그리퍼를 **상태 머신**(APPROACH→GRASP→LIFT→…)으로 통합한 집기/놓기.

## Part 3. 모바일 로봇 (15~19)
JetBot **차동 구동**과 고전 내비게이션(RL 아님). **(내 기존 노트에 전무한 영역)**

- **15 spawn_mobile_robot** — 유니사이클(v,ω)→좌/우 바퀴 각속도 변환, `set_joint_velocity_target`.
- **16 mobile_base_control** — 사각형/원/8자 경로 + **바퀴 오도메트리**(dead-reckoning) vs 실제 위치 오차.
- **17 mobile_navigation** — `atan2` heading 오차 기반 **P-제어 waypoint 추종**, 마커 시각화.
- **18 lidar_maze** — `LidarPatternCfg`(360°/1° = 360 rays) 2D LiDAR를 미로에서 실시간 polar 시각화. *(README 목차 표엔 18이 누락돼 있음 — 파일은 존재)*
- **19 obstacle_avoidance** — 거리 기반 반응형 회피(NAV↔AVOID 상태 전환), 궤적 Trail.

## Part 4. 강화학습 환경 (20~29) — 강좌의 RL 본론
Manager 기반 RL 환경을 **바닥부터 직접 조립**. → 이론은 [[DAY3_ppo_rsl_rl]]·[[PPO_TUNING]]·[[TENSORBOARD]]. 코드 심화는 아래 §갭5.

- **20 base_env** — `ManagerBasedRLEnv` 6대 구성요소(Scene/Action/Obs/Event/Reward/Termination) 최소 Cartpole. `env.step()`→obs/rew/terminated/truncated.
- **21 observation_reward** — 커스텀 ObsTerm(sin/cos, 정규화) + 커스텀 RewTerm(가우시안 목표추적/에너지 페널티), 텀별 기여도 디버깅.
- **22 action_space** — Effort/Position/Velocity 액션 타입 비교, scale/offset 매핑.
- **23 custom_env_complete** — 20~22 종합 + 커스텀 termination까지 **완전한 Cartpole RL 환경** 구축(24강 재사용).
- **24 train_cartpole** — `RslRlVecEnvWrapper` + `OnPolicyRunner`(PPO clip 0.2, GAE 0.95, adaptive KL)로 학습, 체크포인트/TensorBoard.
- **25 evaluate_policy** — 체크포인트 자동 탐색·`load()`·`get_inference_policy()`, 성공률/보상 평가(네트워크 Cfg 일치 필수).
- **26 train_locomotion** — `gym.make("Isaac-Velocity-Flat-Anymal-B-v0")` 사전 등록 태스크로 **ANYmal 보행** PPO 학습(관측 ~48, 다중목표 보상).
- **27 evaluate_locomotion** — 26 정책 추론·시각화, timeout 생존율.
- **28 terrain_generation** — `TerrainGeneratorCfg`(계단/랜덤그리드/웨이브)로 거친 지형 생성, 평지 학습 정책 그대로 적용해 **도메인 시프트(일반화 실패)** 관찰. → [[DAY5_sensors]]의 rough terrain과 연결.
- **29 domain_randomization** — `EventTerm` 3모드(startup 질량·reset 관절·interval 외력/push)로 **도메인 랜덤화** → sim-to-real gap 완화. → [[DAY6_ros2_sim2real]].

## Part 5. 커스텀 로봇 통합 — Pinky Pro (30~34)
실제 ROS2 로봇을 시뮬에 넣고 ROS2로 구동. → [[CUSTOM_ROBOT]]·[[DAY6_ros2_sim2real]]. 코드 심화는 아래 §갭8.

- **30 urdf_preparation** — ROS2 `xacro`→URDF, `package://` 경로 치환, link/joint/mesh 구조 점검(순수 Python).
- **31 urdf_to_usd** — `UrdfConverter`/`UrdfConverterCfg`(floating base, convex_hull, JointDriveCfg)로 **URDF→USD 변환** 후 스폰·검증.
- **32 pinky_control** — 변환 USD로 차동 구동 + P-제어 사각형 waypoint 주행.
- **33 pinky_sensors** — Pinky에 LiDAR/카메라 부착(fixed-joint 병합 후 offset), 센서 데이터 판독.
- **34 ros2_bridge** — **"Gazebo 대체"**. `isaacsim.ros2.bridge` + **OmniGraph** 노드로 `/clock /tf /odom /joint_states /scan` 게시 + `/cmd_vel` 구독, RTX LiDAR. **34-1**: `slam_toolbox` SLAM + **Nav2** 자율주행(8×8 미로), `pinky_slam_nav` ROS2 패키지(slam/nav/slam_nav launch, nav2_params, RViz) 포함 — 매핑→지도 저장→주행 워크플로우.

## Part 6. 휴머노이드 강화학습 (35~36) ★
코드 심화는 아래 §갭7(standing)·§갭6(walking).

- **35 g1_standing** — **G1 서 있기(정적 균형)**. `35_1` 빈 환경 랜덤 액션(넘어짐 baseline) → `35_2` PPO 학습 → `35_3` 평가+토크 그래프. 공유 `g1_standing_env.py`가 핵심 보상 = **높이 0.74m 유지(가우시안, weight 15)** + flat_orientation + 페널티들, 종료 = 기울기45°/높이<0.35.
- **36 g1_walking** — **G1 걷기 = 모션 이미테이션(imitation)**. `36_1` 리타게팅 사람 모션(.pkl) kinematic 재생 → `36_2` **RSI**(Reference State Initialization) + PPO 모방 학습 → `36_3` 평가(관절추종 RMS/전진속도/생존율). `g1_walking_env.py`: 관측 103차원(기본 + 레퍼런스 관절각 + gait phase sin/cos), 보상 = joint_tracking(핵심)/arm/speed/height 추종 + feet_slide(접촉센서), `reset_to_motion_frame`=RSI. 모션 데이터는 HuggingFace `openhe/g1-retargeted-motions` 필요.
- **g1/g1.py** — 36강을 모듈화하기 **이전의 단일 파일 원형**(experiment `g1_walking_2`). `G1_MINIMAL_CFG` 그대로 사용, 별도 로봇 config 정의는 아님.

---

# 심화: 내 노트에서 빠졌던 영역 — 코드 기반 해설

아래는 [[STUDY_GUIDE]] 대비 특히 비어 있던 **갭 5·6·7·8**을 실제 소스로 뜯어본 것. 학습 순서상 **5(RL 환경 바닥부터) → 7(Standing) → 6(Walking 모방) → 8(URDF→USD)** 로 읽는 게 자연스럽다.

---

## 갭 5 — Manager 기반 RL 환경을 "바닥부터" 만들기 (20~24, Cartpole)

내 노트는 **기존 reward cfg를 읽기**는 있어도 **처음부터 만들기**가 약했다. 강좌는 Cartpole로 6대 구성요소를 손으로 조립한다. 핵심 = **MDP를 함수 6종 + configclass 6종으로 쪼갠다.**

### 5-1. MDP 함수는 그냥 `(env) -> torch.Tensor` 다
관측·보상·종료는 전부 **env를 받아 (num_envs,) 또는 (num_envs, D) 텐서를 반환하는 함수**. GPU 병렬이라 for문 없이 텐서 연산.
```python
# 관측: 폴 각도를 sin/cos로 (불연속 wrap 없는 표현)
def obs_pole_sin_cos(env):
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    angle = robot.data.joint_pos[:, joint_ids[0]]
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)  # (N, 2)

# 보상: 직립일수록 1 (cos), 가우시안 중앙 보상, 에너지 페널티
def rew_pole_upright(env):   return torch.cos(angle)                       # 목표
def rew_cart_center(env, sigma=1.0): return torch.exp(-pos**2/(2*sigma**2))# 보조목표
def rew_energy(env):         return torch.sum(env.action_manager.action**2, -1) # 페널티

# 종료: 폴이 90도 넘게 기울면 True
def term_pole_too_far(env, max_angle=0.5*math.pi):
    return torch.abs(angle) > max_angle
```
> 포인트: `env.scene["robot"]`로 로봇, `robot.data.joint_pos`로 상태, `env.action_manager.action`으로 직전 액션에 접근. 이 패턴이 35·36강 G1까지 그대로 쓰인다.

### 5-2. 6대 configclass로 조립
```python
@configclass
class MyCartpoleSceneCfg(InteractiveSceneCfg):     # ① Scene: 바닥+로봇+조명
    robot = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class MyActionsCfg:                                # ② Action: 카트에 힘
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot",
        joint_names=["slider_to_cart"], scale=100.0)

@configclass
class MyObservationsCfg:                           # ③ Observation
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)     # 내장 함수
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        pole_sin_cos  = ObsTerm(func=obs_pole_sin_cos)      # 커스텀 함수
        cart_norm     = ObsTerm(func=obs_cart_normalized,
                                params={"max_pos":3.0}, clip=(-1.0,1.0))
        def __post_init__(self):
            self.concatenate_terms = True           # 텀들을 한 벡터로 이어붙임 → 7차원
    policy: PolicyCfg = PolicyCfg()

@configclass
class MyEventCfg:                                  # ④ Event: 리셋 시 랜덤 초기화
    reset_pole = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "position_range": (-0.25*math.pi, 0.25*math.pi), ...})

@configclass
class MyRewardsCfg:                                # ⑤ Reward: 텀 = 함수 × weight
    alive        = RewTerm(func=mdp.is_alive,      weight=1.0)
    terminating  = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_upright = RewTerm(func=rew_pole_upright,  weight=2.0)   # 핵심 목표
    cart_center  = RewTerm(func=rew_cart_center,   weight=0.5, params={"sigma":1.5})
    energy       = RewTerm(func=rew_energy,        weight=-0.001)

@configclass
class MyTerminationsCfg:                           # ⑥ Termination
    time_out           = DoneTerm(func=mdp.time_out, time_out=True)  # truncation
    cart_out_of_bounds = DoneTerm(func=mdp.joint_pos_out_of_manual_limit, ...)
    pole_too_far       = DoneTerm(func=term_pole_too_far, ...)
```
그리고 이들을 하나로 묶는 **EnvCfg + `__post_init__`(물리 파라미터)**:
```python
@configclass
class MyCartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = MyCartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    actions = MyActionsCfg(); observations = MyObservationsCfg()
    events = MyEventCfg(); rewards = MyRewardsCfg(); terminations = MyTerminationsCfg()
    def __post_init__(self):
        self.decimation = 2            # 정책 1회 → 물리 2회
        self.episode_length_s = 5.0
        self.sim.dt = 1.0/120.0        # 물리 120Hz → 제어 60Hz  ([[DAILY_NOTES]] 2일차 control dt 개념)
```

### 5-3. 학습(24) — 래퍼 한 줄 + Runner 한 줄
```python
env = ManagerBasedRLEnv(cfg=env_cfg)      # ① Isaac Lab 환경
env = RslRlVecEnvWrapper(env)             # ② rsl_rl VecEnv 인터페이스로 변환 (핵심!)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=..., device=...)
runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)
```
PPO 하이퍼파라미터는 `RslRlOnPolicyRunnerCfg`(수집 스텝·반복수) + `RslRlPpoActorCriticCfg`(net [32,32], elu) + `RslRlPpoAlgorithmCfg`(clip 0.2, γ 0.99, GAE λ 0.95, adaptive KL). → 값 의미는 [[PPO_TUNING]] 참고. Cartpole은 150 iter면 충분.

> **내 실습과의 차이**: 나는 `train.py --task ...`(등록된 태스크)만 돌렸다. 여기선 **train 스크립트 안에서 env를 직접 정의**하므로 `gym.make`도, 태스크 등록도 없다. 35·36강 G1도 이 "스크립트에 env 내장" 방식이다.

**체크리스트**
- [ ] `concatenate_terms=True`가 관측 텀들을 한 벡터로 잇는다 → 관측 차원 = 텀 차원 합(=7)임을 이해
- [ ] `RewTerm`의 부호(+목표 / −페널티)와 weight 배분 감각
- [ ] `RslRlVecEnvWrapper`가 없으면 rsl_rl `OnPolicyRunner`가 env를 못 먹는다는 점

---

## 갭 7 — G1 Standing 커스텀 환경 (35, `g1_standing_env.py`)

Cartpole(갭5)과 **완전히 같은 골격**에 로봇만 G1으로 바꾼 것. "서 있기"의 정체 = **몸체 높이 0.74m 유지**를 가우시안 보상으로 준다.

### 7-1. 핵심 = 높이 유지 보상
```python
def base_pos_z_reward(env):
    z = env.scene["robot"].data.root_pos_w[:, 2]     # 몸체 world z (높이)
    return torch.exp(-torch.square(z - 0.74) / 0.01) # 0.74m에서 멀어지면 급격히 0
```
분모 `0.01`이 가우시안 폭 → 몇 cm만 벗어나도 보상이 확 떨어진다(날카로운 목표).

### 7-2. 관측 6항목 / 보상 6항목 / 종료 3조건
```python
class ObservationsCfg.PolicyCfg(ObsGroup):
    base_lin_vel      = ObsTerm(func=mdp.base_lin_vel)        # 몸체 선속도
    base_ang_vel      = ObsTerm(func=mdp.base_ang_vel)        # 몸체 각속도
    projected_gravity = ObsTerm(func=mdp.projected_gravity)   # 중력 투영 = "얼마나 기울었나"
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)              # 관절각(기본자세 상대)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    last_action = ObsTerm(func=mdp.last_action)              # 직전 행동 (떨림 억제 학습에 필요)

class RewardsCfg:
    alive_bonus      = RewTerm(mdp.is_alive,             weight= 1.0)
    height_reward    = RewTerm(base_pos_z_reward,        weight=15.0)   # ★ 핵심
    flat_orientation = RewTerm(mdp.flat_orientation_l2,  weight=-5.0)   # 기울면 감점
    joint_vel        = RewTerm(mdp.joint_vel_l2,         weight=-0.01)  # 마구 흔들면 감점
    action_rate      = RewTerm(mdp.action_rate_l2,       weight=-0.1)   # 급변 감점(부드러움)
    joint_limits     = RewTerm(mdp.joint_pos_limits,     weight=-1.0)

class TerminationsCfg:
    time_out        = DoneTerm(mdp.time_out, time_out=True)                    # 5초
    bad_orientation = DoneTerm(mdp.bad_orientation, params={"limit_angle":0.7854}) # 45°
    base_height     = DoneTerm(base_height_termination, params={"limit":0.35})     # 추락
```
액션은 `JointPositionActionCfg(joint_names=[".*"], scale=0.25)` — **전 관절 목표 위치, scale 0.25로 과격한 움직임 억제**([[DAILY_NOTES]] 3일차: action = 목표 위치 offset, PD가 토크로 변환).

물리: `decimation=4`, `sim.dt=0.005`(200Hz) → 제어 50Hz. 512 env × 1000 iter 기본.
PPO net `[128,64,32]`, lr 1e-3 (`make_ppo_runner_cfg`).

### 7-3. 학습→평가에서 반드시 지킬 것
`g1_standing_env.py` **한 모듈**을 35_1(랜덤)·35_2(학습)·35_3(평가)가 공유한다. 이유: 평가(35_3)가 체크포인트를 로드할 때 **관측 정의·네트워크 형상이 학습과 1비트라도 다르면 로드가 깨진다**(갭5의 25강 교훈과 동일). → 수정은 이 모듈만.

**체크리스트**
- [ ] "서 있기 = 고정 높이 유지"라는 보상 설계 아이디어
- [ ] `projected_gravity`가 기울기 감지 관측인 이유
- [ ] 학습/평가 env 정의 일치의 필요성

---

## 갭 6 — G1 Walking = 모션 이미테이션 (36, `g1_walking_env.py`) ★ 가장 큰 갭

> **내 기존 접근과 근본적으로 다름.** 나는 걷기를 `Isaac-Velocity-Flat-G1`(velocity **command 추종**)으로 했다. 강좌는 **사람 모션캡처를 G1으로 리타게팅한 "정답 동작"을 매 순간 따라 하도록** 보상을 준다(imitation). 자연스러운 걸음걸이를 훨씬 쉽게 얻는다.

### 6-1. 레퍼런스 모션 = 시간축 위의 "정답 자세 테이블"
`.pkl`에는 프레임별 `dof`(관절각), `root_trans`(위치), `root_rot`(xyzw!), `contact_mask`가 들어 있다. `MotionReference` 클래스가 이를 로드해서:
- **순환 구간 트림**: 기본 클립 `B3_-_walk1`의 프레임 42~155(3.77초)만 잘라 무한 반복(modulo). 시작=끝 관절각 차이 ≤0.03rad라 이어붙여도 끊김이 없음.
- **사전 계산**: 프레임별 전진 속도 `ref_speed`(≈1.07m/s), 루트 높이 `ref_root_z`, 관절 속도 `ref_dof_vel`(유한 차분).
- **관절 매핑**: pkl 23관절 → 로봇 관절 인덱스(`build_joint_map`). G1_MINIMAL엔 손목이 없어 `wrist_roll→elbow_roll`로 근사(`ALIAS`).
- **현재 프레임 번호**: `frame_idx = (phase0 + t·fps) % T` — 각 env마다 시작 위상 `phase0`가 다름(RSI).

### 6-2. 관측에 "레퍼런스"와 "위상"을 넣는다 (35강 6항목 → 103차원)
```python
def ref_joint_obs(env):     # 지금 취해야 할 레퍼런스 관절각 (23) — 정책에 "정답"을 알려줌
    idx = MOTION.frame_idx(env)
    return MOTION.dof[idx] - default_joint_pos

def gait_phase_obs(env):    # 보행 주기 위상 sin/cos (2) — 주기 시작=끝 연속
    phase = MOTION.frame_idx(env) / MOTION.T * 2*pi
    return torch.stack([sin(phase), cos(phase)], 1)
```
관측 = 35강의 6항목(속도·중력·관절·직전행동) + **`ref_joint`(23) + `phase`(2)**. 단, 여기선 손가락 제외한 **몸체 23관절**만(`BODY_JOINT_EXPR`)을 관측/행동에 씀.

### 6-3. 보상 = "레퍼런스와의 거리"를 exp 커널로
```python
def joint_tracking_reward(env):     # ★핵심: 전신 23관절이 레퍼런스에 가까울수록 1
    cur = robot.data.joint_pos[:, MOTION.joint_map]
    err = torch.sum(torch.square(cur - MOTION.dof[idx]), dim=1)
    return torch.exp(-1.0 * err)

def speed_tracking_reward(env):     # 전진속도가 레퍼런스 속도에 가까울수록
    vx = robot.data.root_lin_vel_b[:, 0]
    return torch.exp(-4.0 * torch.square(vx - MOTION.ref_speed[idx]))
    # ⚠ "빠를수록 보상"(unbounded)으로 하면 몸을 던지는 돌진을 배움 → 반드시 목표추종 형태
```
보상 텀 구성:
| 텀 | weight | 의미 |
|---|---|---|
| `joint_tracking` | **+5.0** | 전신 자세 추종 (핵심) |
| `arm_tracking` | +2.0 | 팔 흔들기 추종(자연스러움) |
| `speed_tracking` | +2.0 | 목표 전진속도 추종 |
| `height_reward` | +2.0 | **프레임별** 레퍼런스 높이 추종(걷을 땐 높이가 리듬있게 변함 → 35강 고정 0.74와 다름) |
| `flat_orientation` | −2.0 | 과도한 기울기 감점 |
| `feet_slide` | −0.1 | 발이 닿은 채 미끄러지면 감점(**ContactSensor** 사용, `isaaclab_tasks`의 velocity_mdp 재사용) |
| `joint_vel`/`action_rate`/`joint_limits` | − | 부드러움/한계 |

### 6-4. RSI (Reference State Initialization) — 모방 학습의 심장
```python
def reset_to_motion_frame(env, env_ids):        # EventTerm(mode="reset")
    f = torch.randint(0, MOTION.T, (len(env_ids),))   # (0) 임의 프레임 f 뽑기
    MOTION.phase0[env_ids] = f                         #     위상 동기화
    # (1) 그 프레임의 관절각/관절속도를 로봇에 직접 기록
    joint_pos[:, MOTION.joint_map] = MOTION.dof[f]
    joint_vel[:, MOTION.joint_map] = MOTION.ref_dof_vel[f]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    # (2) 루트 높이/전진속도도 레퍼런스에 맞춤 → "걷던 도중"처럼 시작
    root_state[:, 2]  = MOTION.ref_root_z[f]
    root_state[:, 7]  = MOTION.ref_speed[f]
    robot.write_root_pose_to_sim(...); robot.write_root_velocity_to_sim(...)
```
**왜 필요한가**: 항상 "차렷 자세 + 프레임 0"에서 시작하면 ① 보행 주기 뒷부분 상태를 경험하기 어렵고 ② 시작 자세와 레퍼런스 위상이 어긋난다. RSI는 매 리셋마다 주기의 **임의 지점**에서, **이미 그 속도로 걷던 상태**로 출발시켜 전 구간을 골고루 학습시킨다.
> **연관 함정**: `36_2_train_walking.py`는 `runner.learn(..., init_at_random_ep_len=False)`. RSI가 이미 시작 위상을 흩뜨리므로, rsl_rl의 `init_at_random_ep_len=True`(에피소드 길이 랜덤화)까지 겹치면 위상이 이중으로 어긋나 모방이 붕괴한다. (갭5 Cartpole은 RSI가 없어 `True`였음 — 대조 포인트)

PPO net `[256,128,64]`(103차원+어려운 과제), `entropy_coef=0.008`(레퍼런스가 답을 주니 탐색 덜 필요), 2048 env × 2000 iter.

**velocity-tracking(내 기존) vs motion-imitation(강좌) 요약**
| | velocity-tracking (내 노트) | motion-imitation (36강) |
|---|---|---|
| 정답 | 속도 **command**를 추종 | 레퍼런스 **모션(자세 시퀀스)** 추종 |
| 관측 | velocity_commands 포함 | ref_joint + gait_phase 포함 |
| 보상 핵심 | track_lin_vel_xy | joint_tracking (exp 자세 오차) |
| 초기화 | 기본 자세 | **RSI**(임의 프레임 + 그 속도) |
| 걸음 자연스러움 | reward shaping에 크게 의존 | 사람 모션 그대로라 자연스러움 |
| 추가 데이터 | 불필요 | 리타게팅 모션 .pkl 필요(HuggingFace) |

**체크리스트**
- [ ] `frame_idx = (phase0 + t·fps) % T`로 각 env가 서로 다른 위상을 도는 구조
- [ ] 보상이 전부 `exp(-오차)` 커널인 이유(부드러운 0~1, unbounded 회피)
- [ ] RSI가 없으면/위상 랜덤화가 겹치면 왜 학습이 깨지는지
- [ ] 걷기 높이는 고정이 아니라 **프레임별 레퍼런스 높이** 추종

> ▶ **직접 돌려보기**: 이 파이프라인(모션 clone → 36_1 재생 → 36_2 학습 → 36_3 평가)을 단계별 명령·트러블슈팅과 함께 정리한 실습 가이드 → [[IMITATION_WALKING_PRACTICE]]

---

## 갭 8 — URDF → USD 커스텀 로봇 파이프라인 (30~31, Pinky Pro)

내 [[CUSTOM_ROBOT]]엔 개념만 있었다. 실제 절차는 **① xacro→URDF(ROS2) → ② URDF→USD(Isaac Lab) → ③ ArticulationCfg로 스폰**.

### 8-1. 30강: xacro→URDF + 구조 점검 (순수 Python, Isaac Sim 미사용)
```python
# ROS2 Jazzy의 xacro를 subprocess로 실행 (ament_index가 패키지를 찾도록 임시 워크스페이스 심링크 구성)
cmd = ("source /opt/ros/jazzy/setup.bash && "
       "export AMENT_PREFIX_PATH='/tmp/pinky_ws:/opt/ros/jazzy' && "
       f"xacro {XACRO_FILE} namespace:='' is_sim:=false ... -o {URDF_OUTPUT}")
subprocess.run(["bash","-c",cmd], ...)
# 그리고 URDF 안 'package://pinky_description/' → 절대경로로 치환 (USD 변환기가 경로를 못 읽는 문제 예방)
content = content.replace("package://pinky_description/", f"{PINKY_DESC}/")
```
이어서 `xml.etree.ElementTree`로 **link/joint 트리, 질량 합, movable joint(continuous/revolute/prismatic), mesh 파일 존재**를 검증. → 변환 전 "이 URDF가 온전한가"를 눈으로 확인하는 단계.

### 8-2. 31강: UrdfConverter로 USD 생성 — 변환 옵션이 핵심
```python
urdf_cfg = UrdfConverterCfg(
    asset_path=URDF_PATH, usd_dir=..., usd_file_name="pinky_pro.usd",
    fix_base=False,              # 모바일 로봇 → floating-base (휴머노이드도 False; 팔은 True)
    merge_fixed_joints=True,     # fixed로 이어진 링크 병합 → 성능↑ (센서 부착 시 base_link이 사라져 offset 필요 — 33강 함정)
    self_collision=False,
    collider_type="convex_hull", # 충돌 메시 근사 방식
    joint_drive=UrdfConverterCfg.JointDriveCfg(   # 바퀴: 속도 제어 드라이브
        drive_type="force", target_type="velocity",
        gains=PDGainsCfg(stiffness=0.0, damping=10.0)),  # Kp=0(위치제어 off), Kd=10(속도제어)
    force_usd_conversion=True,   # 개발 중 항상 재변환
)
converter = UrdfConverter(urdf_cfg)
usd_path = converter.usd_path
```
변환된 USD를 **ArticulationCfg**로 감싸 스폰(액추에이터는 여기서도 다시 지정):
```python
PINKY_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0,0,0.05)),
    actuators={"wheels": ImplicitActuatorCfg(
        joint_names_expr=[".*wheel.*"], stiffness=0.0, damping=10.0)},
)
```
그다음은 익숙한 패턴: `InteractiveScene`으로 스폰 → 관절/바디 구조 출력 → **차동 구동**으로 바퀴를 굴려 검증.
```python
WHEEL_RADIUS, WHEEL_BASE = 0.028, 0.0811
def unicycle_to_wheel_vel(v, omega):        # (v,ω) → 좌/우 바퀴 각속도
    v_left  = (v - omega*WHEEL_BASE/2)/WHEEL_RADIUS
    v_right = (v + omega*WHEEL_BASE/2)/WHEEL_RADIUS
    return v_left, v_right
robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)
```

> **G1과의 대비**: G1(35·36)은 `isaaclab_assets`에 **이미 등록된 `G1_MINIMAL_CFG`**를 쓰므로 이 변환 과정이 필요 없다. 이 파이프라인은 **저장소에 없는 내 로봇**을 넣을 때의 절차다. Go2와의 액추에이터 차이(DC-Motor vs Implicit PD)는 [[go2/GO2_STRUCTURE_ACTUATOR]] 참고.

**체크리스트**
- [ ] `fix_base` (모바일/휴머노이드 False vs 팔 True), `merge_fixed_joints`(성능↔센서 offset) 트레이드오프
- [ ] `JointDriveCfg`의 stiffness=0/damping으로 "속도 제어 바퀴" 만드는 법
- [ ] `package://` → 절대경로 치환을 왜 하는지
- [ ] 변환 후에도 `ArticulationCfg.actuators`에서 액추에이터를 다시 정의한다는 점

---

## 부록 — 실행 명령 빠른 참고
```bash
cd /isaac-sim/IsaacLab   # 또는 강의 지침대로 venv python

# 개별 예제 (예: 20강)
./isaaclab.sh -p /isaac-sim/rl_course_ws/course_materials/20_base_env/20_base_env.py --headless

# Cartpole 학습 → 평가
./isaaclab.sh -p .../24_train_cartpole/24_train_cartpole.py --headless --num_envs 1024 --max_iterations 300
./isaaclab.sh -p .../25_evaluate_policy/25_evaluate_policy.py

# G1 Standing (35): 랜덤 → 학습 → 평가
./isaaclab.sh -p .../35_g1_standing/35_1_random_actions.py
./isaaclab.sh -p .../35_g1_standing/35_2_train_standing.py --headless
./isaaclab.sh -p .../35_g1_standing/35_3_play_standing.py

# G1 Walking (36): 모션재생 → 모방학습 → 평가  (먼저 모션 데이터 clone 필요)
git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions \
  .../36_g1_walking/g1_retargeted_motions
./isaaclab.sh -p .../36_g1_walking/36_1_motion_playback.py
./isaaclab.sh -p .../36_g1_walking/36_2_train_walking.py --headless
./isaaclab.sh -p .../36_g1_walking/36_3_play_walking.py

# 커스텀 로봇 (30~31): URDF 준비 → USD 변환
cd .../30_urdf_preparation && git clone https://github.com/pinklab-art/pinky_pro.git
python .../30_urdf_preparation/30_urdf_preparation.py
./isaaclab.sh -p .../31_urdf_to_usd/31_urdf_to_usd.py
```

## 관련 노트
- 이론 공유: [[STUDY_GUIDE]] · [[DAILY_NOTES]] · [[DAY3_ppo_rsl_rl]] · [[PPO_TUNING]] · [[TENSORBOARD]] · [[DAY6_ros2_sim2real]]
- 커스텀 로봇/센서: [[CUSTOM_ROBOT]] · [[DAY5_sensors]]
- Go2 델타(4족 비교): [[go2/GO2_VS_G1]] · [[go2/GO2_REWARD]]
