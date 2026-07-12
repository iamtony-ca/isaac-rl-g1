# Isaac Sim 로봇 강화학습 강좌 — 기술 스택 예습 가이드

> 대상 강좌: PinkLab "Isaac Sim 로봇 강화학습 교육" (6일 / 48시간)
> 로컬 환경: **Isaac Sim 5.1.0 + Isaac Lab 2.3.2** (`/isaac-sim/`)
> 설치된 RL 프레임워크: `rsl_rl` ✅, `rl_games` ✅ (skrl / sb3 / ray 는 미설치)
> 실행 파이썬: `/isaac-sim/python.sh` (Kit 내장 Python, 시스템 pip 아님)

---

## 0. 예습 전 큰 그림 (반드시 먼저 이해)

강좌 기술 스택은 아래 4개 층으로 나눠서 보면 됩니다.

```
[ ROS2 Bridge ]          ← 6일차: 외부 연동
[ 강화학습 라이브러리 ]   ← rsl_rl / rl_games (PPO)
[ Isaac Lab ]            ← 환경/태스크 정의 (Manager-based vs Direct)
[ Isaac Sim / Omniverse ] ← PhysX 물리, USD 씬, 렌더링, 센서
```

핵심 개념 축 3가지:
1. **USD + PhysX**: 로봇·씬은 USD, 물리는 PhysX. articulation(관절 트리) 이해가 전부의 기초.
2. **병렬 시뮬레이션**: Isaac Lab은 GPU에서 수천 개 환경(`num_envs`)을 동시에 굴려 RL 데이터를 모음. 이게 sim RL이 빠른 이유.
3. **Manager-based vs Direct 워크플로우**:
   - *Manager-based*: Observation/Action/Reward/Termination을 "Manager" 설정 클래스로 조립 (보상함수 튜닝 실습에 유리 → 강좌 4일차 핵심).
   - *Direct*: 하나의 Env 클래스에 직접 코드로 작성 (성능·유연성).

---

## 로봇 선택: Unitree G1 vs H1 — 어느 걸로 예습할까?

> 결론: **G1로 예습하세요.** 참고자료가 더 풍부하고, Isaac Lab 안에서도 태스크 종류가 더 많습니다.

### 하드웨어 스펙 비교

| 항목 | **Unitree G1** | **Unitree H1** |
|---|---|---|
| 포지셔닝 | 교육·연구 보급형 | 엔터프라이즈·고성능 연구 |
| 키 | 약 1,320 mm (작음) | 약 1,800 mm (큼) |
| 최고 속도 | ~2 m/s | ~3.3 m/s |
| 무릎 최대 토크 | 낮음 | ~360 N·m (업계 최상위급) |
| 가격(실물) | ~$16,000 | ~$90,000+ |
| DOF | 기본 23 (EDU 최대 43, RL은 29-DOF 흔함) | 기본 19 (H1-2는 팔 7-DOF로 확장) |
| 손 | 3/5-DOF Dexterous hand 옵션 (조작 연구) | 기본 없음(고가 옵션) |

> 로컬 `unitree.py`의 G1 설정을 보면 legs / feet / **arms(+손가락 joint: one/two/three...)** 액추에이터 그룹이 정의돼 있어, 다리뿐 아니라 팔·손까지 제어 가능한 구조임을 확인할 수 있습니다.

### 왜 예습은 G1이 유리한가 (핵심)

1. **참고자료 풍부도: G1 압승.**
   - 값이 싸고 교육용으로 널리 보급 → 커뮤니티 튜토리얼/영상/블로그/논문 재현 코드가 훨씬 많음.
   - Unitree 공식 RL 저장소(`unitree_rl_gym`, `unitree_rl_lab`)가 **G1(29-DOF)** 중심으로 관리됨.
   - Isaac Lab / MuJoCo / ROS2 예제도 G1이 더 다양.
2. **Isaac Lab 내 태스크 종류: G1이 더 많음** (이 로컬 환경 기준 실측):
   - G1: `Isaac-Velocity-Flat-G1-v0`, `Isaac-Velocity-Rough-G1-v0` **+ 로코-매니퓰레이션/PickPlace** (`Isaac-PickPlace-Locomanipulation-G1-Abs-v0`, `Isaac-PickPlace-G1-InspireFTP-Abs-v0` 등) **+ OpenXR 텔레오퍼레이션 리타게터**.
   - H1: `Isaac-Velocity-Flat-H1-v0`, `Isaac-Velocity-Rough-H1-v0` (보행 velocity 태스크 위주).
3. **학습 부담**: G1이 더 작고 가벼워 초기 보행 학습이 상대적으로 안정적. H1은 크고 무거워 튜닝 난이도가 조금 더 높음.

### 그럼 H1은 언제?
- 강좌 자료가 H1을 지정하거나, 큰 로봇의 동역학(높은 토크·관성)을 다뤄보고 싶을 때.
- 두 로봇의 태스크 구조(observation/reward)가 거의 동일하므로, **G1로 익힌 뒤 `--task`만 H1로 바꿔 비교**하는 게 가장 남는 게 많은 예습법입니다.

> 주의: 강좌가 어떤 로봇/DOF 구성을 쓸지는 강사 자료에 따라 다를 수 있음. DOF 수치는 모델·옵션(EDU/hands/waist)에 따라 달라지니 참고값으로만.

---

## PART 1 — 기초 (1~3일차)

### 1일차: Isaac Sim & 휴머노이드 로봇 기초
**예습 목표**: Omniverse/USD 개념, articulation(관절), GUI 조작.

- 개념: USD Stage / Prim / Xform, articulation root, joint(revolute/prismatic), DOF
- 실습 준비:
  ```bash
  # GUI 실행 (원격이면 스트리밍/헤드리스 고려)
  /isaac-sim/isaac-sim.sh
  ```
- 예습 포인트: 휴머노이드 URDF/USD의 관절 구조, base link, actuator 개념.
- 참고 자산: Isaac Lab에 **Unitree H1, G1** 휴머노이드 내장.

### 2일차: 시뮬레이션 환경 구축 (PhysX)
**예습 목표**: 물리 파라미터가 학습에 미치는 영향 이해.

- 개념: simulation dt, decimation(제어 주기), solver iteration, friction, restitution, self-collision, actuator 모델(PD gain: stiffness/damping)
- Isaac Lab 구조 살펴보기:
  ```
  /isaac-sim/IsaacLab/source/isaaclab/          # 코어 (sim, assets, sensors, managers)
  /isaac-sim/IsaacLab/source/isaaclab_tasks/    # 등록된 태스크(환경)들
  /isaac-sim/IsaacLab/source/isaaclab_rl/       # RL 라이브러리 래퍼
  ```
- 예습 포인트: `ArticulationCfg`, `ActuatorCfg`(PD gain), `SceneCfg` 가 어떻게 로봇을 구성하는지.

### 3일차: 강화학습 기초 + PPO + 균형(Standing)
**예습 목표**: PPO 개념 + rsl_rl로 실제 학습 돌려보기.

- RL 개념: MDP(state/action/reward), on-policy vs off-policy, policy/value network, GAE, advantage, PPO clip objective, entropy
- **PPO를 rsl_rl로 실행** (아래 명령 참고)
- 예습 포인트: 학습 로그(reward 곡선), `num_envs`, `max_iterations`, learning rate, clip 등 하이퍼파라미터 위치 파악.

---

## PART 2 — 심화 (4~6일차)

### 4일차: 보행(Walking) RL + Reward 설계/튜닝
**예습 목표**: velocity locomotion 태스크의 보상 항 구조 파악 (강좌 하이라이트).

- 태스크: `Isaac-Velocity-Flat-H1-v0`, `Isaac-Velocity-Rough-G1-v0` 등
- 보상 항 예: linear/angular velocity tracking, feet air time, joint torque/accel 페널티, 자세 유지, action rate
- 코드 위치:
  ```
  /isaac-sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
  ```
- 예습 포인트: Manager-based의 `RewardsCfg` 각 항의 weight를 바꾸면 걸음걸이가 어떻게 변하는지 가설 세워보기.

### 5일차: 센서 기반 RL (Camera, LiDAR, Vision)
**예습 목표**: 관찰(observation)에 센서 데이터를 넣는 구조 이해.

- 센서: RayCaster(LiDAR/height scan), Camera(RGB/Depth), contact sensor
- 개념: proprioceptive vs exteroceptive observation, height-scan을 이용한 rough terrain 보행
- 예습 포인트: `Isaac-Velocity-Rough-*` 태스크가 height scanner를 observation으로 쓰는 부분 확인.

### 6일차: ROS2 연동 + 종합 프로젝트
**예습 목표**: 학습된 policy를 ROS2로 내보내는 파이프라인 개념.

- Isaac Sim ROS2 Bridge (Action Graph / OmniGraph)
- 개념: 학습 policy export(ONNX/TorchScript) → inference, joint state/command 토픽, sim-to-real 고려사항(domain randomization, observation noise)

---

## 실전: rsl_rl로 PPO 돌려보기 (예습 필수 실습)

> 모든 명령은 `/isaac-sim/IsaacLab/` 에서 `./isaaclab.sh -p` (= python.sh 래퍼) 로 실행.

```bash
cd /isaac-sim/IsaacLab

# 등록된 태스크 목록 확인
./isaaclab.sh -p scripts/environments/list_envs.py | grep -iE 'H1|G1|Humanoid'

# 학습 (헤드리스 권장, 환경 수 줄여서 가볍게)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 1024 --max_iterations 300

# 학습 중 영상 저장 옵션
#   --video --video_length 200 --video_interval 2000

# 결과 재생 (학습된 체크포인트 확인, Play 버전 태스크 사용)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-G1-Play-v0 --num_envs 32

# 텐서보드로 reward 곡선 보기
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl
```

주요 CLI: `--task`, `--num_envs`, `--max_iterations`, `--seed`, `--headless`, `--video`, `--distributed`(멀티 GPU)

---

## PPO 라이브러리 선택: rsl_rl vs rl_games

| 기준 | **rsl_rl** ✅ 강좌 추천 | rl_games |
|---|---|---|
| 출신/특화 | ETH RSL, legged_gym 계보 → **다리형/휴머노이드 보행 특화** | 범용 고성능 벤치마크 |
| 알고리즘 | PPO(+distillation) 중심, 단순 | PPO/A2C/SAC 등 범용, 비대칭 actor-critic |
| 코드/설정 | Python cfg, 짧고 읽기 쉬움 → **보상·하이퍼파라미터 학습 최적** | YAML 설정, 추상화 많아 입문 진입장벽 |
| 이 강좌 정합성 | Standing/Walking, Reward shaping, 튜닝에 딱 맞음 | 과함 |

**추천: `rsl_rl`로 예습하세요.**
- 이유 1: H1/G1 등 휴머노이드 velocity 보행 태스크의 **레퍼런스 구현**이 rsl_rl.
- 이유 2: PPO 코드가 짧아 GAE·clip·entropy 등 개념과 실제 코드를 1:1로 대응시켜 보기 좋음.
- 이유 3: 강좌 핵심인 **Reward 설계 + 하이퍼파라미터 튜닝** 실습에 가장 직관적.
- rl_games는 심화(비대칭 critic, vision, 다른 알고리즘, 대규모 성능 실험) 필요 시 선택. 같은 태스크를 `--agent` 엔트리포인트만 바꿔 비교해볼 수 있으니, 예습 여유가 있으면 rsl_rl로 학습한 뒤 rl_games로 한 번 돌려 로그·속도를 비교해보는 것을 추천.

> 참고: skrl(멀티백엔드, 교육용으로도 좋음), Stable-Baselines3(가장 표준적 입문)도 Isaac Lab이 지원하지만 **현재 이 환경엔 미설치**. 강좌는 rsl_rl/rl_games 위주로 보면 됨.

---

## 예습 우선순위 (시간이 부족하다면)

1. **rsl_rl로 G1/H1 보행 학습 1회 완주** (train → play → tensorboard) ← 가장 중요
2. Manager-based locomotion의 **RewardsCfg 구조** 읽고 weight 1~2개 바꿔 재학습
3. USD/PhysX articulation·actuator(PD gain) 개념 정리
4. rough terrain 태스크에서 **height-scan observation** 확인 (5일차 대비)
5. policy export & ROS2 bridge 개념 훑기 (6일차 대비)
