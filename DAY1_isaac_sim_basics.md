# 1일차 심화 — Isaac Sim & 휴머노이드 로봇 기초

> 목표: "휴머노이드 로봇이 시뮬레이터 안에서 **무엇으로 이뤄져 있고**, 그 구조가 **RL의 관찰/행동으로 어떻게 이어지는가**"를 이 환경의 실제 G1 설정으로 이해한다.
> 이 노트는 실측(로컬 Isaac Lab 2.3.2, `G1_MINIMAL_CFG`, `Isaac-Velocity-*-G1-v0`) 기반이다.

---

## 1. 계층 구조: Kit → Omniverse → USD → PhysX

Isaac Sim은 단일 프로그램이 아니라 **NVIDIA Kit** 위에 쌓인 확장(extension) 묶음이다.

```
Isaac Sim  = Kit(앱 프레임워크) + Omniverse(USD 씬/렌더) + PhysX(물리) + Isaac 확장들
Isaac Lab  = 그 위에서 "RL 환경"을 만들기 쉽게 감싼 파이썬 라이브러리
```

- **왜 중요한가**: `./isaac-sim.sh`는 GUI 앱, `./python.sh`(=`isaaclab.sh -p`)는 **헤드리스 파이썬**. RL 학습은 후자로 돌린다. import 순서에 규칙이 있는 이유도 여기 있다(아래 4장).

---

## 2. USD: 씬을 표현하는 "문서"

모든 것(바닥, 조명, 로봇)은 **USD Stage**라는 트리 문서 안의 **Prim**(노드)이다.

- Prim 경로 예: `/World/Robot`, `/World/Robot/pelvis`, `/World/ground`
- Prim에는 **속성(attribute)** 과 **API 스키마**가 붙는다:
  - `UsdPhysics.*` : 질량, 충돌, 조인트 정의 (물리의 "무엇")
  - `PhysxSchema.*` : 솔버 반복수, 마찰 조합, 관성 등 (PhysX의 "어떻게")
- 로봇 USD 안에서 **articulation root**를 가진 Prim이 로봇의 시작점이다.

> Isaac Lab에서 우리는 USD를 손으로 안 짜고 **`ArticulationCfg`(파이썬 설정)** 로 로드/오버라이드한다. 우리가 `STUDY_GUIDE`에서 본 `G1_CFG.spawn.usd_path = ".../G1/g1.usd"`가 바로 이 USD를 가리킨다.

---

## 3. Articulation 해부 (휴머노이드의 핵심)

**Articulation = 링크(rigid body)들이 조인트(joint)로 연결된 트리.** RL에서 다루는 로봇은 전부 이것이다.

### 3.1 구성 요소
| 요소 | 뜻 | G1에서의 예 |
|---|---|---|
| **link (body)** | 강체 세그먼트 | `pelvis`, `torso_link`, `left_knee_link`, `.*_ankle_roll_link` |
| **joint** | 두 링크의 연결 (회전/직선) | `left_hip_pitch_joint`, `.*_knee_joint`, `torso_joint` |
| **DOF** | 자유도 = revolute joint 1개당 1 | 관절 개수 = action 차원 |
| **floating base** | 지면에 안 붙은 루트 | `pelvis` (RL에서 자세·균형의 기준 프레임) |
| **actuator** | 관절 구동 모델(PD) | legs / feet / arms 그룹 |

### 3.2 관절 명명 규칙 = RL 설정의 "주소" (가장 실전적인 포인트)
Isaac Lab의 모든 reward/observation은 관절·링크를 **정규식 이름**으로 지정한다. 강좌에서 계속 만나게 될 패턴:

```python
".*_hip_pitch_joint"   # 좌/우 고관절 pitch 둘 다
".*_knee_joint"        # 좌/우 무릎
".*_ankle_roll_link"   # 좌/우 발목 링크 (발-지면 접촉 판정에 사용)
"torso_joint"          # 몸통 (단수)
```
- `.*` = 좌(`left_`)/우(`right_`) 동시 매칭.
- **이 규칙을 모르면 reward를 못 고친다.** 4일차 튜닝의 전제.

### 3.3 G1은 다리만 있는 게 아니다
`G1_MINIMAL_CFG`의 actuator 그룹(로컬 `unitree.py` 실측):
- `legs` : hip(yaw/roll/pitch) + knee + `torso_joint` — Kp 150~200, Kd 5
- `feet` : ankle(pitch/roll) — Kp 20, Kd 2 (발은 약하게)
- `arms` : shoulder/elbow **+ 손가락 joint**(`.*_one_joint`, `.*_two_joint`, ... `.*_five_joint`) — Kp 40

→ 보행 태스크에서도 팔·손가락 관절까지 전부 articulation에 존재한다. 그래서 보행 reward에 "팔·손가락은 기본 자세에서 벗어나지 마"라는 `joint_deviation_arms/fingers` 페널티가 붙어 있다(4일차에서 재등장).

### 3.4 직접 눈으로 확인하기 (실습)
방금 만든 스크립트로 DOF/링크/기본자세/게인을 출력:
```bash
cd /isaac-sim/IsaacLab
./isaaclab.sh -p /isaac-sim/rl_course_ws/scripts/inspect_g1.py
```
출력에서 확인할 것:
- [ ] `DOF 수` = 이 뒤 나오는 **action 차원**과 같은가?
- [ ] `joint_names` 순서 = observation의 `joint_pos`/`joint_vel` 순서와 동일 (인덱스가 곧 관절)
- [ ] `default_joint_pos` = `G1_CFG`의 `init_state.joint_pos`와 일치 (약간 무릎 굽힌 준비 자세)
- [ ] `feet` 그룹 Kp(≈20)가 `legs`(≈150~200)보다 훨씬 작음 → 발목은 부드럽게

### 3.5 실측 결과 (이 환경에서 직접 출력한 값, RTX 5090)
```
DOF(관절) 수 : 37     body(링크) 수 : 44     actuator 그룹 : 3
legs : 관절  9개, Kp≈177.8, Kd≈ 5.00
feet : 관절  4개, Kp≈ 20.0, Kd≈ 2.00
arms : 관절 24개, Kp≈ 40.0, Kd≈10.00
```
- **DOF 37** = 다리 9(hip3×2+knee×2+torso) + 발 4(ankle 2×2) + 팔·손 24(shoulder3×2+elbow2×2 + 손가락 7×2=14). 즉 **action 벡터도 37차원**.
- **body 44** = 각 관절 링크 + `pelvis`, `torso_link`, `head_link`, `imu_link`, `.*_palm_link` 등 비구동 링크 포함.

**⚠️ 가장 중요한 발견 — 관절 순서는 "설정 그룹 순"이 아니다.**
우리가 `unitree.py`에서 본 순서는 legs→feet→arms였지만, 실제 출력 인덱스는 뒤섞여 있다:
```
[ 0] left_hip_pitch   [ 1] right_hip_pitch   [ 2] torso_joint
[ 3] left_hip_roll    [ 4] right_hip_roll    [ 5] left_shoulder_pitch ...
```
- 이유: articulation의 관절 인덱스는 **키네마틱 트리 순회 순서**(pelvis에서 가까운 관절부터, 좌/우 짝지어)로 PhysX가 정한다. 우리 config의 그룹 순서와 무관.
- **실전 교훈**: action/observation을 **하드코딩된 숫자 인덱스로 다루면 안 된다.** 항상 `joint_names`/정규식(`SceneEntityCfg("robot", joint_names=[...])`)으로 지정한다. Isaac Lab이 이름→인덱스 매핑을 내부적으로 처리해준다. (이게 3.2의 정규식 규칙이 존재하는 근본 이유)

**기본 자세(default_joint_pos) 읽기** — 로봇의 "준비 자세":
- `hip_pitch −0.20, knee +0.42, ankle_pitch −0.23` → 무릎을 살짝 굽힌 **반쯤 앉은 안정 자세** (넘어지기 어려운 시작점).
- `shoulder_pitch +0.35, elbow_pitch +0.87` → 팔을 약간 앞으로 굽힘.
- `one/two_joint = ±1.0/±0.52` → 손가락 프리셋(좌우 부호 반대 = 대칭).
- 이 값이 3일차에서 action offset의 기준점이자, 4일차 `joint_deviation` 페널티의 "돌아가야 할 자세"가 된다.

---

## 4. Isaac Lab 스크립트의 철칙: "App 먼저, import 나중"

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args_cli)      # ← 먼저 시뮬레이터(Kit)를 부팅
simulation_app = app_launcher.app
# 이 줄들 "다음에야" 물리/자산 모듈 import 가능
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
```
- **왜**: Isaac Lab의 물리 모듈들은 Kit 런타임이 살아있어야 로드된다. 순서를 어기면 import 에러. (강좌 내내 만드는 모든 standalone 스크립트가 이 골격을 따른다 — `inspect_g1.py`도 동일.)

---

## 5. Articulation → RL 연결 (1일차와 3일차의 다리 놓기)

여기까지가 1일차의 "구조"이고, 아래가 그 구조가 RL로 이어지는 지점이다. **미리 봐두면 3일차가 쉬워진다.**

이 환경의 `Isaac-Velocity-*-G1-v0` 실측 매핑:

### 5.1 Action = 관절 목표 위치
```python
# velocity_env_cfg.py
joint_pos = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"],   # 모든 관절
    scale=0.5, use_default_offset=True,
)
```
- 정책의 출력(action) = **기본자세 대비 관절 각도 offset** (×0.5). 이걸 PD 컨트롤러가 토크로 바꿔 구동.
- **직접 토크가 아니다** — action_dim = DOF 수. (초보가 가장 헷갈리는 부분)

### 5.2 Observation = 로봇이 매 스텝 "보는" 것
```python
# ObservationsCfg.PolicyCfg (velocity_env_cfg.py)
base_lin_vel        # 몸통 선속도 (3)
base_ang_vel        # 몸통 각속도 (3)
projected_gravity   # 중력벡터를 몸통 좌표로 → "얼마나 기울었나"(자세) (3)
velocity_commands   # 지금 따라야 할 속도 명령 (3)
joint_pos (rel)     # 관절 각도 (DOF)
joint_vel (rel)     # 관절 각속도 (DOF)
last_action         # 직전 action (DOF)
height_scan         # (rough 태스크만) 발밑 지형 높이맵
```
- 대부분 항에 `noise=Unoise(...)`가 붙음 → **일부러 노이즈를 넣어** sim-to-real 강건성 확보(5일차 복선).
- `projected_gravity`가 IMU 없이도 "균형"을 알려주는 핵심 관찰 → **3일차 Standing의 열쇠**.

### 5.3 Scene에 붙는 센서 (5일차 복선)
```python
# MySceneCfg
height_scanner = RayCasterCfg(...)   # torso_link 밑으로 광선 → 지형 스캔
contact_forces = ContactSensorCfg(prim_path=".../Robot/.*", track_air_time=True)  # 발 접촉/공중시간
```
- `contact_forces`가 `feet_air_time`(발을 들어 걷게 하는 보상)과 넘어짐 판정(termination)에 쓰인다.

---

## 6. 1일차 자기점검 (이걸 설명할 수 있으면 통과)

1. USD Prim / articulation root / link / joint / DOF를 각각 한 줄로 설명할 수 있는가?
2. `.*_knee_joint` 같은 정규식이 왜 필요한가? (좌우 동시 지정 + reward 주소)
3. G1의 action은 토크인가 각도인가? action 차원은 무엇과 같은가? → 각도 offset, DOF 수
4. `projected_gravity` 관찰이 왜 균형에 중요한가?
5. Isaac Lab standalone 스크립트에서 왜 `AppLauncher`를 import보다 먼저 호출하는가?
6. `inspect_g1.py`를 돌려 DOF 수·기본자세·발목 게인을 직접 확인했는가?

---

## 다음
- 2일차: 여기서 본 **PD 게인(Kp/Kd)** 과 **sim dt/decimation**을 깊게 → `DAILY_NOTES.md` 2일차
- 3일차: 위 5장의 observation/action이 **PPO**로 학습되는 과정
