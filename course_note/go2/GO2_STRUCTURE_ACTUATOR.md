# Go2 구조 + DC-Motor 액추에이터 (DAY1·DAY2 델타)

> 공유 개념(USD·articulation·PhysX 시간구조·PD게인 의미)은 [[DAY1_isaac_sim_basics]]·[[DAY2_physics_actuators]] 그대로.
> 여기선 **Go2에서 실제로 다른 것**: 12-DOF 4족 트리 + DC-Motor 모델.
> 값 출처(실측): `inspect_go2.py` 출력(`logs/go2_inspect.log`) + `unitree.py`의 `UNITREE_GO2_CFG`.

---

## 1. Articulation 실측 (12-DOF 4족)

`inspect_go2.py` 결과:
```
DOF(관절) 수   : 12
body(링크) 수  : 19
actuator 그룹  : 1  (base_legs)
articulation root : base
```

### 관절 이름·인덱스 (action / joint-observation 순서와 동일)
```
[ 0] FL_hip_joint    [ 1] FR_hip_joint    [ 2] RL_hip_joint    [ 3] RR_hip_joint
[ 4] FL_thigh_joint  [ 5] FR_thigh_joint  [ 6] RL_thigh_joint  [ 7] RR_thigh_joint
[ 8] FL_calf_joint   [ 9] FR_calf_joint   [10] RL_calf_joint   [11] RR_calf_joint
```
- **다리 = hip(옆으로 벌림) → thigh(허벅지) → calf(정강이)** 3관절 × 4다리 = 12.
- 명명 규칙: `F/R`(front/rear) + `L/R`(left/right) + 관절. reward·센서가 정규식 `.*_foot`, `.*_thigh_joint`, `F[L,R]_thigh_joint` 등으로 참조.
- ⚠️ **인덱스는 "관절 레벨별"로 묶인다**(hip 4개 → thigh 4개 → calf 4개), 다리별(FL의 3관절 연속)이 **아니다**. 이게 [[CUSTOM_ROBOT]]이 경고한 "인덱스는 키네마틱 트리 순" 함정의 실례 → **항상 이름/정규식으로 참조**할 것.

### 링크(body) 19개
```
base  (= floating base, root)
{FL,FR,RL,RR}_hip / _thigh / _calf / _foot   (4다리 × 4링크 = 16)
Head_upper, Head_lower                        (머리 2, 물리 있음)
```
- **발 = `*_foot`** (contact sensor·feet_air_time이 참조). G1의 `*_ankle_roll_link`에 대응.
- height-scan은 이 중 **`base`** 에 부착된다(G1은 `torso_link`).

### 기본 자세 default_joint_pos (안정 시작 = 웅크린 준비자세)
| 관절 | 값(rad) | 의미 |
|---|---|---|
| `.*L_hip_joint` | +0.10 | 왼다리 살짝 벌림 |
| `.*R_hip_joint` | −0.10 | 오른다리 살짝 벌림(대칭) |
| `F[L,R]_thigh_joint` | +0.80 | 앞다리 허벅지 |
| `R[L,R]_thigh_joint` | +1.00 | 뒷다리 허벅지(더 굽힘) |
| `.*_calf_joint` | −1.50 | 정강이 접음 |
- **action=0이면 이 자세 유지**(`use_default_offset` 개념). 정책은 "이 웅크린 자세에서 얼마나 벗어날까"만 학습 → 학습 쉬움([[DAY1_isaac_sim_basics]] 5절과 동일 원리).
- 앞/뒤 thigh 각도가 다른 건(0.8 vs 1.0) 4족 특유의 **앞낮뒤높** 균형 자세.

---

## 2. DC-Motor 액추에이터 (Go2의 핵심 차이)

G1은 `ImplicitActuatorCfg`(순수 PD)였지만 Go2는 **`DCMotorCfg`**:
```python
"base_legs": DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],  # 12개 전부 한 그룹
    effort_limit=23.5,        # 최대 토크 (N·m) — G1 다리 ~300의 1/13
    saturation_effort=23.5,   # 이 토크에서 속도-토크 곡선이 포화
    velocity_limit=30.0,      # 최대 각속도 (rad/s)
    stiffness=25.0,           # Kp (G1은 100~200)
    damping=0.5,              # Kd (G1은 2~5)
    friction=0.0,
)
```

### DC-Motor vs Implicit PD — 무엇이 다른가
| | Implicit PD (G1) | **DC-Motor (Go2)** |
|---|---|---|
| 토크 계산 | `τ = Kp·(q*−q) − Kd·q̇`, effort_limit로 **단순 클립** | 위 + **속도-토크 곡선**으로 제한 |
| 속도 영향 | 없음(한계 토크가 속도와 무관) | **속도↑ → 낼 수 있는 토크↓** (실제 모터 특성) |
| 실물 근접도 | 근사 | **더 정확** |
| sim-to-real | 액추에이터 gap 큼 | **gap 작음**(모터 물리 반영) |

- **왜 중요한가**: 실제 BLDC 모터는 빨리 돌수록 역기전력 때문에 토크가 준다. DC-Motor 모델은 `saturation_effort`·`velocity_limit`로 이 곡선을 흉내 → **정책이 "실물 모터로 낼 수 없는 토크"에 의존하지 않게** 학습된다 = sim-to-real의 액추에이터 gap([[DAY6_ros2_sim2real]] 표)을 학습 단계에서 미리 메움.
- **낮은 Kp(25)**: Go2는 가볍고(≈15kg) 관절 토크한계도 작아(23.5N·m) 게인이 G1보다 훨씬 낮다. Kp가 낮으면 "물렁"하지만 4족은 4점 지지라 그걸로 충분하고 오히려 부드럽다.

### 관절 한계·속도 (inspect / 내부 로그 실측)
| 관절군 | 위치 한계(rad) | 속도 한계(rad/s) |
|---|---|---|
| thigh(앞) | [−1.571, 3.491] | 30.1 |
| thigh(뒤) | [−0.524, 4.538] | 30.1 |
| calf | [−2.723, −0.838] | 15.7 |
> calf는 속도 한계가 절반(15.7) — 무릎이 더 느리게 설계됨.

---

## 3. 시간 구조 / PhysX (G1과 동일 개념)
- sim dt, decimation, control dt = sim dt × decimation, solver iteration 개념은 [[DAY2_physics_actuators]] 그대로.
- Go2 articulation: `solver_position_iteration_count=4`, `self_collisions=False`(성능).
- 이 부분은 **로봇과 무관하게 공유** → 새로 볼 것 없음. 게인 감각만 위 2절로 갱신.

---

## 4. 자기점검 (Go2 델타)
- [ ] Go2가 왜 12-DOF인지(3관절×4다리) + 관절 인덱스가 레벨별로 묶이는 이유를 말할 수 있다.
- [ ] `DCMotorCfg`가 Implicit PD와 뭐가 다른지(속도-토크 곡선), 왜 sim-to-real에 유리한지 설명할 수 있다.
- [ ] height-scan이 `base`에, 발 접촉이 `*_foot`에 붙는다는 걸 안다.
- [ ] Go2 Kp(25)가 G1(100~200)보다 낮은 이유(가볍고 토크한계 작음)를 안다.

→ 다음: reward 델타 [[GO2_REWARD]]
