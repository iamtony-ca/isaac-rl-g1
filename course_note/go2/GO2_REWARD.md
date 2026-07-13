# Go2 Reward 델타 — 형태가 reward를 결정한다 (DAY4 델타)

> reward 설계 원칙(세 부류 균형·dense·exp커널·황금률)은 [[DAY4_reward_shaping]] 그대로.
> 여기선 **Go2(4족)에서 실제로 다른 reward 항**과 그 이유.
> 값 출처(실측): base `velocity_env_cfg.py`의 `RewardsCfg` → `go2/rough_env_cfg.py` → `go2/flat_env_cfg.py` 오버라이드 3단계.

---

## 1. 큰 그림 — "팔·손이 없으면 그걸 규제하던 항도 없다"

reward는 **로봇 형태를 따라간다.** G1은 팔·손·몸통·2족균형 때문에 규제 항이 많았지만, Go2는 그게 없어 **더 단순**하다.

| G1(biped)에 있던 항 | Go2(quadruped) | 왜 |
|---|---|---|
| `feet_air_time_positive_biped` | → `feet_air_time`(표준) | 2족 전용 gait 함수 대신 4족 표준 |
| `feet_slide` (발 미끄럼 벌점) | ❌ 없음 | 2족은 미끄러지면 곧 넘어짐 → 필요, 4족은 덜 치명적 |
| `joint_deviation_arms` | ❌ 없음 | **팔이 없다** |
| `joint_deviation_fingers` | ❌ 없음 | **손가락이 없다** |
| `joint_deviation_torso` | ❌ 없음 | **허리 관절이 없다** |
| `joint_deviation_hip` | ❌ 없음(별도 항 아님) | 4족 hip은 gait에 필수라 규제 안 함 |
| — | **`undesired_contacts`(`.*THIGH`)** | 4족: 허벅지가 땅에 끌리면 벌점 |

> 결론: **Go2 reward = "속도추종 + 4족 gait(air_time) + 부드러움 페널티 + 수평유지"** 로 G1보다 항이 적고 직관적이다.

---

## 2. Go2 실제 reward 구성 (3단계 상속 실측)

### (A) base `RewardsCfg` — 모든 velocity 태스크 공통 (G1과 공유)
| 항 | 함수 | weight | 부류 |
|---|---|---|---|
| `track_lin_vel_xy_exp` | exp 커널 | +1.0 | Task(주목적) |
| `track_ang_vel_z_exp` | exp 커널 | +0.5 | Task |
| `lin_vel_z_l2` | 수직속도 | −2.0 | Style(튐 억제) |
| `ang_vel_xy_l2` | roll/pitch 각속도 | −0.05 | Style(흔들림) |
| `dof_torques_l2` | 토크² | −1.0e-5 | Style(에너지) |
| `dof_acc_l2` | 관절가속² | −2.5e-7 | Style(부드러움) |
| `action_rate_l2` | action 급변 | −0.01 | Style(떨림) |
| `feet_air_time` | **4족 표준** `.*FOOT` | +0.125 | Task(gait) |
| `undesired_contacts` | `.*THIGH` 접촉 | −1.0 | Safety |
| `flat_orientation_l2` | 몸통 기울기 | 0.0(기본) | Style |

### (B) `go2/rough_env_cfg.py` 오버라이드 (거친지형)
```python
self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"  # 발 이름 확정
self.rewards.feet_air_time.weight = 0.01        # air_time 비중 낮춤
self.rewards.undesired_contacts = None          # rough에선 끔
self.rewards.dof_torques_l2.weight = -0.0002    # 토크 페널티 ↑ (거친지형 에너지)
self.rewards.track_lin_vel_xy_exp.weight = 1.5  # 속도추종 강화
self.rewards.track_ang_vel_z_exp.weight = 0.75
self.rewards.dof_acc_l2.weight = -2.5e-7
```

### (C) `go2/flat_env_cfg.py` 오버라이드 (평지)
```python
self.rewards.flat_orientation_l2.weight = -2.5  # 평지: 몸통 수평 강하게
self.rewards.feet_air_time.weight = 0.25        # 평지: 걸음 더 또렷하게
# + terrain=plane, height_scan=None (평지는 지형인지 불필요)
```
> **flat vs rough의 reward 차이**: 평지는 `flat_orientation`(수평)·`feet_air_time`(또렷한 걸음)을 키우고, 거친지형은 속도추종·토크효율을 키운다.

---

## 3. 튜닝 직관 (Go2용 가설)
- `feet_air_time` weight ↑ → 발을 더 오래 들어 **또박또박** 걸음(질질 끌기 방지). Go2 평지는 0.25로 꽤 크다.
- `flat_orientation_l2`(−2.5, 평지) ↑ → 몸통을 수평 유지. 너무 크면 지형 적응이 뻣뻣해짐.
- `undesired_contacts`(`.*THIGH`) → 허벅지 끌기/주저앉기 방지. rough에선 지형상 접촉이 잦아 아예 끔(None).
- **4족 gait 특성**: 4족은 trot(대각선 다리 쌍)이 자연스럽게 창발한다 — air_time reward가 이를 유도. biped처럼 좌우 교대만이 아니라 **대각 리듬**이 나오는지 play에서 관찰.

> reward 디버깅 워크플로(TensorBoard `Episode_Reward/<항>` 항별 진단·한번에 한 항)는 [[DAY4_reward_shaping]] 5절 그대로.

---

## 4. 자기점검 (Go2 델타)
- [ ] G1의 `joint_deviation_arms/fingers/torso`·`feet_slide`가 Go2에서 왜 사라졌는지 말할 수 있다.
- [ ] Go2가 쓰는 `feet_air_time`(표준)과 G1의 `feet_air_time_positive_biped`(2족) 차이를 안다.
- [ ] flat vs rough에서 어떤 weight가 왜 달라지는지(수평·air_time vs 속도·토크) 설명할 수 있다.
- [ ] `undesired_contacts`가 `.*THIGH`를 보는 이유(4족 허벅지 끌기 방지)를 안다.

→ 다음: 직접 돌린 학습/튜닝 결과 [[GO2_EXPERIMENTS]]
