# 4일차 심화 — 보행(Walking) & Reward 설계/튜닝 (강좌 하이라이트)

> 목표: reward를 이루는 각 항이 **무엇을 시키고**, weight를 바꾸면 **걸음이 어떻게 변하는지**를, 실제 함수 구현 + 직접 돌린 비교 실험으로 체득한다.
> 실측 근거: `mdp/rewards.py`, `flat_env_cfg.py` / `rough_env_cfg.py`의 `G1Rewards`, 그리고 이 환경에서 돌린 **weight 변경 A/B 실험**.

---

## 1. Reward는 "여러 항의 가중합" — 세 부류로 나눠 보라

매 스텝 reward = Σ (weight × 항). 보행 태스크의 항들은 목적이 3가지로 갈린다:

| 부류 | 부호 | 목적 | 이 태스크의 항 |
|---|---|---|---|
| **① Task(과제)** | **+** | "명령한 속도로 움직여라" | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `feet_air_time` |
| **② Safety(안전)** | **−(큼)** | "넘어지지 마라" | `termination_penalty` (−200) |
| **③ Style(정규화)** | **−(작음)** | "자연스럽고 효율적으로" | `lin_vel_z`, `ang_vel_xy`, `flat_orientation`, `dof_torques`, `dof_acc`, `action_rate`, `feet_slide`, `joint_deviation_*`, `dof_pos_limits` |

> **reward 설계 = 이 셋의 균형 잡기.** ①만 크면 거칠게라도 빨리 감(비현실적). ③이 과하면 안 움직임(게으름). ②가 약하면 막 넘어짐.

---

## 2. Flat G1의 실제 reward 항 (weight + 함수 뜻)

`Isaac-Velocity-Flat-G1-v0`의 유효 weight(rough 상속 + flat 오버라이드):

| 항 | weight | 함수가 하는 일 (구현 기준) |
|---|---:|---|
| `track_lin_vel_xy_exp` | **+1.0** | `exp(−‖cmd_xy − v_xy‖²/std²)`, std=0.5. 명령 속도에 가까울수록 1에 근접(최대 1). **주 목적** |
| `track_ang_vel_z_exp` | **+1.0** | 위와 같은 exp 커널, yaw rate 추종 |
| `feet_air_time` | **+0.75** | 한 발씩(single stance) 공중시간을 threshold(0.4s)까지 보상 → **제대로 걷게**. 명령≈0이면 0 |
| `termination_penalty` | **−200** | 넘어짐(종료) 시 큰 벌점. 압도적으로 큼 |
| `lin_vel_z_l2` | −0.2 | 수직 튐(위아래 바운스) 억제 |
| `ang_vel_xy_l2` | −0.05 | 몸통 roll/pitch 흔들림 억제 |
| `flat_orientation_l2` | −1.0 | 몸통을 수평으로(중력벡터 xy 성분 벌점) |
| `dof_torques_l2` | −2e-6 | 토크(에너지) 페널티 — hip/knee |
| `dof_acc_l2` | −1e-7 | 관절 가속 페널티(부드러움) |
| `action_rate_l2` | −0.005 | **연속 action 급변** 벌점(떨림 방지) |
| `feet_slide` | −0.1 | 접촉 중 발이 미끄러지면 벌점(질질 끌기 방지) |
| `joint_deviation_hip/arms/fingers/torso` | −0.1/−0.1/−0.05/−0.1 | 다리 외 관절은 **기본자세 유지**(팔·손 휘젓지 마) |
| `dof_pos_limits` | −1.0 | 발목이 각도 한계에 닿으면 벌점 |

### exp 커널을 왜 쓰나 (track_* 이해의 핵심)
```python
return torch.exp(-lin_vel_error / std**2)   # 오차 0 → 1.0, 오차 클수록 0으로
```
- 선형 벌점(−error) 대신 **exp**를 쓰면: 목표 근처에서 기울기가 완만 → "거의 맞았을 때"도 꾸준히 보상 → 미세 추종 학습이 안정적. std가 "관대함" 폭을 정함(작을수록 엄격).

### feet_air_time가 "보행"을 만드는 원리
- `feet_air_time_positive_biped`: **한 발만 땅에 있을 때(single stance)** 그 공중 발의 체공시간을 threshold까지 보상.
- 두 발 다 들면(점프) 보상 없음, 두 발 다 붙어 질질 끌어도 없음 → **번갈아 딛는 걸음**을 유도.
- 명령 속도가 0.1 미만이면 0 → "서 있을 땐 걷지 마"(Standing과 Walking을 자동 구분).

---

## 3. 튜닝 직관 (실험 전에 세우는 가설)

| 이렇게 바꾸면 | 예상 결과 |
|---|---|
| `track_lin_vel_xy` weight ↑ | 명령 추종↑ 하지만 거칠어짐(style 무시) |
| `action_rate`/`dof_acc` 페널티 ↑ (더 음수) | 부드럽지만 **소극적·느려짐**, 극단이면 안 움직임 |
| `feet_air_time` weight ↑ | 보폭·체공↑ (성큼성큼), 과하면 불안정 |
| `termination_penalty` ↓(0에 가깝게) | 넘어짐을 덜 무서워함 → 위험한 동작↑ |
| `flat_orientation` 페널티 ↑ | 몸통 꼿꼿, 과하면 걸음이 뻣뻣 |
| `joint_deviation_arms` ↓(0) | 팔을 자유롭게 휘저음(균형엔 도움될 수도) |

> **황금률**: 한 번에 한 항만, 조금씩. reward 항들은 상호작용해서 두 개를 동시에 크게 바꾸면 원인 분석이 불가능해진다.

---

## 4. 직접 실험: weight만 바꿔 재학습 (hydra CLI override)

train.py는 **hydra override**를 지원 → 코드 수정 없이 CLI로 reward weight를 바꿀 수 있다:
```bash
cd /isaac-sim/IsaacLab
# baseline
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 512 --max_iterations 300
# action_rate 페널티를 100배로 (부드러움 강제)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless --num_envs 512 --max_iterations 300 \
  env.rewards.action_rate_l2.weight=-0.5
```
- 문법: `env.rewards.<항이름>.weight=<값>` (점 경로로 config 트리에 직접 주입).
- 다른 예: `env.rewards.feet_air_time.weight=1.5`, `env.commands.base_velocity.ranges.lin_vel_x='[0.0,1.5]'`.

### 실험 결과 (이 환경에서 직접 돌림, 각 300 iter · num_envs 512)

**A = baseline / B = `action_rate_l2.weight` −0.005 → −0.5 (100배)**

에피소드 길이(=버티는 스텝, 3일차 교훈대로 이걸 본다):
| iter | A baseline eplen | B heavy-action_rate eplen |
|---:|---:|---:|
| 0 | 11.25 | 11.25 |
| 100 | 46.68 | 23.78 |
| 200 | 58.76 | 19.33 |
| 299 | **103.68** (≈2.1s) | **18.03** (≈0.36s) |

최종(iter 299) 항별 기여 & 행동 지표:
| 지표 | A baseline | B heavy | 해석 |
|---|---:|---:|---|
| `track_lin_vel_xy_exp` (+) | **0.0206** | 0.0030 | A가 명령 속도로 훨씬 잘 감 |
| `feet_air_time` (+) | 0.0014 | **0.0000** | **B는 발을 아예 안 뗌 = 안 걸음** |
| `action_rate_l2` (−) | −0.041 | −0.128 | B는 움직임 자체를 벌 받아 억제됨 |
| `base_contact` 종료율 | 1.0 | 1.0 | 둘 다 아직 결국 넘어짐(300 iter은 짧음) |

### 해석 — "과한 Style 페널티 = 얼어붙은 로봇"
- **A(baseline)**: 버티는 시간이 11 → 104스텝으로 꾸준히 늘고, 발을 떼며(feet_air_time>0) 명령 속도로 이동(track 0.02). **걷기를 배워가는 중.**
- **B(action_rate 100배)**: eplen이 ~18에서 **오히려 정체·감소**(iter100 23.78 → iter299 18.03). `feet_air_time`이 정확히 **0** — 발을 한 번도 안 뗀다. action을 바꾸면 큰 벌을 받으니 **정책이 "안 움직이는 게 이득"** 이라 판단해 얼어붙음 → 걸음 자체를 학습 못 함. **예측했던 "과한 부드러움 강제 → 게으른/굳은 로봇"이 실제로 재현됨.**
- ⚠️ **함정 지표**: B의 `error_vel_xy`(0.038)가 A(0.199)보다 오히려 **낮게** 찍힘. 얼핏 "B가 더 잘 추종?" 같지만, `feet_air_time=0` + track 보상 6배 낮음 + eplen 1/6이 진실을 말한다 → B는 그냥 안 움직여서 명령≈0 구간만 맞춘 것. **한 지표로 판단하지 말라(3일차 교훈)의 완벽한 예.**

> 결론: **reward weight 하나(action_rate)를 100배 키웠을 뿐인데 "걷는 로봇"이 "얼어붙은 로봇"이 됐다.** 이게 reward shaping이 강력하면서도 위험한 이유다. Style 페널티는 Task를 방해하지 않는 선에서 "작게" 유지해야 한다.

---

## 5. Reward 디버깅 워크플로 (실전에서 계속 쓰게 됨)

1. 학습을 돌리고 tensorboard에서 **`Episode_Reward/<각 항>`** 을 개별로 본다.
2. 어떤 항이 **비정상적으로 지배적**인가? (예: termination_penalty가 −0.2로 계속 크면 → 아직 잘 넘어짐)
3. 원하는 행동이 안 나오면: 그 행동을 유도하는 항 weight를 ↑, 방해하는 항을 ↓.
4. `Metrics/base_velocity/error_vel_xy`, `Episode_Termination/base_contact` 같은 **행동 지표**로 검증(3일차 교훈: reward 총합만 보지 말 것).
5. 한 항만 바꿔 재학습 → 반복.

---

## 6. 4일차 자기점검

1. reward 항을 Task/Safety/Style 세 부류로 분류하고 각 예를 들 수 있는가?
2. `track_lin_vel_xy_exp`가 선형이 아니라 **exp 커널**인 이유는?
3. `feet_air_time_positive_biped`는 어떻게 "번갈아 걷기"를 유도하는가? 명령이 0이면 왜 0인가?
4. `action_rate` 페널티를 크게 하면 걸음이 어떻게 변하나? (실험으로 확인했는가)
5. 왜 "한 번에 한 항만, 조금씩" 바꿔야 하는가?
6. 원하는 행동이 안 나올 때 reward를 어떻게 진단·수정하나?
7. hydra override로 `feet_air_time` weight를 2배로 주는 명령을 쓸 수 있는가?

---

## 다음
- 5일차: 지금까지 proprioceptive 관찰만 썼다. **height-scan/카메라(exteroceptive)** 를 observation에 넣어 거친 지형·비전 기반 보행으로 확장.

---

## 부록 A — Reward weight가 실제 코드 어디에 있나 (상속 3단계)

DAY4에서 소개한 weight들은 **한 파일이 아니라 상속 체인에 흩어져** 있다. "이 항의 실제 weight가 몇이냐"는 **가장 아래에서 마지막으로 덮어쓴 곳**을 봐야 한다.

> 경로 약칭: `velocity/` = `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/`

### 상속 3단계 (아래로 갈수록 우선)
```
[1] 베이스 기본값        velocity/velocity_env_cfg.py:231        class RewardsCfg
        ↓ 상속
[2] G1 rough 오버라이드   velocity/config/g1/rough_env_cfg.py:20  class G1Rewards(RewardsCfg)
        ↓ 상속
[3] G1 flat 오버라이드    velocity/config/g1/flat_env_cfg.py:14   G1FlatEnvCfg.__post_init__
```
`Isaac-Velocity-Flat-G1-v0` 학습 시 이 셋이 순서대로 적용되고, **가장 아래에서 건드린 값이 최종값**.

### weight를 세팅하는 두 문법 (섞여 있어 헷갈림)
```python
# ① 항 전체 재정의 — rough_env_cfg.py:29
track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=2.0, ...)
# ② __post_init__에서 weight만 수정 — flat_env_cfg.py:30
self.rewards.action_rate_l2.weight = -0.005
```

### Flat G1 각 항의 "최종 weight가 어디서 왔나"
| Reward 항 | 최종 weight | 고칠 위치 |
|---|---:|---|
| `track_lin_vel_xy_exp` | +1.0 | `rough_env_cfg.py:24` |
| `track_ang_vel_z_exp` | +1.0 | `flat_env_cfg.py:28` (rough 2.0을 덮어씀) |
| `feet_air_time` | +0.75 | `flat_env_cfg.py:32` |
| `termination_penalty` | −200 | `rough_env_cfg.py:23` |
| `action_rate_l2` | −0.005 | `flat_env_cfg.py:30` |
| `dof_acc_l2` | −1.0e-7 | `flat_env_cfg.py:31` |
| `dof_torques_l2` | −2.0e-6 | `flat_env_cfg.py:34` |
| `flat_orientation_l2` | −1.0 | `rough_env_cfg.py:135` |
| `lin_vel_z_l2` | −0.2 | `flat_env_cfg.py:29` |
| `feet_slide` | −0.1 | `rough_env_cfg.py:41` |
| `joint_deviation_hip/arms/fingers/torso` | −0.1/−0.1/−0.05/−0.1 | `rough_env_cfg.py:57,62,78,96` |
| `dof_pos_limits` | −1.0 | `rough_env_cfg.py:51` |

> 규칙: **flat → rough → base 순으로 내려가며 그 항을 마지막으로 건드린 곳**이 최종값.

### weight(숫자) vs 함수(계산)는 다른 파일
- **weight** = 위 3개 config → "각 항을 얼마나 중요하게 볼까"
- **함수** = `velocity/mdp/rewards.py`(locomotion 전용) + `source/isaaclab/isaaclab/envs/mdp/rewards.py`(공통) → "무엇을 계산하나"

### 4번째 층 — 런타임 hydra override (파일 안 고침)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-G1-v0 --headless env.rewards.feet_air_time.weight=1.5
```
빠른 실험은 hydra(재현 명령이 남음), 영구 변경은 위 표의 파일:줄 직접 수정.
