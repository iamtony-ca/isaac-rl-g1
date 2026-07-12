# 5일차 심화 — 센서 기반 RL (height-scan / LiDAR / Camera)

> 목표: 지금까지의 **proprioceptive(자기 상태) 관찰**에 **exteroceptive(외부 환경) 센서**를 더해 거친 지형·비전 보행으로 확장하는 원리를 이해한다.
> 실측 근거: `MySceneCfg`의 `height_scanner`(RayCasterCfg), `mdp/observations.py`의 `height_scan`, `Isaac-Velocity-Rough-G1-v0` 학습 로그(네트워크 입력 차원 직접 확인).

---

## 1. Proprioceptive vs Exteroceptive (5일차의 핵심 구분)

| 구분 | 뜻 | 예 | 특징 |
|---|---|---|---|
| **Proprioceptive** | 로봇 "자기 상태" | base 속도, `projected_gravity`, joint pos/vel, last action | 저차원, 항상 사용, 학습 쉬움 |
| **Exteroceptive** | "외부 환경" 인지 | **height-scan**, LiDAR, Camera(RGB/Depth) | 고차원, 지형/장애물 인지, 학습·연산 부담↑ |

- **1~4일차의 flat 보행은 전부 proprioceptive만** 썼다(123차원). 평지라서 발밑을 "볼" 필요가 없었다.
- **거친 지형(계단·경사·박스)** 에선 발밑을 못 보면 헛디딘다 → exteroceptive가 필요. 이게 `Isaac-Velocity-Rough-G1-v0`.

---

## 2. Height-scan — 가장 기본적인 exteroceptive 센서 (실측)

### 2.1 센서 정의 (`RayCasterCfg`)
```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",      # G1은 torso_link로 오버라이드
    offset=OffsetCfg(pos=(0.0, 0.0, 20.0)),     # 로봇 20m 위에서
    ray_alignment="yaw",                         # 로봇 heading을 따라 회전
    pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # 격자 광선
    mesh_prim_paths=["/World/ground"],           # 지면 메쉬에 쏨
)
```
- **동작**: 로봇 위 20m에서 아래로 **광선(ray)** 을 격자로 쏴, 지면과 만나는 높이를 읽는다 = "발밑 지형 높이맵".
- **광선 개수(=observation 차원)**: size 1.6×1.0m, resolution 0.1 →
  - x축 = 1.6/0.1 + 1 = **17**, y축 = 1.0/0.1 + 1 = **11** → **17 × 11 = 187개 광선**.
- `ray_alignment="yaw"`: 로봇이 도는 방향으로 격자도 회전 → "내가 가는 앞쪽" 지형을 항상 스캔.

### 2.2 관찰 값 (`height_scan` 함수)
```python
return sensor.data.pos_w[:,2] - sensor.data.ray_hits_w[...,2] - offset   # offset=0.5
# = (센서 높이) − (지면 hit 높이) − 0.5  →  로봇 기준 상대 지형 높이
```
- 평지면 값이 대략 0 근처, 계단/턱이 있으면 +/−로 튐. `clip=(−1,1)`, `noise=±0.1` 부여.

### 2.3 observation 차원: flat vs rough (신경망 입력으로 직접 확인)
| 태스크 | 구성 | 차원 |
|---|---|---:|
| Flat G1 | proprioceptive만 | **123** |
| Rough G1 | proprioceptive 123 **+ height_scan 187** | **310** |

> 실제 rough 학습 로그의 Actor 신경망: `Linear(in_features=310, out_features=512)` → **310 확정**. 센서 하나가 입력을 2.5배로 키운다. 이게 exteroceptive의 대가(연산·학습 부담).

---

## 3. 거친 지형 = 커리큘럼 (rough terrain이 하는 일)

`ROUGH_TERRAINS_CFG`(실측) 구성 지형:
- `pyramid_stairs` / `pyramid_stairs_inv` : 오르막/내리막 계단
- `boxes` (MeshRandomGrid) : 랜덤 높이 박스밭
- `random_rough` (HfRandomUniform) : 자잘한 요철
- `hf_pyramid_slope` / `_inv` : 경사면

- **terrain curriculum**(`max_init_terrain_level=5`, `Curriculum/terrain_levels`): 로봇이 잘하면 **점점 어려운 지형**으로 승급. 처음부터 계단을 주면 학습이 안 되기 때문(2일차 curriculum 개념의 실제 적용).
- height-scan이 이 지형을 "미리 봐서" 발 디딜 곳을 조절 → blind(proprioceptive만) 정책보다 훨씬 유리.

---

## 4. Camera / Vision 기반 RL (심화 방향)

height-scan은 "지형 높이"만 준다. 더 나아가면 **카메라**:

| 센서 | Isaac Lab 클래스 | 용도 | 비용 |
|---|---|---|---|
| Camera | `Camera` | 단일 카메라 RGB/Depth/Semantic | 렌더링 비쌈 |
| **TiledCamera** | `TiledCamera` | **여러 env를 GPU에서 배치 렌더** → RL용 | 상대적으로 효율적(그래도 무거움) |

- vision 정책은 관찰이 이미지(H×W×C) → **CNN 인코더**로 특징 추출 후 MLP 정책. 입력이 수만 차원.
- **현실적 제약**: 카메라는 렌더링이 비싸 `num_envs`를 크게 못 씀(height-scan은 수천 env 가능). 그래서 보행 RL은 보통 height-scan을 쓰고, vision은 조작/내비게이션에서 주로 씀.
- 강좌 5일차는 "센서를 observation에 넣는 구조" 이해가 목표 → height-scan으로 원리를 잡고, 카메라는 "같은 틀에 이미지가 들어갈 뿐"으로 확장.

---

## 5. 센서와 sim-to-real (6일차로 이어지는 다리)

- **height-scan은 "이상적" 센서**다: 시뮬에선 지면 메쉬에 광선을 쏴 정확한 높이를 공짜로 얻지만, **실제 로봇엔 그런 센서가 없다.** 실물은 LiDAR/카메라로 지형을 "추정"해야 하고 오차·지연이 크다.
- 그래서 관찰에 `noise=±0.1`, `clip`을 주고 domain randomization(2일차)을 함께 쓴다 → **시뮬의 완벽한 센서에 정책이 과의존하지 않게**.
- 최신 연구는 "teacher(height-scan으로 학습) → student(카메라/proprioception만으로 모방)" 방식으로 sim-to-real 격차를 줄인다(참고 개념).

---

## 6. 실험: Flat vs Rough (같은 로봇, 센서 유무)

```bash
cd /isaac-sim/IsaacLab
# rough (height-scan 포함, 310차원)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-G1-v0 --headless --num_envs 512 --max_iterations 300
```

### 결과 (이 환경에서 직접 돌림, 각 300 iter · num_envs 512)

버티는 시간(episode length) 성장:
| iter | Flat (123차원, 평지) | Rough (310차원, 계단/경사) |
|---:|---:|---:|
| 0 | 11.25 | 11.00 |
| 100 | 46.68 | 40.24 |
| 200 | 58.76 | 49.00 |
| 299 | **103.68** | **64.68** |

최종(iter 299) 지표:
| 지표 | Flat | Rough | 해석 |
|---|---:|---:|---|
| episode length | **103.7** | 64.7 | 같은 학습량인데 rough가 덜 버팀 = **더 어려움** |
| `track_lin_vel_xy_exp` (+) | 0.0206 | 0.0149 | rough가 명령 추종도 아직 낮음 |
| `Curriculum/terrain_levels` | — | **0.0020** | 지형 커리큘럼이 거의 안 올라감(=아직 쉬운 지형 단계) |
| throughput | 19,451 st/s | 16,080 st/s | 광선 캐스팅 + 지형으로 **~17% 느림** |

### 해석
- **같은 300 iter인데 rough는 65스텝, flat은 104스텝** 버틴다 → 거친 지형 보행이 명백히 더 어렵다. 이유 둘: (a) 계단·경사·박스를 딛어야 하고, (b) **310차원(그중 187이 height-scan)** 을 해석하는 정책을 학습해야 함.
- `terrain_levels=0.002` → 커리큘럼이 거의 초급에 머묾. rough의 정식 학습량은 **3000 iter**(flat 1500의 2배)인데 여기선 10%만 돌린 warm-up이라 당연. 그래도 **eplen이 꾸준히 우상향(11→65)** 하므로 학습 방향은 정상.
- 개념적으로: **height-scan이 없었다면(blind) rough는 훨씬 못 배운다.** 발밑을 "봐야" 계단에 발을 맞출 수 있기 때문 — 이게 exteroceptive 센서를 넣는 이유의 실증.
- (3일차 교훈 재확인) reward 절대값(flat −6.03 vs rough −5.04)으로 우열을 논하지 말 것. 둘은 reward 항 구성·지형이 달라 직접 비교 불가. **같은 태스크 안에서 episode length·track 보상의 추세**로 봐야 함.

---

## 7. 5일차 자기점검

1. proprioceptive와 exteroceptive를 구분하고 각 예를 들 수 있는가?
2. height-scanner는 어떻게 지형을 읽는가? (20m 위에서 격자 광선 → 지면 hit 높이)
3. Rough G1의 observation이 왜 310차원인가? flat(123)과의 차이는 정확히 무엇 몇 개인가?
4. terrain curriculum이 왜 필요한가? (2일차 curriculum 개념 연결)
5. 왜 보행 RL은 카메라보다 height-scan을 즐겨 쓰는가?
6. height-scan이 "sim에선 쉽지만 real에선 어려운" 이유와, 그 격차를 줄이는 방법은?

---

## 다음
- 6일차: 학습한 정책을 **ROS2로 내보내고**(policy export → OmniGraph ROS2 Bridge), sim-to-real 관점에서 종합 미니 프로젝트로 마무리.
