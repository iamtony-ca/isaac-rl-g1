# 6일차 심화 — ROS2 연동 & Sim-to-Real 종합

> 목표: 학습한 정책을 **경량 모델로 export**하고 **ROS2로 연결**해 실제 로봇/외부 시스템과 통신하는 파이프라인을 이해하고, 5일간 배운 것을 sim-to-real 관점에서 종합한다.
> 실측 근거: `play.py`의 export 코드, `exporter.py`, 이 환경에 설치된 ROS2 Bridge 확장(`isaacsim.ros2.*`) + ROS2 Jazzy.

---

## 1. 배포 관점의 대전환: 학습 ≠ 추론

**학습에 쓴 거대한 장치의 대부분은 배포에 안 쓰인다.**
| | 학습(training) | 추론/배포(deployment) |
|---|---|---|
| 필요한 것 | actor + **critic** + PPO + GAE + 수천 env + optimizer | **actor(정책망)만** |
| 장치 | rsl_rl, 시뮬 수천 개 | 작은 MLP 하나 |
| 행동 | 확률적 샘플링(탐험) | **결정론적**(평균 μ 사용, noise 0) |

> 배포되는 정책 = **normalizer → actor MLP**. 입력 observation, 출력 action. 그게 전부다. critic·PPO는 버려진다.

---

## 2. Policy Export (play.py가 실제로 하는 일)

`play.py`는 체크포인트를 로드해 **두 포맷으로 export**한다(실측, line 170~173):
```python
export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
export_policy_as_jit(policy_nn, normalizer, path=..., filename="policy.pt")   # TorchScript
export_policy_as_onnx(policy_nn, normalizer, path=..., filename="policy.onnx") # ONNX
```
- **`policy.pt`** (TorchScript/JIT): 파이썬 없이 libtorch(C++)에서 실행 가능.
- **`policy.onnx`** (ONNX): 프레임워크 독립. onnxruntime, TensorRT 등으로 어디서나 추론 → **실제 로봇 온보드 배포에 유리**.

### export된 모델의 정확한 입출력 (exporter.py 구현)
```python
def forward(self, x):
    return self.actor(self.normalizer(x))   # 비순환 정책은 이 한 줄
```
- **입력**: observation 벡터 (Flat G1 = **123차원**, Rough = 310차원)
- **출력**: action 벡터 (**37차원** = 관절 목표각 offset)
- **normalizer**: 학습 중 관찰 정규화를 썼다면 함께 저장(안 썼으면 Identity) → 추론 시 관찰 스케일이 학습과 동일하게 유지됨. **export에 normalizer를 포함하는 게 중요**(안 그러면 배포에서 관찰 분포가 어긋남).
- critic·표준편차(σ) 없음 → 순수 함수 obs→action. RNN 정책이면 hidden state도 함께 내보냄(우리 G1은 MLP라 해당 없음).

### 실습 — 직접 export
```bash
cd /isaac-sim/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-G1-Play-v0 --headless --num_envs 4 \
  --checkpoint logs/rsl_rl/g1_flat/<런>/model_299.pt
# → logs/rsl_rl/g1_flat/<런>/exported/policy.pt , policy.onnx 생성
```

### 실제 export 산출물 (이 환경에서 직접 export)
```
exported/
  policy.onnx   337 KB
  policy.pt     347 KB   (TorchScript)
```
`onnx`로 뜯어본 실제 그래프:
```
INPUTS : [('obs',     [1, 123])]     # observation 123차원
OUTPUTS: [('actions', [1, 37])]      # action 37차원(관절 목표각 offset)
nodes  : 7,  ops = {Gemm, Elu}       # Linear 4개([256,128,128]→37) + ELU 3개
```
- **배포되는 정책은 고작 ~340KB짜리 MLP 하나.** 학습에 쓴 수천 env·critic·PPO 옵티마이저는 흔적도 없다 → obs 123 → action 37의 순수 함수.
- normalizer가 **Identity**로 들어간 이유: 이 태스크 config가 `actor_obs_normalization=False`(3일차)라 관찰 정규화를 안 썼기 때문. 정규화를 켰다면 그 통계도 그래프에 포함된다.
- 이 파일 하나만 있으면 온보드(onnxruntime/TensorRT)에서 실시간 추론 가능 → **이게 sim에서 real로 넘어가는 실체.**

---

## 3. 추론 루프 = 정책을 "돌리는" 법 (개념 코드)

배포 측(시뮬이든 실물이든)에서 매 제어 주기(50Hz)에:
```
매 20ms(control dt)마다:
  1) observation 구성:  base 속도·자세(IMU) + joint pos/vel + 이전 action + [명령]   # 123-D
  2) action = policy(obs)            # ONNX/TorchScript 추론, 37-D
  3) q_target = default_pos + 0.5 * action     # action scale 0.5 (1·2일차)
  4) 관절에 q_target 전송 → 온보드 PD 컨트롤러가 토크로 구동
  반복
```
- **관찰을 정확히 학습 때와 같은 순서·스케일로** 만들어 넣는 게 배포의 핵심 난제(1일차의 "관절 순서" 교훈이 여기서 결정타).

---

## 4. Isaac Sim ROS2 Bridge (이 환경: ROS2 Jazzy 설치됨)

설치 확인된 확장: `isaacsim.ros2.bridge`, `isaacsim.ros2.sim_control`, `isaacsim.ros2.tf_viewer`, `isaacsim.ros2.urdf` (+ `/opt/ros/jazzy`).

### 어떻게 연결되나 — OmniGraph(Action Graph)
Isaac Sim은 코드가 아니라 **노드 그래프(OmniGraph)** 로 ROS2 토픽을 시뮬에 잇는다:
| 방향 | OmniGraph 노드 | ROS2 토픽(예) |
|---|---|---|
| 발행(pub) | `ROS2 Publish JointState` | `/joint_states` (관절 상태) |
| 발행 | `ROS2 Publish Odometry / TF` | `/tf`, `/odom` |
| 발행 | `ROS2 Publish Clock` | `/clock` (시뮬 시간 동기화) |
| 구독(sub) | `ROS2 Subscribe Twist` | `/cmd_vel` (속도 명령) |
| 구독 | `ROS2 Subscribe JointState` | 관절 목표 명령 |

- 전형 구성: 외부에서 `/cmd_vel` 발행 → 시뮬이 구독해 정책의 velocity command로 사용 → 정책이 관절 구동 → 시뮬이 `/joint_states`·`/tf` 발행 → RViz/외부 노드가 소비.
- 확인: `ros2 topic list`, `ros2 topic echo /joint_states`.

### 활성화
- Isaac Sim GUI의 Extensions 창에서 `isaacsim.ros2.bridge` 활성화(이미 설치돼 있음). `ROS_DISTRO=jazzy` 환경에서 동작.

---

## 5. Sim-to-Real 종합 (5일치 개념의 수렴)

학습된 정책이 실제 로봇에서 동작하려면 넘어야 할 **격차(reality gap)** 와 그 대비책:
| 격차 | 어디서 배웠나 | 대비책 |
|---|---|---|
| 물리 파라미터(마찰·질량·CoM) | 2일차 | **domain randomization**(EventCfg) |
| 액추에이터(토크·지연) | 2일차 | effort_limit·PD를 실물에 맞춤, actuator net |
| 센서(이상적 height-scan vs 실물 추정) | 5일차 | 관찰 **noise/clip**, teacher→student |
| 관찰 지연·주파수 | 2·3일차 | control dt(50Hz)를 실물과 일치, latency 랜덤화 |
| 관찰 순서·스케일 불일치 | 1·6일차 | export에 normalizer 포함, 관찰 파이프라인 검증 |

> **핵심**: sim-to-real은 "마지막에 하는 것"이 아니라 **1일차부터 관찰·물리·보상을 설계할 때 이미 시작**된다. 도메인 랜덤화와 관찰 노이즈가 그 씨앗.

---

## 6. 종합 미니 프로젝트 아이디어 (6일차 실습)

1. **원격조종 보행**: 학습한 G1 정책 로드 → ROS2 `/cmd_vel`을 velocity command로 연결 → 키보드/조이스틱 노드로 실시간 조종.
2. **정책 A/B 시연**: 4일차의 baseline vs 튜닝 정책을 각각 export해 같은 명령에서 걸음 비교(RViz 시각화).
3. **ONNX 온보드 추론 흉내**: `policy.onnx`를 onnxruntime로 로드해, 시뮬이 발행한 `/joint_states`로 observation을 구성 → action을 되돌려 보내는 **외부 추론 노드** 작성(실물 배포 구조 그대로).

---

## 7. 6일차 & 전체 자기점검

1. 배포되는 정책에는 무엇이 남고 무엇이 버려지는가? 추론이 결정론적인 이유는?
2. `policy.pt`와 `policy.onnx`의 차이와 각각의 쓰임은?
3. export된 모델의 입력/출력 차원은? (Flat: 123→37) normalizer를 함께 export하는 이유는?
4. 추론 루프 4단계를 쓸 수 있는가? action에서 관절 목표각으로 가는 변환은?
5. OmniGraph로 ROS2를 잇는다는 게 무슨 뜻인가? `/cmd_vel`과 `/joint_states`의 역할은?
6. reality gap 5가지와 각 대비책을 짝지을 수 있는가?
7. "sim-to-real은 1일차부터 시작된다"는 말의 의미는?

---

## 강좌 6일 완주 — 한 장 요약
1일차 **articulation**(DOF 37, 관절=정규식 주소) → 2일차 **물리·PD게인**(control dt=sim.dt×decimation) → 3일차 **PPO/rsl_rl**(obs 123→action 37, on-policy) → 4일차 **reward 설계**(Task/Safety/Style 균형) → 5일차 **센서**(exteroceptive, obs 310) → 6일차 **export·ROS2·sim-to-real**. 모든 단계가 다음 단계의 전제가 된다.
