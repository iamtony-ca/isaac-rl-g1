# Custom 로봇 온보딩 체크리스트 (Isaac Lab + rsl_rl)

> 강좌는 G1이 이미 config가 다 짜여 있어 "weight만" 만졌다. 이 노트는 **내 로봇을 붙일 때** 무엇을 새로 쓰고
> 무엇을 재사용하는지 단계별 체크리스트. (강좌 범위 밖 · 실전용)
> 핵심 원칙: **밑바닥부터가 아니라 "가장 비슷한 기존 로봇 디렉토리를 복사 → 이름·경로·관절참조만 치환".**
> 관련: [[DAY1_isaac_sim_basics]](articulation), [[DAY2_physics_actuators]](PD게인), [[DAY4_reward_shaping]](reward 위치).

---

## 큰 그림 — 재사용 vs 새로 작성

| 재사용 (안 건드림) | 새로 작성 (내 로봇용) |
|---|---|
| reward 함수(`mdp/rewards.py`), observation/action manager | ① 로봇 **asset config**(ArticulationCfg) |
| PPO(rsl_rl), terrain generator, height-scan 센서 | ② **태스크 config**(상속 후 오버라이드) |
| 환경 베이스 `LocomotionVelocityRoughEnvCfg` | ③ **에이전트 config**(PPO 하이퍼파라미터) |
| | ④ **gym 등록**(`__init__.py`) |

> 비슷한 다리로봇 + 같은 태스크(속도 보행)면 프레임워크 90%+ 재사용, 위 4개 config만 작성.

---

## STEP 0 — 사전 준비
- [ ] 로봇 **URDF(또는 MJCF) + 메쉬(STL/OBJ/DAE)** 확보
- [ ] URDF 열어 **관절 이름 / 링크 이름 / DOF 수** 목록화 (reward·센서가 이 이름을 참조하므로 필수)
- [ ] 각 관절의 **PD 게인 감각**(무게 받치는 관절 vs 발/손)과 **토크 한계(effort limit)** 파악
- [ ] 어떤 태스크인지 확정: 속도 보행? 4족? 조작? → 복사할 템플릿이 정해짐

---

## STEP 1 — URDF → USD 변환
Isaac Lab의 변환 스크립트 사용 (articulation은 base를 고정하지 않으므로 `--fix-base` **없이**):
```bash
cd /isaac-sim/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /path/to/my_robot.urdf  /path/to/my_robot.usd
# (MJCF면 scripts/tools/convert_mjcf.py)
```
- [ ] 생성된 `my_robot.usd`를 GUI(`./isaac-sim.sh`)에서 열어 **관절이 제대로 물리고 충돌 메쉬가 맞는지** 확인
- [ ] Stage 트리에서 **articulation root**와 관절/링크 이름이 URDF와 일치하는지 확인

---

## STEP 2 — 로봇 asset config 작성 (ArticulationCfg) ★핵심
템플릿: `source/isaaclab_assets/isaaclab_assets/robots/unitree.py`의 `G1_CFG`. 복사해 내 로봇용으로:
- [ ] `spawn.usd_path` → 내 `my_robot.usd` 경로
- [ ] `init_state.pos` / `init_state.joint_pos` → **안정된 시작 자세**(예: 무릎 살짝 굽힘 — DAY1의 G1 준비자세 참고)
- [ ] `actuators={...}` 그룹 정의 — 관절을 **정규식으로 묶고** 그룹별 `stiffness(Kp)`/`damping(Kd)`/`effort_limit` 지정
  - 다리(강하게) / 발(부드럽게) / 팔(감쇠 크게) 식 배분 → [[DAY2_physics_actuators]]
  - `ImplicitActuatorCfg` 사용(입문 기본, 안정적)
- [ ] `soft_joint_pos_limit_factor`, `articulation_props`(solver iteration, self_collision) 설정
- [ ] `activate_contact_sensors=True` (발 접촉 reward를 쓸 거면 필수)

---

## STEP 3 — 검사 (asset가 제대로 로드되는지)
`scripts/inspect_g1.py`(우리가 만든 것)를 복사해 `G1_MINIMAL_CFG` → `MY_ROBOT_CFG`로 바꿔 실행:
```bash
./isaaclab.sh -p /isaac-sim/rl_course_ws/scripts/inspect_my_robot.py
```
- [ ] **DOF 수 / 관절 이름 / 링크 이름 / 기본자세 / 그룹별 게인**이 의도대로 찍히는지 확인
- [ ] ⚠️ 관절 **인덱스 순서**는 키네마틱 트리 순(설정 그룹 순 아님) → 이름/정규식으로만 참조할 것 (DAY1 함정)

---

## STEP 4 — 태스크 config 작성 (상속 + 오버라이드)
템플릿: `.../locomotion/velocity/config/g1/` 디렉토리를 통째로 복사 → `config/my_robot/`.
`rough_env_cfg.py`에서:
- [ ] `from isaaclab_assets import MY_ROBOT_CFG` 로 교체, `self.scene.robot = MY_ROBOT_CFG.replace(...)`
- [ ] **센서 부착 링크** 이름 교체 — `self.scene.height_scanner.prim_path = ".../Robot/<몸통링크>"` (G1은 `torso_link`; 내 로봇 이름으로)
- [ ] **종료 조건** 링크 — `self.terminations.base_contact...body_names = "<몸통링크>"`
- [ ] **reward의 관절/링크 이름 전부 교체** — `feet_air_time`의 `.*_ankle_roll_link`, `joint_deviation_*`의 관절 정규식 등을 내 로봇 이름으로 (DAY4 부록의 reward 위치 표 참고)
- [ ] **내 로봇에 없는 항 제거** (예: 손가락 없으면 `joint_deviation_fingers` 삭제)
- [ ] `commands.base_velocity.ranges` (전/횡/회전 속도 명령 범위) 조정
- [ ] `flat_env_cfg.py`도 동일하게 관절/링크 이름·weight 정리
- [ ] ⚠️ **2족 전용 함수 주의**: `feet_air_time_positive_biped`는 2족용. 4족이면 4족용 함수로 교체

---

## STEP 5 — 에이전트 config (PPO 하이퍼파라미터)
템플릿: `config/g1/agents/rsl_rl_ppo_cfg.py` 복사.
- [ ] `experiment_name` → 내 로봇 이름 (로그 폴더가 이걸로 생김)
- [ ] `actor/critic_hidden_dims`, `max_iterations` 등 조정(보통 그대로 시작해도 됨)
- [ ] 대부분 그대로 두고 나중에 튜닝 (rsl_rl은 lr을 adaptive로 자동 조절 → [[DAY3_ppo_rsl_rl]])

---

## STEP 6 — gym 등록
`config/my_robot/__init__.py`에서 (G1의 `__init__.py`가 템플릿):
- [ ] `gym.register(id="Isaac-Velocity-Flat-<Robot>-v0", ...)` — `env_cfg_entry_point` → 내 cfg 클래스, `rsl_rl_cfg_entry_point` → 내 runner cfg
- [ ] Flat/Rough × 일반/Play 총 4개 등록 (G1과 동일 패턴)
- [ ] 상위 패키지가 이 디렉토리를 import하도록 연결(자체 extension이면 `extsUser/`에 두고 설치)
- [ ] 확인: `./isaaclab.sh -p scripts/environments/list_envs.py | grep <Robot>`

---

## STEP 7 — 스모크 테스트 (작게 먼저)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-<Robot>-v0 --headless --num_envs 512 --max_iterations 100
```
- [ ] **에러 없이 도는지**(관절/링크 이름 오타는 여기서 터짐)
- [ ] TensorBoard로 `Train/mean_episode_length`가 우상향하는지 → [[TENSORBOARD]]
- [ ] 안 되면 관찰: 로봇이 즉시 폭발/관통? → 게인·solver iteration·초기자세 재점검

---

## STEP 8 — 튜닝 루프
- [ ] reward weight 조정(파일 직접 또는 hydra override) → [[DAY4_reward_shaping]] 부록
- [ ] `num_envs` 크게(4096) + `max_iterations` 정식(1500~3000)으로 본 학습
- [ ] `play.py`로 재생·정책 export(ONNX/TorchScript) → [[DAY6_ros2_sim2real]]

---

## 흔한 함정 체크
- [ ] 관절/링크 **이름 불일치** → reward·센서가 조용히 빈 텐서를 잡거나 에러. STEP3 검사로 선제 차단.
- [ ] **게인 부적절** → 로봇이 부들부들/폭발. effort_limit·Kp를 실물 스펙 근처로.
- [ ] **self-collision** 켜서 초기부터 무겁거나, 꺼서 팔이 몸통 통과 → 상황에 맞게.
- [ ] **접촉 버퍼 오버플로**(대규모 env) → `sim.physx.gpu_max_rigid_patch_count` ↑ (DAY2).
- [ ] 내 로봇 형태에 **안 맞는 reward** 방치 → 이상한 걸음. 형태에 맞는 항만 남길 것.

---

## 태스크가 많이 다르면 (참고)
| 상황 | 추가로 할 일 |
|---|---|
| 4족 | 4족용 gait reward, 접촉 스케줄 함수로 교체 |
| 조작/로코매니퓰레이션 | custom observation(물체 상태)·reward 함수 작성. G1 PickPlace 태스크 참고 |
| 바퀴·비정형 동역학 | Manager-based가 안 맞으면 **Direct workflow**로 env 클래스 직접 작성 |

> 요약: **다리로봇+속도보행이면 "config 4종 복사·치환"으로 끝**, 태스크가 근본적으로 다를수록 custom 함수/Direct env로 무게중심이 옮겨간다.
