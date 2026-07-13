# Go2 (4족) 델타 노트 — 읽는 법

> 기존 `course_note/`의 노트들은 **Unitree G1(휴머노이드) 기준**으로 쓰였다.
> 이 `go2/` 폴더는 같은 강좌 파이프라인을 **Unitree Go2(4족 보행)** 로 다시 진행하며,
> **G1과 달라지는 부분(=델타)만** 정리한 노트 세트다.
> **PPO·GAE·clip·sim-to-real 같은 공유 이론은 다시 쓰지 않고 기존 G1 노트로 링크**한다.

## 왜 "델타"인가
로봇이 바뀌어도 **프레임워크의 90%는 그대로**다 (Isaac Lab, rsl_rl PPO, manager 구조, terrain·센서·export 파이프라인).
바뀌는 건 **로봇 형태에서 오는 것들**뿐:
- 형태(morphology): 4다리 12-DOF vs 2다리+팔+손 다-DOF
- 액추에이터 모델: **DC-Motor(Go2)** vs Implicit PD(G1)
- reward 항: **quadruped gait** vs biped 균형 항
- action scale·terrain 스케일·태스크 ID

그래서 공유 개념은 [[STUDY_GUIDE]]·[[DAY0_foundations]]·[[DAY3_ppo_rsl_rl]]·[[PPO_TUNING]]·[[TENSORBOARD]]·[[DAY6_ros2_sim2real]]을 그대로 보고,
여기서는 **Go2에서 실제로 달라진 것 + 직접 돌린 Go2 실험 수치**만 본다.

## 노트 구성
| 노트 | 내용 | 대응 G1 노트 |
|---|---|---|
| [[GO2_VS_G1]] | **한 장 종합 비교** (형태·액추에이터·reward·태스크·config) — 여기부터 | 전체 |
| [[GO2_STRUCTURE_ACTUATOR]] | 실측 articulation(12-DOF) + DC-Motor 액추에이터 심화 | [[DAY1_isaac_sim_basics]] · [[DAY2_physics_actuators]] |
| [[GO2_REWARD]] | quadruped reward 델타 (biped 항이 왜 빠졌나) | [[DAY4_reward_shaping]] |
| [[GO2_EXPERIMENTS]] | 직접 돌린 Go2 학습/튜닝 결과 (flat·rough·entropy·kl) | [[DAY3_ppo_rsl_rl]] · [[PPO_TUNING]] · [[DAY5_sensors]] |

## Go2 태스크 ID (G1과 이름 규칙이 다름 — 주의)
```
Isaac-Velocity-Flat-Unitree-Go2-v0        # 평지 학습
Isaac-Velocity-Flat-Unitree-Go2-Play-v0   # 평지 재생/export
Isaac-Velocity-Rough-Unitree-Go2-v0       # 거친지형 학습
Isaac-Velocity-Rough-Unitree-Go2-Play-v0  # 거친지형 재생/export
```
> G1은 `Isaac-Velocity-Flat-G1-v0`였는데 Go2는 이름에 **`Unitree`가 들어간다.** (등록 태스크 ID 실측)

## 실행 (모두 `/isaac-sim/IsaacLab`에서)
```bash
cd /isaac-sim/IsaacLab
# 구조 실측
./isaaclab.sh -p /isaac-sim/rl_course_ws/scripts/inspect_go2.py
# 평지 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless --num_envs 4096 --max_iterations 300
# 재생 + policy export
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --num_envs 32
```
> 이 노트 세트의 모든 실험은 `rl_course_ws/logs/go2_*.log` 에 저장돼 있다.
> 실행 드라이버: `scripts/`(inspect) + 통합 학습 스크립트(flat→export→튜닝4종→rough).
