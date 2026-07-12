# Copyright / 학습용 스크립트
# G1 휴머노이드의 articulation 구조(관절/링크/DOF/기본 자세/게인)를 출력한다.
#
# 실행:
#   cd /isaac-sim/IsaacLab
#   ./isaaclab.sh -p /isaac-sim/rl_course_ws/scripts/inspect_g1.py
#
# (처음 실행 시 Nucleus에서 G1 USD를 받아오므로 시간이 걸릴 수 있음)

import argparse

from isaaclab.app import AppLauncher

# ---- 1) 시뮬레이터 먼저 실행 (Isaac Lab 스크립트의 철칙: import 전에 App 부팅) ----
parser = argparse.ArgumentParser(description="Inspect G1 articulation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # GUI 없이 구조만 출력

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 2) App 부팅 이후에만 Isaac Lab 모듈 import 가능 ----
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import G1_MINIMAL_CFG  # 우리가 STUDY_GUIDE에서 본 그 설정


def main():
    # 최소한의 씬: 물리 컨텍스트 + 바닥 + 조명 + 로봇
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    robot = Articulation(G1_MINIMAL_CFG.replace(prim_path="/World/Robot"))

    sim.reset()  # 여기서 물리/텐서 버퍼가 초기화됨

    print("\n" + "=" * 70)
    print("G1 ARTICULATION 구조")
    print("=" * 70)
    print(f"DOF(관절) 수         : {robot.num_joints}")
    print(f"body(링크) 수        : {robot.num_bodies}")
    print(f"actuator 그룹 수     : {len(robot.actuators)}")

    print("\n--- 관절 이름 (action/joint observation 순서와 동일) ---")
    for i, name in enumerate(robot.joint_names):
        print(f"  [{i:2d}] {name}")

    print("\n--- 링크(body) 이름 (contact/height-scan에서 body_names로 참조) ---")
    for i, name in enumerate(robot.body_names):
        print(f"  [{i:2d}] {name}")

    print("\n--- 기본 자세 default_joint_pos (초기/기준 각도) ---")
    default_q = robot.data.default_joint_pos[0]
    for name, q in zip(robot.joint_names, default_q.tolist()):
        print(f"  {name:32s} = {q:+.3f} rad")

    print("\n--- actuator 그룹별 PD 게인 ---")
    for group_name, act in robot.actuators.items():
        kp = act.stiffness.mean().item()
        kd = act.damping.mean().item()
        print(f"  {group_name:8s}: 관절 {act.stiffness.shape[-1]:2d}개, Kp≈{kp:7.1f}, Kd≈{kd:6.2f}")

    print("=" * 70 + "\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
