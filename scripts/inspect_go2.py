# Copyright / 학습용 스크립트
# Unitree Go2(4족) articulation 구조(관절/링크/DOF/기본 자세/게인)를 출력한다.
#
# 실행:
#   cd /isaac-sim/IsaacLab
#   ./isaaclab.sh -p /isaac-sim/rl_course_ws/scripts/inspect_go2.py
#
# (처음 실행 시 Nucleus에서 Go2 USD를 받아오므로 시간이 걸릴 수 있음)
# inspect_g1.py의 Go2판 — G1_MINIMAL_CFG → UNITREE_GO2_CFG 로만 바꾼 구조.

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---- 1) 시뮬레이터 먼저 실행 (Isaac Lab 스크립트의 철칙: import 전에 App 부팅) ----
parser = argparse.ArgumentParser(description="Inspect Go2 articulation.")
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

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # Go2 = DC-Motor 4족


def main():
    # 최소한의 씬: 물리 컨텍스트 + 바닥 + 조명 + 로봇
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    robot = Articulation(UNITREE_GO2_CFG.replace(prim_path="/World/Robot"))

    sim.reset()  # 여기서 물리/텐서 버퍼가 초기화됨

    print("\n" + "=" * 70)
    print("Go2 ARTICULATION 구조 (4족 / DC-Motor)")
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
        print(f"  {group_name:10s}: 관절 {act.stiffness.shape[-1]:2d}개, Kp≈{kp:7.1f}, Kd≈{kd:6.2f}")

    print("=" * 70 + "\n")
    sys.stdout.flush()  # 파일 리다이렉트 시 print 버퍼가 안 비워지는 문제 방지

    # simulation_app.close()가 헤드리스에서 오래 스핀하는 경우가 있어 즉시 종료.
    os._exit(0)


if __name__ == "__main__":
    main()
