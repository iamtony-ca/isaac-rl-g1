# Go2 직접 돌린 실험 (DAY3·PPO_TUNING·DAY5 델타)

> 이 환경에서 실제로 돌린 Go2 학습·튜닝 결과. 방법론·지표 개념은 [[DAY3_ppo_rsl_rl]]·[[PPO_TUNING]]·[[TENSORBOARD]] 그대로.
> 모든 로그: `rl_course_ws/logs/go2_*.log`. 실행: `scripts/`의 통합 드라이버(flat→export→튜닝4종→rough).
> ⚠️ 지표 읽는 법: **reward 절대값보다 `eplen`(episode length)이 진짜 신호** — G1 3일차 교훈 동일([[DAY3_ppo_rsl_rl]]).

---

## 1. Flat 학습 — 4족은 평지 보행을 빠르게 배운다

`Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 4096 --max_iterations 300 --seed 1`

| iter | Mean reward | Mean episode length |
|---:|---:|---:|
| 296 | 34.66 | 1000.00 |
| 299 | **34.83** | **1000.00** |

- **결과: 300 iter(약 2분 40초, RTX 5090)만에 eplen 1000(만점) 도달** = 로봇이 20초 에피소드 내내 안 넘어지고 완벽히 걷는다.
- **G1 대비**: G1 flat은 기본 `max_iterations=1500`이 필요했다. Go2는 **1/5인 300**으로 config에 박혀 있고 실제로 그 안에 수렴 → **4족 평지 보행이 그만큼 쉽다**(4점 지지 정적 안정, [[GO2_VS_G1]] 1절).
- eplen 1000 = 에피소드 최대 스텝(20초 @ 50Hz)에 도달, 조기종료(넘어짐) 없음.

### observation / action / 네트워크 (export 로그 실측)
```
Observation (policy group) : 48차원
Actor MLP : Linear(48→128) → ELU → 128 → 128 → Linear(128→12)
```
- **obs 48차원 분해**: base 선속도(3) + 각속도(3) + projected_gravity(3) + velocity_commands(3) + joint_pos(12) + joint_vel(12) + last_action(12) = **48**. (평지라 height_scan 없음.)
- **G1 flat은 123차원** → Go2가 관절이 적어(12 vs 다수) obs·action·망이 모두 작다 = **flat 학습이 가볍고 빠른 근본 이유**.
- action 12차원 = 12관절 목표각 offset(`scale=0.25`).

---

## 2. Policy Export (sim-to-real 1단계) — G1과 동일 파이프라인

`play.py --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0` 실행 시 자동 export:
```
logs/rsl_rl/unitree_go2_flat/<run>/exported/
  ├─ policy.onnx   (164 KB)
  └─ policy.pt     (TorchScript, 174 KB)
```
- actor만 추출(critic·PPO 버림) → 온보드 실시간 추론용. 개념은 [[DAY6_ros2_sim2real]] 그대로.
- ⚠️ **play.py는 export 후 무한 재생 루프**를 돈다(headless에서도). 배치 스크립트에선 export 산출물 생성(로그의 `Exporting`)을 확인하면 프로세스를 종료해도 된다.

---

## 3. 튜닝 대조 실험 — `entropy_coef` (G1과 동일 조건으로 재현)

동일 조건: `flat Go2, seed 1, 300 iter, num_envs 512`, **entropy_coef만** 0.0 vs 0.05.
> (G1 PPO_TUNING 부록과 완전히 같은 설계 — 로봇만 Go2로.)

| iter | ent=0.0 · noise_std | ent=0.05 · noise_std | ent=0.0 · eplen | ent=0.05 · eplen | ent=0.0 · reward | ent=0.05 · reward |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 1.01 | 13.5 | 13.5 | −0.54 | −0.54 |
| 100 | 0.50 ↓ | 1.58 ↑ | **1000** | 884 | +4.37 | −24.4 |
| 200 | 0.39 ↓ | 2.02 ↑ | **1000** | 13.1 | +9.31 | −1.13 |
| 299 | **0.33** ↓ | **1.95** ↑ | **1000** | **13.3** | **+13.11** | **−0.90** |

**관찰 (G1보다 훨씬 극적)**
- `entropy_coef=0.0`: 탐험량 σ가 **1.00 → 0.33으로 수렴**, eplen이 **iter 100에 이미 1000(만점)**, reward가 꾸준히 +13까지 상승 → **완벽히 걷는다.**
- `entropy_coef=0.05`: 엔트로피 보너스가 σ를 **1.00 → 1.95로 폭발**시켜 정책이 계속 흔들림 → eplen이 100에서 잠깐 884까지 갔다가 **13으로 붕괴**(넘어짐), reward는 끝까지 음수(−0.90). **끝내 못 걷는다.**
- **G1과 비교**: G1은 ent=0.05에서도 eplen 38로 "덜 걸음"이었는데, Go2는 아예 **13으로 완전 붕괴** — Go2 flat이 쉬운 만큼 과한 탐험이 더 치명적으로 수렴을 막는다.

**교훈** ([[PPO_TUNING]] entropy 부록과 동일 메시지): `entropy_coef`는 탐험 유지 다이얼이고 `mean_noise_std`가 그 결과다. 낮으면 σ↓(수렴), 높으면 σ↑(발산). ⚠️ "그럼 0이 최고?"는 아니다 — 쉬운 flat에선 조기수렴이 유리했을 뿐, 어려운 태스크에선 entropy=0이 **조기수렴(premature convergence)** 위험. 기본값 0.01이 안전한 중간. (로그: `logs/go2_exp_ent_low.log`, `logs/go2_exp_ent_high.log`)

---

## 4. 튜닝 대조 실험 — `desired_kl` (adaptive lr을 몬다)

동일 조건에서 **desired_kl만** 0.002 vs 0.05.

lr은 stdout에 없어 TB 이벤트(`Loss/learning_rate`)에서 추출.

| iter | lr (kl=0.002) | lr (kl=0.05) | noise_std (0.002) | noise_std (0.05) | eplen (0.002) | eplen (0.05) | reward (0.05) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0e-5 | 5.1e-3 | 1.00 | 1.00 | 13.5 | 13.5 | −0.54 |
| 100 | 1.0e-5 | 1.1e-4 | 0.93 | 0.78 | 988 | 991 | +0.16 |
| 200 | 1.0e-5 | 2.0e-3 | 0.87 | 0.72 | 1000 | 955 | +11.1 |
| 299 | **1.0e-5** (바닥) | **8.6e-4** (높게 진동) | **0.84** | **0.65** | **23.6**(불안정) | **905** | **+18.70** |

**메커니즘 (KL 온도조절기 실증 — G1과 동일)**
- rsl_rl은 매 업데이트 후 실제 KL을 재서 lr을 ×/÷ 해 `desired_kl`에 맞춘다(lr 범위 ≈[1e-5, 1e-2]).
- **desired_kl=0.002**("아주 조금만 바꿔"): 작은 스텝도 KL이 0.002를 넘어 → 스케줄러가 lr을 계속 줄여 **최소값 1.0e-5에 붙어버림** → 정책이 거의 안 변함(**noise_std 0.84로 높게 유지**, 수렴 못함) → reward 정체(−0.66), eplen 불안정(299에서 23.6으로 붕괴).
- **desired_kl=0.05**("크게 바꿔도 돼"): KL 여유 → lr을 키움(최대 5e-3, kl=0.002 대비 ~500배) → **빠르고 안정적 학습**(reward +18.70로 전체 최고, eplen 905, noise_std 0.65로 수렴).

**교훈** ([[PPO_TUNING]] desired_kl 부록과 동일):
- **손잡이는 lr이 아니라 `desired_kl`** — lr은 그걸 맞추려 자동으로 따라온다(같은 방향: desired_kl↑ → lr↑).
- **Go2 특이점**: flat이 너무 쉬워 kl=0.002도 iter 200에 잠깐 eplen 1000을 찍지만, **정책이 "수렴 안 된 채"**(noise_std 0.84)라 불안정하게 붕괴. lr 바닥 메커니즘은 noise_std가 안 내려가는 것으로 확인된다.
- ⚠️ 이 쉬운 flat에선 공격적(0.05)이 유리했지만, rough·비전 등 어려운 태스크에선 큰 lr이 붕괴를 부를 수 있다. 기본값 **0.01이 안전한 중간.** (로그: `logs/go2_exp_kl_low.log`, `logs/go2_exp_kl_high.log`)

---

## 5. Rough 학습 + Flat vs Rough (센서 델타, DAY5)

`Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 4096 --max_iterations 1000 --seed 1`

### observation 차원 — height_scan이 붙는다 (실측)
| 태스크 | obs 차원 | 구성 |
|---|---:|---|
| Go2 **flat** | **48** | proprioceptive만 (base 속도6 + 중력3 + command3 + joint 24 + last_action12) |
| Go2 **rough** | **235** | 48 + **height_scan 187** (17×11 격자 RayCaster, `base`에 부착) |

- **핵심**: rough는 flat에 **height_scan 187차원**을 더한다(48 → 235). 이 187은 **G1 rough와 동일한 격자**(G1은 123→310, 역시 +187) — height-scan 그리드는 로봇과 무관하고, **부착 링크만 다르다**(Go2 `base` vs G1 `torso_link`). 개념은 [[DAY5_sensors]] 그대로.
- 그래서 rough 정책망도 커진다: flat `[128,128,128]` → rough **`[512,256,128]`** (235차원의 복잡한 지형함수를 배워야 하므로 표현력↑).

### rough 학습 결과 (4096env / 1000iter / seed1)
| iter | reward | eplen |
|---:|---:|---:|
| 0 | −0.57 | 13 |
| 250 | 14.25 | 946 |
| 500 | 19.37 | 949 |
| 750 | 18.85 | 926 |
| 999 | **21.63** | **944** |
- terrain curriculum(평지→경사·계단·박스 난이도 점증)에서도 **eplen 944**로 안정적으로 걷는다. 다만 flat의 1000(만점)엔 못 미침 = 거친 지형이라 가끔 넘어짐(정상).
- **주의**: rough reward(21.6)가 flat(34.8)보다 낮은 건 로봇이 못해서가 아니라 **reward weight 구성이 다르기 때문**(rough는 `feet_air_time` 0.01·`flat_orientation` 약함, flat은 0.25·−2.5). → **reward 절대값을 flat/rough 간 직접 비교 금지**, eplen이 공정한 신호([[TENSORBOARD]]).

### Flat vs Rough 종합 비교 (같은 로봇, 센서 유무)
| 항목 | **Flat** | **Rough** | 의미 |
|---|---:|---:|---|
| observation | 48 | **235** (+height_scan 187) | 지형 인지 추가 |
| 정책망 | [128,128,128] | **[512,256,128]** | 235차원 복잡함 → 표현력↑ |
| max_iterations | 300 | 1000 | 어려워서 더 오래 |
| 최종 eplen | **1000** | 944 | rough가 살짝 어려움 |
| throughput | **~215,000 steps/s** | **~38,000 steps/s** | **rough가 ~5.6배 느림** |
| 학습 시간(4096env) | ~3분 | ~44분 | 위 속도차 × iter수 |

- **핵심**: rough가 5.6배 느린 이유 = **height_scan 레이캐스팅(187 rays × 4096 env를 매 스텝) + 거친 지형 메쉬 충돌** 연산. 평지는 단순 plane이라 물리·센서가 가볍다. → 5일차 vision RL에서 "카메라·센서는 `num_envs`를 크게 못 쓴다"([[DAY5_sensors]])는 원칙의 정량적 실례.
- **G1과 동일 패턴**: flat→rough에서 obs가 +187(height_scan), 망이 커지고, eplen이 낮아지고, 속도가 느려진다. 부착 링크만 `base`(Go2) vs `torso_link`(G1) 차이([[GO2_STRUCTURE_ACTUATOR]]).

---

## 6. 자기점검 (Go2 델타)
- [ ] Go2 flat이 왜 G1보다 빨리 수렴하는지(obs·망 작음 + 4점 지지) 설명할 수 있다.
- [ ] `entropy_coef`·`desired_kl` 튜닝 결과가 G1과 같은 경향인지 다른지 안다.
- [ ] flat(48) vs rough obs 차원 차이가 height_scan에서 온다는 걸 안다.
- [ ] export 산출물(onnx/pt) 위치와 play.py 무한루프 주의점을 안다.
