# TensorBoard 읽는 법 — rsl_rl 학습 모니터링

> rsl_rl(PPO)로 G1/H1을 학습할 때 TensorBoard에서 **뭘 봐야 하는지** 정리한 참고 노트.
> 여는 법: `cd /isaac-sim/IsaacLab && ./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl`
> → 브라우저 `http://localhost:6006` (또는 `6007`). 태그 이름은 rsl_rl 소스에서 실측한 실제 이름.

---

## 0. 화면 사용법 (3가지만)

- **왼쪽 `Runs` 목록**: 지금까지 돌린 학습이 각각 다른 색 선으로 나온다(예: `g1_flat/2026-...`, `g1_rough/...`).
  체크박스로 켜고 끄면 **여러 학습을 겹쳐 비교**할 수 있다 → 4일차 A/B 실험이 여기서 두 선으로 보인다.
- **`Smoothing` 슬라이더**(좌하단): 곡선이 지글거리면 **0.6~0.9**로. 흐린 선=원본, 진한 선=평활화 추세. **추세는 진한 선**을 본다.
- **가로축(X) = Step = iteration 번호**, 세로축(Y) = 값. 상단 검색창에 **아래 태그를 그대로 입력**하면 그 그래프만 골라 본다.

---

## 0-B. 지표가 실제로 "뭘 뜻하는" 숫자냐 (개념 풀이)

그래프 이름만 봐선 감이 안 오는 핵심 지표 3개:

- **`Train/mean_episode_length` (eplen) = 한 판 버틴 스텝 수.**
  에피소드(리셋~넘어짐/시간초과 한 판)가 평균 몇 스텝 지속됐나. **50Hz라 스텝 ÷ 50 = 초** (eplen 104 ≈ 2.08초).
  넘어지면 판이 끝나니 **길수록 오래 안 넘어짐 = 잘 걷는다.** 최대값은 `episode_length_s`(G1=20초=1000스텝)에서 잘림.
  → 초반 학습의 가장 정직한 신호.

- **`Policy/mean_noise_std` (noise_std) = 탐험량 σ.**
  정책은 하나의 행동이 아니라 확률분포(평균 μ + 퍼짐 σ)를 낸다([[DAY0_foundations]]). σ가 그 **퍼짐 = 얼마나 무작위로 흔들며 시도하나 = 탐험량**.
  크다=여기저기 크게 시험(탐험↑), 작다=거의 정해진 대로(수렴). **보통 학습되며 서서히 감소**가 건강. `entropy_coef`가 이 값을 조종함([[PPO_TUNING]] 부록 실측: ent 0.0→σ감소, 0.05→σ증가).

- **`Train/mean_reward` (reward) = 매 판 받은 총 점수 평균.**
  ⚠️ 절대값만 보지 말 것 — 초반엔 짧게 넘어져 벌점이 안 쌓여 0에 가깝다가, 잘 버틸수록 오히려 더 음수가 될 수 있다.
  **판단은 eplen·track 보상·넘어짐률과 함께**(아래 ①②③).

> PPO 손잡이(clip_param, learning_rate, KL, desired_kl, entropy_coef)의 개념은 [[PPO_TUNING]] "개념 사전"에 정리.

---

## 1. 꼭 봐야 할 그래프 (우선순위 순)

### ① `Train/mean_episode_length` — 가장 먼저, 가장 중요
넘어지기 전까지 **몇 스텝 버티는가**. **올라가면 학습되는 것**. 초반엔 가장 정직한 신호.
- 좋음: 우상향 📈 (우리 실험: 11 → 104)

### ② `Train/mean_reward` — 전체 보상 (⚠ 함정 주의)
- ⚠️ **초반엔 내려가도 정상.** 처음엔 즉시 넘어져(짧은 에피소드) 벌점 쌓일 새가 없어 0에 가깝고,
  잘 버틸수록 자잘한 벌점이 오래 누적돼 더 음수가 된다. → **반드시 ①과 같이 본다.**
- 후반(수백 iter 뒤)엔 추종 보상이 커지며 우상향으로 돌아선다.

### ③ `Episode_Reward/track_lin_vel_xy_exp` — 진짜 목표
명령한 속도로 **실제로 가고 있는가**. **올라가야** 잘 걷는 것. (0 근처면 아직 명령 무시 중)

### ④ `Episode_Termination/base_contact` — 넘어짐 비율
몸통이 땅에 닿아 종료된 비율(1.0 = 100% 넘어짐). **내려가야**(→0) 좋다.

### ⑤ `Metrics/base_velocity/error_vel_xy` — 속도 오차
명령 속도와 실제 속도의 차이. **내려가야** 좋다.

---

## 2. 참고만 하면 되는 그래프 (초보는 안 봐도 됨)

| 태그 | 뜻 | 보는 법 |
|---|---|---|
| `Loss/learning_rate` | rsl_rl이 **자동 조절**하는 학습률(adaptive) | 오르내리는 게 정상 |
| `Policy/mean_noise_std` | 탐험량 σ | 서서히 **감소** = 정책이 자신감 생김 |
| `Loss/value_function`, `Loss/surrogate`, `Loss/entropy` | PPO 내부 학습 손실 | 발산(치솟음)만 아니면 OK |
| `Perf/total_fps`, `Perf/collection time`, `Perf/learning_time` | 학습 **속도** | 학습 품질과 무관 |
| `Episode_Reward/<각 항>` | 보상 항별 기여(feet_air_time, action_rate, joint_deviation …) | **4일차 reward 튜닝** 때 개별로 확인 |
| `Curriculum/terrain_levels` | (rough만) 지형 난이도 승급 | 올라갈수록 어려운 지형 통과 |

---

## 3. 태그 그룹 지도 (검색창에 그룹 이름만 쳐도 됨)

```
Train/        mean_reward, mean_episode_length          ← 큰 그림
Episode_Reward/   track_lin_vel_xy_exp, feet_air_time, action_rate_l2, ...  ← 보상 항별
Episode_Termination/  base_contact, time_out            ← 왜 에피소드가 끝났나
Metrics/      base_velocity/error_vel_xy, error_vel_yaw ← 추종 오차(행동 지표)
Loss/         value_function, surrogate, entropy, learning_rate  ← PPO 내부
Policy/       mean_noise_std                            ← 탐험량
Perf/         total_fps, collection time, learning_time ← 속도
Curriculum/   terrain_levels                            ← (rough) 난이도
```

---

## 4. 한 문장 요약

> **`Train/mean_episode_length` ↑ + `Episode_Reward/track_lin_vel_xy_exp` ↑ + `Episode_Termination/base_contact` ↓**
> 이 셋이 동시에 좋아지면 잘 학습되는 것. **`mean_reward` 하나만 보고 판단하지 말 것**(3일차 핵심 교훈).

**첫 행동**: 검색창에 `episode_length` → 우상향하는지부터 확인. 그게 "학습이 살아있다"는 첫 신호.

---

## 5. 이 워크스페이스의 학습 로그 (stdout 텍스트로도 확인 가능)
`logs/day3_train.log`, `logs/day4_A_baseline.log`, `logs/day4_B_heavy_actionrate.log`, `logs/day5_rough.log`
→ TensorBoard 없이도 로그에서 `Mean reward` / `Mean episode length` / `Episode_Reward/*`를 grep으로 볼 수 있다.
관련 개념: [[DAY3_ppo_rsl_rl]](학습 루프·지표 해석), [[DAY4_reward_shaping]](보상 항별 튜닝).
