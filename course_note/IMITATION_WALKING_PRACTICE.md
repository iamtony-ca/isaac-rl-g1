# G1 Walking (모션 이미테이션) 실습 가이드 — 36_1 → 36_2 → 36_3

> **목적**: 강좌 36강의 **모션 모방(imitation) 보행 파이프라인**을 이 파일 하나로 처음부터 끝까지 직접 돌린다.
> **경로**: 모든 예제는 `/isaac-sim/rl_course_ws/course_materials/36_g1_walking/`. 실행 래퍼는 `/isaac-sim/IsaacLab/isaaclab.sh -p`.
> **왜 별도 가이드인가**: 내 기존 [[QUICK_START]]는 `Isaac-Velocity-Flat-G1`(velocity **command** 추종) 파이프라인이다. 여기는 **레퍼런스 모션(사람 보행 리타게팅)을 정답으로 추종하는** 완전히 다른 방식이다. 개념 해설은 [[course_materials]]의 §갭6, 코드 구조는 `g1_walking_env.py` 참고.

```
36_1 재생 (정답 확인) → 36_2 학습 (RSI+PPO) → TensorBoard 모니터 → 36_3 평가 (지표+토크)
```

> **핵심 원리 3줄 요약** ([[course_materials]] §갭6):
> 1. 관측에 **레퍼런스 관절각 + 보행 위상(sin/cos)** 을 넣어 정책이 "지금 취할 자세"를 본다.
> 2. 보상은 전부 `exp(-오차)` 커널 — **joint_tracking(전신 자세 추종)** 이 핵심(weight 5).
> 3. **RSI**(리셋 시 레퍼런스 임의 프레임의 자세·속도로 시작)가 수렴을 만든다.

---

## 0. 사전 준비 (딱 한 번)

### 0-1. 실행 디렉토리 규칙 (중요)
세 스크립트는 같은 폴더의 `g1_walking_env.py`를 `import`하고, **로그/체크포인트를 실행 위치(cwd) 기준**으로 쓴다. 그래서 **반드시 36 폴더 안에서 실행**한다. 그러면 `36_2`가 저장한 체크포인트를 `36_3`가 자동으로 찾는다.

```bash
cd /isaac-sim/rl_course_ws/course_materials/36_g1_walking
# 이 안에서 아래 명령들을 실행. 래퍼는 절대경로로 호출.
ISAACLAB=/isaac-sim/IsaacLab/isaaclab.sh
```

### 0-2. 레퍼런스 모션 데이터 clone (필수)
용량 문제로 저장소에 모션 `.pkl`이 없다. HuggingFace에서 받아 **이 폴더 아래** `g1_retargeted_motions/`로 넣는다. (git-lfs 필요)

```bash
# git-lfs 준비 (한 번만)
git lfs install

# 36_g1_walking/ 안에서:
git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions g1_retargeted_motions
```
- 코드가 찾는 위치(우선순위): `36_g1_walking/g1_retargeted_motions/` → 없으면 `course_materials/g1/g1_retargeted_motions/`.
- 기본 클립: **`B3_-_walk1_stageii.pkl`** (전체 229프레임 @30fps, 학습엔 프레임 42~155 순환 구간만 사용, 평균 ~1.07 m/s).
- 데이터셋 하위 폴더: `ACCAD / lafan1 / dance_db / kungfu` 등 → `--motion_pkl 이름`으로 부분 일치 검색 가능.

### 0-3. 모션 파일이 잘 잡히는지 스모크 확인
아래 `36_1`을 `--use_cycle`로 5초만 띄워 에러 없이 재생되면 준비 완료.

---

## ① 36_1 — 레퍼런스 모션 재생 (학습 전 "정답 동작" 눈으로 확인)

물리·RL 없이 레퍼런스 pose를 매 스텝 로봇에 **직접 기록**(kinematic)해 보여준다. 학습이 목표로 삼을 동작을 먼저 본다.

🖥️ **GUI (권장 — 눈으로 확인)**
```bash
# 전체 클립 3회 재생
$ISAACLAB -p 36_1_motion_playback.py

# 학습에 실제 쓰는 순환 구간(프레임 42~155)만 반복 — 걸음이 매끄럽게 이어지는지 확인
$ISAACLAB -p 36_1_motion_playback.py --use_cycle

# 다른 모션 재생 (이름/부분일치 검색)
$ISAACLAB -p 36_1_motion_playback.py --motion_pkl walk_turn_left
```
주요 인자: `--use_cycle`(순환구간만), `--loops N`(반복횟수, 기본 3), `--num_envs`, `--motion_pkl`.

**무엇을 볼까**
- [ ] 콘솔의 **관절 매핑 표**(pkl 23관절 → 로봇 관절, `wrist→elbow_roll`은 `(alias)` 표시)가 에러 없이 출력되는지
- [ ] G1이 **자연스러운 사람 걸음**으로 움직이는지 (이게 학습의 상한선)
- [ ] `[WARNING] 몸통 위 방향 z=...` 이 뜨면 쿼터니언(xyzw→wxyz) 문제 — 정상이면 안 뜬다
- [ ] `--use_cycle`에서 구간 끝→처음 이음새가 튀지 않는지 (시작=끝 관절각 차 ≤0.03rad라 매끄러워야 함)

> `--headless`로 돌리면 "재생 완료" 출력 후 종료 단계에서 멈출 수 있음 → `Ctrl+C`로 끝내도 됨. 재생 확인은 GUI가 목적이니 headless는 불필요.

---

## ② 36_2 — 모방 학습 (RSI + PPO)

`g1_walking_env.py`의 환경(관측 103차원 / 보상 10항목 / RSI 이벤트)에 rsl_rl PPO를 적용. 기본 **2048 env × 2000 iter**.

### 2-1. 먼저 스모크 테스트 (1~2분, 파이프라인 검증용)
전체 학습은 무겁다. 먼저 작게 돌려 에러·크래시가 없는지 확인한다.
```bash
$ISAACLAB -p 36_2_train_walking.py --headless --num_envs 64 --max_iterations 5
```
- 통과 기준: 환경 생성 → "PPO 학습 시작" → iter 5까지 돌고 체크포인트 저장되면 OK. (아직 못 걷는 게 정상)

### 2-2. 본 학습 (Headless 권장)
```bash
$ISAACLAB -p 36_2_train_walking.py --headless
# 가벼운 GPU면 환경 수를 줄여서:
$ISAACLAB -p 36_2_train_walking.py --headless --num_envs 1024 --max_iterations 1500
```
주요 인자: `--num_envs`(기본 2048), `--max_iterations`(기본 2000), `--motion_pkl`.
- 체크포인트: `36_g1_walking/logs/rsl_rl/g1_walking_tutorial/` 에 **50 iter마다** `model_*.pt` 저장.
- 네트워크 `[256,128,64]`, `entropy_coef=0.008`(레퍼런스가 답을 주니 탐색 덜 필요), adaptive KL. → 값 의미는 [[PPO_TUNING]].

> ⚠️ **RSI 함정 (코드에 이미 반영됨, 손대지 말 것)**: `runner.learn(..., init_at_random_ep_len=False)`.
> 35강 standing은 `True`였지만, walking은 **RSI가 이미 시작 위상을 흩뜨린다.** `True`로 바꾸면 `episode_length_buf` 기반 위상 계산이 RSI 시작 자세와 어긋나 **모방 학습이 붕괴**한다. → 배경은 [[course_materials]] §갭6-4.

> ℹ️ 이 튜토리얼 스크립트에는 **`--resume` 플래그가 없다.** (표준 IsaacLab `train.py`의 resume 규칙 [[isaaclab-resume-gotcha]]와 무관 — 여기선 중단하면 처음부터.) 이어학습이 필요하면 스크립트에서 `runner.load(...)`를 직접 추가해야 한다.

---

## ③ TensorBoard 모니터링 (학습 로그 읽는 법)

새 터미널에서 (역시 36 폴더 기준):
```bash
cd /isaac-sim/rl_course_ws/course_materials/36_g1_walking
/isaac-sim/IsaacLab/isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl/g1_walking_tutorial
# 원격이면 --bind_all 추가 후 브라우저로 접속
```
개념·그래프 보는 법은 [[TENSORBOARD]]. **이 태스크에서 특히 봐야 할 곡선**:

| 지표 | 정상 신호 | 이상 신호 → 대응 |
|---|---|---|
| `Episode_Reward/joint_tracking` | **초반부터 0보다 큼** | 계속 0 근처 → RSI/관절매핑 문제 (36_1로 매핑 재확인) |
| `Train/mean_episode_length` | **500(=10초)을 향해 증가** | 낮게 정체 → 금방 넘어짐 (아래 참고) |
| `Train/mean_reward` | 꾸준히 증가 | — |

> **진단 팁 (36_2 docstring)**: `mean_reward`는 오르는데 `episode_length`가 정체 → **"걷지 않고 제자리에서 버티기"만 배우는 중.** `RewardsCfg`에서 `speed_tracking` weight를 올려본다(예 2.0→3.0). 반대로 몸을 던지듯 돌진하면 speed 커널이 unbounded가 아닌지 확인(이미 exp 형태라 정상).

---

## ④ 36_3 — 학습 정책 평가 (지표 + 토크 시각화)

최신 체크포인트를 자동 탐색해 로드하고, G1이 실제로 걷는 모습 + 정량 지표를 낸다.

🖥️ **GUI (권장 — 걷는 모습 관찰)**
```bash
# 최신 model_*.pt 자동 로드 후 평가 (env 4, 10 에피소드)
$ISAACLAB -p 36_3_play_walking.py

# 특정 체크포인트 지정
$ISAACLAB -p 36_3_play_walking.py --checkpoint logs/rsl_rl/g1_walking_tutorial/model_2000.pt

# 학습 전후 비교용: 랜덤 정책
$ISAACLAB -p 36_3_play_walking.py --random
```
주요 인자: `--checkpoint`(생략 시 최신 자동), `--random`, `--num_envs`(기본 4), `--num_episodes`(기본 10), `--motion_pkl`.

**출력 지표 읽는 법**
| 지표 | 의미 | 좋은 값 방향 |
|---|---|---|
| **관절 추종 RMS 오차 [rad]** | 23관절이 레퍼런스와 얼마나 벌어졌나 | **작을수록** 모방 정확 |
| **평균 전진 속도 [m/s]** | 몸 기준 +x 속도 (레퍼런스 ~1.07) | 레퍼런스에 **가까울수록** |
| **생존율 (10초 보행)** | 에피소드 길이 ≥ 최대의 95%(≈475/500스텝) 비율 | **높을수록** (안 넘어짐) |

- 종료 사유는 `생존 (10초)`(=truncation, 성공) vs `넘어짐`(=termination)으로 구분 출력.
- **토크 그래프**: env 0의 왼다리 6관절 + 허리를 3단(다리 / 발목 / 허리)으로 실시간 표시하고 종료 시 **`36_3_torque_result.png`** 저장 + 관절별 평균/최대 |τ| 통계 출력. G1은 implicit(PD) 액추에이터라 PhysX가 토크를 직접 안 주고, IsaacLab이 `PD(위치오차,속도오차)`로 근사한 `applied_torque`를 쓴다([[DAILY_NOTES]] 2일차 PD 게인).

**비교 실습**
- [ ] `--random` 먼저 → 즉시 넘어짐(생존율 ~0%, RMS 큼) 확인
- [ ] 학습 정책 → 생존율↑, RMS↓, 속도가 ~1.07에 근접하는지
- [ ] `model_500.pt` vs `model_2000.pt` 지정 비교 → 학습이 진행될수록 지표 개선되는지

---

## ⑤ 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `FileNotFoundError: 레퍼런스 모션 pkl` | 0-2 데이터 clone 누락 또는 위치 오류. `36_g1_walking/g1_retargeted_motions/` 아래 `.pkl`이 있는지 확인. git-lfs 미설치면 파일이 포인터만 받아짐 → `git lfs install` 후 재clone |
| `ModuleNotFoundError: g1_walking_env` | 36 폴더 **밖**에서 실행. `cd .../36_g1_walking` 후 실행 |
| 36_3가 "체크포인트를 찾을 수 없습니다 → 랜덤" | 36_2를 다른 cwd에서 돌려 `logs/`가 딴 곳에 생김. 둘 다 **같은 36 폴더**에서 실행하거나 `--checkpoint`로 직접 지정 |
| 학습이 안 오름 / joint_tracking≈0 | 관절 매핑 실패 가능 → 36_1의 매핑 표 재확인. `init_at_random_ep_len`을 `True`로 바꾸지 않았는지 확인(반드시 False) |
| `[WARNING] 몸통 위 방향 z<0.9` (36_1) | 쿼터니언 xyzw→wxyz 변환 문제. 다른 데이터셋 클립이 순서가 다를 수 있음 |
| headless 종료가 멈춤 (36_1) | "재생 완료" 떴으면 정상, `Ctrl+C` |

---

## ⑥ 더 해볼 실험 (이해가 남는 것들)

1. **모션 바꿔 학습**: `--motion_pkl run1_subject2` 등으로 다른 클립을 36_1로 확인 → 36_2로 학습 → 걸음이 그 모션을 닮는지. (달리기/방향전환 클립은 난이도↑)
2. **보상 가중치 튜닝**: `g1_walking_env.py`의 `RewardsCfg`에서 `speed_tracking`↑ 또는 `arm_tracking`↓ → 걸음/팔흔들기 변화 관찰. 수정 후 **반드시 재학습**(관측/네트워크는 그대로 둘 것 — 그래야 36_3 로드 호환).
3. **RSI 제거 실험(교육용)**: `EventsCfg`의 `reset_motion`을 비활성화하고 학습 → 수렴이 얼마나 나빠지는지 체감 (RSI의 가치 확인). *끝나면 되돌리기.*
4. **standing과 비교**: [[course_materials]] §갭7의 35강(고정 높이 유지)과 걸음(프레임별 높이 추종)의 보상 설계 차이를 코드로 대조.
5. **velocity-tracking과 비교**: 같은 G1을 [[QUICK_START]]의 `Isaac-Velocity-Flat-G1`로도 학습해 걸음걸이·자연스러움·튜닝 난이도를 비교 → imitation vs command-tracking 체감.

---

## 관련 노트
- 개념/코드 구조: [[course_materials]] (§갭6 imitation, §갭7 standing, §갭5 커스텀 env)
- 이론: [[DAY3_ppo_rsl_rl]] · [[PPO_TUNING]] · [[TENSORBOARD]] · [[DAILY_NOTES]]
- 대조 파이프라인(velocity-tracking): [[QUICK_START]]
- resume 주의: [[isaaclab-resume-gotcha]]
