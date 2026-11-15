# Minecraft Survival RL (강화학습 팀 프로젝트)

마인크래프트 세계관에서 에이전트가 하루(6개 시간대)마다 행동을 선택하며 **최대한 오래 생존**하도록 학습시키는 프로젝트. 수업에서 배운 Tabular RL(Q-learning, SARSA, Monte Carlo)을 직접 구현해 비교하고, 학습 과정에서 행동 선택 전략이 어떻게 바뀌는지 분석한다.

> 실행·분석 방법 전체는 `ANALYSIS_GUIDE.txt`를 참고.

## 1. 팀 구성 및 역할
| 이름 | 역할 | 주요 담당 |
|---|---|---|
| (천대웅) | 환경·보상 설계 | `envs/`, 보상 구조 실험 |
| (김시은) | 알고리즘·실험·분석 | `agents/`, `scripts/train.py`, 하이퍼파라미터 실험, `scripts/analyze.py` |
| (이효재) | 발표·문서 | 발표자료, 문서화 |

## 2. 환경 세팅 (Windows / VS Code)
**권장: Python 3.12** (3.14에서는 pygame 설치가 실패할 수 있음. 팀원 모두 같은 버전 사용)
```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
- 프롬프트 맨 앞에 `(.venv)`가 보이는지 확인한다. 보이지 않으면 `.venv\Scripts\activate`를 다시 실행한다.
- VS Code: `Ctrl+Shift+P` -> `Python: Select Interpreter` -> `.venv` 선택.
- 모든 명령은 **프로젝트 루트 폴더**(`agents`, `scripts`가 보이는 위치)에서 실행한다.
- 시각화는 `pygame-ce`를 사용한다 (`pygame`이 아님). 둘을 같이 설치하면 충돌하므로 `pip uninstall pygame` 후 설치한다.
- `pip install -r requirements.txt`는 한 패키지만 실패해도 전체가 중단된다. 실패하면 오류가 난 패키지를 빼고 따로 설치한다.

설치 확인:
```
python -c "import numpy, pandas, gymnasium, pygame; print('OK')"
```

## 3. 실행 순서
> 명령 뒤에 `#` 주석을 붙이지 말 것 (Windows cmd에서는 주석이 아니라 인자로 들어가 오류가 난다).

1. 기준선(random / 규칙 기반) 확인
```
python -m agents.baselines
```
2. 학습 (기본 100,000 에피소드, 몇 분 소요. 빠른 확인은 `--episodes 20000`)
```
python -m scripts.train --algo qlearning --tag base
python -m scripts.train --algo sarsa --tag base
python -m scripts.train --algo mc --tag base
```
3. 분석 그래프 (`results/<tag>/figs/`에 저장)
```
python -m scripts.analyze --tag base --algo qlearning
```
4. 알고리즘 비교 (위에서 세 알고리즘을 같은 `--tag`, 같은 `--episodes`로 학습한 뒤 실행)
```
python -m scripts.compare --tag base
```
`results/base/figs/compare_curve.png`(학습 곡선 비교)와 `results/base/compare_table.csv`(최종 성능 표)가 저장된다.

4-1. 여러 시드 반복 실험 (시드 평균 ± 표준편차, 탐험 없는 평가 기준). 시간이 오래 걸리므로 먼저 `--episodes 20000 --seeds 0 1`로 확인
```
python -m scripts.run_seeds --tag base --algos qlearning sarsa mc --seeds 0 1 2 3 4 --episodes 100000 --workers 3
python -m scripts.compare --tag base --multi
```
`results/base_s0`, `base_s1`, ... 에 시드별 결과가, `results/base_summary/`에 비교 그래프와 표가 저장된다.
학습 중 `--eval_every`(기본 1000) 에피소드마다 탐험 없이(greedy) 고정 시드 100판을 평가해 `<algo>_eval.csv`에 기록한다.

4-2. 보상 구조 비교 실험 (공통 지표는 총 보상이 아니라 **탐험 없는 평가의 생존 일수**)
```
python -m scripts.run_seeds --tag rw_base  --algos qlearning --seeds 0 1 2
python -m scripts.run_seeds --tag rw_step  --algos qlearning --seeds 0 1 2 --day_bonus 0
python -m scripts.run_seeds --tag rw_bonus --algos qlearning --seeds 0 1 2 --day_bonus 20
python -m scripts.run_seeds --tag rw_grow  --algos qlearning --seeds 0 1 2 --step_growth 0.1
python -m scripts.compare_rewards --tags rw_base rw_step rw_bonus rw_grow --algo qlearning --name reward_study
```
보상 옵션: `--step_reward`, `--day_bonus`, `--death_penalty`, `--invalid_penalty`, `--step_growth`(오래 살수록 step 보상 증가), `--day_growth`(하루 보너스 증가).
결과: `results/reward_study/reward_compare_curve.png`, `reward_compare_table.csv` (학습 후반 10%의 종료 원인 비율 포함)

4-3. 학습 단계별 분석 (최신 `train.py`로 학습한 결과에서만 동작)
```
python -m scripts.analyze --tag base_s0 --algo qlearning
```
`figs/`에 학습 곡선, 시간대별 행동 분포, **종료 원인 비율**(굶주림/기력 고갈/몬스터/생존), **행동 지표**(밤 시작 시 쉼터 준비율, 하루당 식사 횟수, 식사 시점 배고픔, 무의미한 행동 비율)가 저장된다.
종료 원인은 마지막 step에서 가장 큰 피해를 준 원인으로 기록한다.

5. 텍스트 시연
```
python -m scripts.play --tag base --algo qlearning
```
6. 시각화 (pygame 창). 학습 결과가 있어야 `--policy q` 사용 가능. `--ckpt`로 학습 시점(0, 10, 30, 60, 100)을 고를 수 있다 (생략하면 최종)
```
python -m scripts.visualize --policy q --tag base --algo qlearning
python -m scripts.visualize --policy q --tag base --algo qlearning --ckpt 30
python -m scripts.visualize --policy heuristic
python -m scripts.visualize --policy random
```
7. 시점별 시연 영상(GIF) 일괄 생성 (학습 전/초기/중간/최종 비교용, 같은 시드로 생성)
```
python -m scripts.record_all --tag base --algos qlearning sarsa mc
```
`results/base/videos/`에 `random.gif`, `heuristic.gif`, `<algo>_p<진행률>.gif`가 저장된다.

조작: `Space` 일시정지 / `R` 새 에피소드 / `↑↓` 속도 조절 / `ESC` 종료. 한 판이 끝나도 창은 자동으로 닫히지 않는다.

## 4. 폴더 구조
```
Survival/
  envs/survival_env.py      환경 (Gymnasium 호환, 이산 상태 31,104개)
  agents/tabular.py         Q-learning / SARSA / Monte Carlo
  agents/baselines.py       random / 규칙 기반 정책 + 평가 함수
  scripts/train.py          학습 + 로그 저장 (생존일, 시간대별 행동 횟수)
  scripts/analyze.py        학습 곡선, 학습 단계별 행동 분포 그래프
  scripts/compare.py        알고리즘 비교 그래프·표 (단일 실행 / --multi 다중 시드)
  scripts/run_seeds.py      시드 x 알고리즘 일괄 학습 (병렬, 보상 옵션 전달)
  scripts/compare_rewards.py 보상 구조 비교 (생존일 기준)
  scripts/play.py           텍스트 시연
  scripts/visualize.py      pygame 시각화 (체크포인트 선택, GIF 저장)
  scripts/record_all.py     시점별 시연 GIF 일괄 생성
  results/<tag>/            실험 결과 (tag별로 분리)
```

## 5. 환경 설명
- **시간**: 1 step = 하루의 한 시간대, 하루 = 6 step (아침, 점심, 오후, 저녁, 밤, 늦은밤)
- **상태**: 배고픔(0~5), 체력 HP(0~5), 기력(0~5), 음식(0~3), 목재(0~2), 시간대(0~5), 쉼터 유무(0/1)
- **행동**: 휴식 / 식사 / 사냥(채집) / 목재 수집 / 쉼터 건설
- **규칙**: 밤에 쉼터가 없으면 몬스터 습격 확률이 높고, 쉼터는 하루만 유지된다. 배고픔이나 기력이 0이면 HP가 줄고, HP가 0이면 사망한다.
- **종료**: 사망 또는 목표 일수(기본 30일) 도달
- **보상**: 매 step +1, 하루 생존 시 +5, 사망 -20, 잘못된 행동 -0.5 (기본값). `train.py`/`run_seeds.py`의 보상 옵션으로 변경 가능

## 6. 실험 계획 및 결과
| 실험 | 내용 | 상태 |
|---|---|---|
| 기준선 | random 약 1.9일, 규칙 기반 약 8.9일 (평균 생존일) | 완료 |
| 알고리즘 비교 | Q-learning vs SARSA vs MC | (진행 중) |
| 보상 구조 비교 | step 보상만 / 하루 보너스 크게 / 증가형 보상 | (예정) |
| 하이퍼파라미터 | alpha, gamma, epsilon 감소 속도 | (예정) |
| 전략 변화 분석 | 학습 초기/중기/후기의 시간대별 행동 분포 | (예정) |

결과 그래프와 해석은 이 표 아래에 추가한다. (예: `results/base/figs/qlearning_curve.png`)

## 7. 한계 및 배운 점
- 상태 수가 많아 Q-learning 수렴이 느리다 (에피소드 수와 평균 생존일의 관계 정리 필요).
- 환경 파라미터는 임의로 정한 값이라 튜닝에 따라 결과가 달라질 수 있다.
- (실험하며 배운 점을 계속 기록)

## 8. 참고
- 수업 자료: Lecture 2~5 (MDP, Value Iteration, Monte Carlo, SARSA, Q-learning)
- Gymnasium: https://gymnasium.farama.org/
