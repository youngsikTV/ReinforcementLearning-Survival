"""
보상 구조 실험 일괄 실행기: 프리셋 조건들을 같은 시드/에피소드/알고리즘으로 학습하고 compare_rewards 까지 실행
사용법:
  python -m scripts.run_reward_study                                  # 기본 프리셋 전부 (오래 걸림)
  python -m scripts.run_reward_study --only base scale2 step_only --seeds 0 1 --episodes 20000   # 동작 확인
  python -m scripts.run_reward_study --include_growth                 # 증가형 보상(주의 사항 참고)도 포함
  python -m scripts.run_reward_study --skip_train                     # 학습은 건너뛰고 비교만 다시
결과: results/<prefix>_<프리셋>_s<시드>/ (학습),  results/<name>/ (비교 그래프, 표)

프리셋 의도
  base        기본값 (step 1, day_bonus 5, death -20, invalid -0.5)
  scale2      base 의 모든 보상을 정확히 2배 -> 보상 '크기'만 바꾼 대조군.
              정책이 같으므로 base 와 결과가 (거의) 같아야 정상. 다르면 실험 설정/코드 점검 신호
  step_only   하루 보너스 제거 (day_bonus 0)
  day_only    step 보상 제거 (step_reward 0): 하루 완료 시점에만 보상이 오는 희소한 구조
  bonus20     하루 보너스 강조 (day_bonus 20)
  death0      사망 페널티 없음
  death100    사망 페널티 강화 (-100)
  invalid0    무효 행동 페널티 없음
  grow        (--include_growth) step 보상이 경과 일수에 따라 증가 (step_growth 0.1)
              !! 주의: 보상이 '일수'에 의존하는데 Q-table 상태에는 일수가 없다.
                 에이전트가 일수를 볼 수 없어 보상이 상태만으로 정해지지 않는다(마르코프 성질 불성립).
                 따라서 이 조건의 결과는 '일수를 모르는 에이전트' 기준으로만 해석할 것.
"""
import argparse
import subprocess
import sys

PRESETS = {
    "base":      [],
    "scale2":    ["--step_reward", "2", "--day_bonus", "10", "--death_penalty", "-40", "--invalid_penalty", "-1"],
    "step_only": ["--day_bonus", "0"],
    "day_only":  ["--step_reward", "0"],
    "bonus20":   ["--day_bonus", "20"],
    "death0":    ["--death_penalty", "0"],
    "death100":  ["--death_penalty", "-100"],
    "invalid0":  ["--invalid_penalty", "0"],
    "grow":      ["--step_growth", "0.1"],
}
DEFAULT_ORDER = ["base", "scale2", "step_only", "day_only", "bonus20", "death0", "death100", "invalid0"]


def run(cmd):
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main(a):
    names = a.only if a.only else list(DEFAULT_ORDER) + (["grow"] if a.include_growth else [])
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        raise SystemExit(f"알 수 없는 프리셋: {unknown}. 사용 가능: {list(PRESETS)}")
    tags = []
    for n in names:
        tag = f"{a.prefix}_{n}"
        tags.append(tag)
        if a.skip_train:
            continue
        cmd = [sys.executable, "-m", "scripts.run_seeds", "--tag", tag, "--algos", a.algo,
               "--seeds", *[str(s) for s in a.seeds], "--episodes", str(a.episodes), "--alpha", str(a.alpha),
               "--gamma", str(a.gamma), "--eval_every", str(a.eval_every), "--eval_episodes", str(a.eval_episodes),
               "--workers", str(a.workers)] + PRESETS[n]
        run(cmd)
    run([sys.executable, "-m", "scripts.compare_rewards", "--tags", *tags, "--algo", a.algo, "--name", a.name])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="+", default=None, help="실행할 프리셋만 선택")
    p.add_argument("--include_growth", action="store_true", help="증가형 보상(grow) 프리셋 포함")
    p.add_argument("--algo", default="qlearning")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--episodes", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.1, help="모든 조건에 같은 값 사용 (알고리즘 비교와 같은 0.1 권장)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--prefix", default="rw")
    p.add_argument("--name", default="reward_study", help="비교 결과 폴더 이름 (results/<name>/)")
    p.add_argument("--skip_train", action="store_true")
    main(p.parse_args())
