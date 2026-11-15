"""
여러 시드 x 여러 알고리즘 학습을 한 번에 실행 (병렬)
사용법:  python -m scripts.run_seeds --tag base --algos qlearning sarsa mc --seeds 0 1 2 3 4 --episodes 100000 --workers 3
결과는 results/<tag>_s<seed>/ 에 저장된다. (예: results/base_s0, results/base_s1, ...)
이후 python -m scripts.compare --tag base --multi 로 시드 평균 ± 표준편차를 비교한다.
보상 구조 실험: --day_bonus 0 (step만) / --day_bonus 20 / --step_growth 0.1 등을 주고 tag를 다르게 할 것.
주의: 시간이 오래 걸린다. 처음에는 --episodes 20000 --seeds 0 1 로 동작부터 확인할 것.
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run_one(cmd):
    print(">", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main(a):
    cmds = []
    for algo in a.algos:
        for seed in a.seeds:
            cmds.append([sys.executable, "-m", "scripts.train", "--algo", algo, "--seed", str(seed),
                         "--tag", f"{a.tag}_s{seed}", "--episodes", str(a.episodes),
                         "--alpha", str(a.alpha), "--gamma", str(a.gamma),
                         "--eval_every", str(a.eval_every), "--eval_episodes", str(a.eval_episodes),
                         "--step_reward", str(a.step_reward), "--day_bonus", str(a.day_bonus),
                         "--death_penalty", str(a.death_penalty), "--invalid_penalty", str(a.invalid_penalty),
                         "--step_growth", str(a.step_growth), "--day_growth", str(a.day_growth)])
    print(f"총 {len(cmds)}개 학습 실행 (동시 {a.workers}개)")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        codes = list(ex.map(run_one, cmds))
    failed = [c for c, rc in zip(cmds, codes) if rc != 0]
    print(f"완료: 성공 {len(cmds) - len(failed)} / 실패 {len(failed)}")
    for c in failed:
        print("실패:", " ".join(c))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base")
    p.add_argument("--algos", nargs="+", default=["qlearning", "sarsa", "mc"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--episodes", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--step_reward", type=float, default=1.0)
    p.add_argument("--day_bonus", type=float, default=5.0)
    p.add_argument("--death_penalty", type=float, default=-20.0)
    p.add_argument("--invalid_penalty", type=float, default=-0.5)
    p.add_argument("--step_growth", type=float, default=0.0)
    p.add_argument("--day_growth", type=float, default=0.0)
    main(p.parse_args())