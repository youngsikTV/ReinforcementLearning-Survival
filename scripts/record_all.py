"""
학습 전/초기/중간/최종 시점의 시연 GIF를 한 번에 생성 (발표·보고서용)
사용법:  python -m scripts.record_all --tag base
        python -m scripts.record_all --tag base --algos qlearning sarsa --ckpts 0 10 30 60 100 --seed 123
결과: results/<tag>/videos/<algo>_p<ckpt>.gif, random.gif, heuristic.gif
같은 --seed 로 만들기 때문에 시점별 행동 차이를 나란히 비교하기 좋다.
"""
import argparse
import subprocess
import sys


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=False)


def main(a):
    base = [sys.executable, "-m", "scripts.visualize", "--tag", a.tag, "--seed", str(a.seed),
            "--delay", str(a.delay)]
    out = f"results/{a.tag}/videos"
    run(base + ["--policy", "random", "--record", f"{out}/random.gif"])
    run(base + ["--policy", "heuristic", "--record", f"{out}/heuristic.gif"])
    for algo in a.algos:
        for c in a.ckpts:
            run(base + ["--policy", "q", "--algo", algo, "--ckpt", c, "--record", f"{out}/{algo}_p{c}.gif"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base")
    p.add_argument("--algos", nargs="+", default=["qlearning"])
    p.add_argument("--ckpts", nargs="+", default=["0", "10", "30", "60", "100"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--delay", type=int, default=500)
    main(p.parse_args())
