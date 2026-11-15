"""
알고리즘 비교 (학습 곡선, 탐험 없는 평가, 최종 성능 표)
 - 단일 실행 비교:   python -m scripts.compare --tag base
 - 다중 시드 비교:   python -m scripts.compare --tag base --multi      (run_seeds.py 로 만든 results/<tag>_s*/ 사용)
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agents.baselines import evaluate, random_policy, heuristic_policy
from scripts.train import EVAL_SEED0

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac: AppleGothic
plt.rcParams["axes.unicode_minus"] = False
ALGOS = ["qlearning", "sarsa", "mc"]


def baselines(n=200):
    rnd = evaluate(random_policy, n_episodes=n, seed=EVAL_SEED0)[0]
    heu = evaluate(heuristic_policy, n_episodes=n, seed=EVAL_SEED0)[0]
    return rnd, heu


def first_reach(episodes, values, threshold):
    idx = np.flatnonzero(np.asarray(values) >= threshold)
    return int(np.asarray(episodes)[idx[0]]) if len(idx) else None


# ---------------- 단일 실행 ----------------
def single(a):
    d = f"results/{a.tag}"
    logs = {k: pd.read_csv(f"{d}/{k}_log.csv") for k in ALGOS if os.path.exists(f"{d}/{k}_log.csv")}
    if not logs:
        raise SystemExit(f"{d} 에 학습 결과가 없습니다. 먼저 train.py를 실행하세요.")
    evals = {k: pd.read_csv(f"{d}/{k}_eval.csv") for k in logs if os.path.exists(f"{d}/{k}_eval.csv")}
    os.makedirs(f"{d}/figs", exist_ok=True)
    rnd, heu = baselines()

    fig, axes = plt.subplots(1, 2 if evals else 1, figsize=(13 if evals else 7, 4.5), squeeze=False)
    ax = axes[0][0]
    for k, df in logs.items():
        ax.plot(df.days.rolling(a.window).mean(), label=k)
    ax.axhline(rnd, ls="--", c="gray", label="random"); ax.axhline(heu, ls="--", c="green", label="heuristic")
    ax.set_title("학습 중 생존일 (탐험 포함, 이동평균)"); ax.set_xlabel("episode"); ax.set_ylabel("평균 생존 일수"); ax.legend()
    if evals:
        ax = axes[0][1]
        for k, df in evals.items():
            ax.plot(df.episode, df.eval_days_mean, label=k)
        ax.axhline(rnd, ls="--", c="gray"); ax.axhline(heu, ls="--", c="green")
        ax.set_title("탐험 없는 평가 (greedy, 고정 시드)"); ax.set_xlabel("episode"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{d}/figs/compare_curve.png", dpi=150); plt.close()

    rows = []
    for k, df in logs.items():
        last = df.tail(a.last)
        row = dict(algo=k, episodes=len(df), train_avg_days_last=round(last.days.mean(), 2))
        if k in evals:
            ev = evals[k]; fin = ev.iloc[-1]
            row.update(eval_days_final=round(fin.eval_days_mean, 2), eval_death_rate=round(fin.eval_death_rate, 3),
                       ep_to_beat_heuristic=first_reach(ev.episode, ev.eval_days_mean, heu))
        rows.append(row)
    table = pd.DataFrame(rows); table.to_csv(f"{d}/compare_table.csv", index=False)
    print(f"[기준선] random={rnd:.2f}일, heuristic={heu:.2f}일"); print(table.to_string(index=False))
    print(f"그래프: {d}/figs/compare_curve.png, 표: {d}/compare_table.csv")


# ---------------- 다중 시드 ----------------
def multi(a):
    dirs = sorted(glob.glob(f"results/{a.tag}_s*"))
    if not dirs:
        raise SystemExit(f"results/{a.tag}_s* 가 없습니다. 먼저 run_seeds.py를 실행하세요.")
    out = f"results/{a.tag}_summary"; os.makedirs(out, exist_ok=True)
    rnd, heu = baselines()

    plt.figure(figsize=(8.5, 5))
    rows = []
    for k in ALGOS:
        per_seed = [pd.read_csv(f"{d}/{k}_eval.csv") for d in dirs if os.path.exists(f"{d}/{k}_eval.csv")]
        if not per_seed:
            continue
        allv = pd.concat([df.assign(seed=i) for i, df in enumerate(per_seed)])
        g = allv.groupby("episode").eval_days_mean.agg(["mean", "std"]).fillna(0)
        plt.plot(g.index, g["mean"], label=f"{k} (n={len(per_seed)})")
        plt.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)
        finals = np.array([df.eval_days_mean.iloc[-1] for df in per_seed])
        deaths = np.array([df.eval_death_rate.iloc[-1] for df in per_seed])
        reach = [first_reach(df.episode, df.eval_days_mean, heu) for df in per_seed]
        reached = [r for r in reach if r is not None]
        rows.append(dict(algo=k, n_seeds=len(per_seed),
                         final_eval_days=f"{finals.mean():.2f} ± {finals.std():.2f}",
                         final_death_rate=f"{deaths.mean():.2f}",
                         seeds_beating_heuristic=f"{len(reached)}/{len(per_seed)}",
                         median_ep_to_beat=int(np.median(reached)) if reached else None))
    plt.axhline(rnd, ls="--", c="gray", label="random"); plt.axhline(heu, ls="--", c="green", label="heuristic")
    plt.xlabel("episode"); plt.ylabel("탐험 없는 평가: 평균 생존 일수")
    plt.title("알고리즘 비교 (시드 평균 ± 표준편차)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/compare_multi_curve.png", dpi=150); plt.close()

    table = pd.DataFrame(rows); table.to_csv(f"{out}/compare_multi_table.csv", index=False)
    print(f"[기준선] random={rnd:.2f}일, heuristic={heu:.2f}일  (시드 디렉터리 {len(dirs)}개)")
    print(table.to_string(index=False))
    print(f"그래프: {out}/compare_multi_curve.png, 표: {out}/compare_multi_table.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base")
    p.add_argument("--multi", action="store_true", help="results/<tag>_s* 의 여러 시드 결과를 합쳐 비교")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--last", type=int, default=1000)
    a = p.parse_args()
    multi(a) if a.multi else single(a)