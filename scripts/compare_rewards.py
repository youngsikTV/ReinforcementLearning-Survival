"""
보상 구조 비교: 서로 다른 보상 설정으로 학습한 여러 tag 를 '생존 일수'(공통 지표)로 비교
(보상 크기가 설정마다 다르므로 총 보상끼리 비교하면 안 된다 -> 탐험 없는 평가의 생존일 사용)

사용 예:
  python -m scripts.run_seeds --tag rw_base   --algos qlearning --seeds 0 1 2
  python -m scripts.run_seeds --tag rw_step   --algos qlearning --seeds 0 1 2 --day_bonus 0
  python -m scripts.run_seeds --tag rw_bonus  --algos qlearning --seeds 0 1 2 --day_bonus 20
  python -m scripts.run_seeds --tag rw_grow   --algos qlearning --seeds 0 1 2 --step_growth 0.1
  python -m scripts.compare_rewards --tags rw_base rw_step rw_bonus rw_grow --algo qlearning --name reward_study
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.compare import baselines, first_reach

plt.rcParams["font.family"] = "Malgun Gothic"   # Mac: AppleGothic
plt.rcParams["axes.unicode_minus"] = False
CAUSE_KEYS = ["hunger", "exhaustion", "monster", "survived"]


def main(a):
    out = f"results/{a.name}"; os.makedirs(out, exist_ok=True)
    rnd, heu = baselines()
    plt.figure(figsize=(8.5, 5))
    rows = []
    for tag in a.tags:
        dirs = sorted(glob.glob(f"results/{tag}_s*"))
        evs = [pd.read_csv(f"{d}/{a.algo}_eval.csv") for d in dirs if os.path.exists(f"{d}/{a.algo}_eval.csv")]
        if not evs:
            print(f"[건너뜀] {tag}: results/{tag}_s*/{a.algo}_eval.csv 가 없습니다."); continue
        g = pd.concat(evs).groupby("episode").eval_days_mean.agg(["mean", "std"]).fillna(0)
        plt.plot(g.index, g["mean"], label=f"{tag} (n={len(evs)})")
        plt.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2)
        finals = np.array([e.eval_days_mean.iloc[-1] for e in evs])
        reach = [r for r in (first_reach(e.episode, e.eval_days_mean, heu) for e in evs) if r is not None]
        row = dict(tag=tag, n_seeds=len(evs), final_eval_days=f"{finals.mean():.2f} ± {finals.std():.2f}",
                   seeds_beating_heuristic=f"{len(reach)}/{len(evs)}",
                   median_ep_to_beat=int(np.median(reach)) if reach else None)
        # 학습 후반 10% 구간의 종료 원인 비율 (시드 평균)
        causes = []
        for d in dirs:
            lp = f"{d}/{a.algo}_log.csv"
            if os.path.exists(lp):
                lg = pd.read_csv(lp)
                if "death_cause" in lg.columns:
                    tail = lg.tail(max(1, len(lg) // 10))
                    causes.append([(tail.death_cause == k).mean() for k in CAUSE_KEYS])
        if causes:
            for k, v in zip(CAUSE_KEYS, np.mean(causes, axis=0)):
                row[f"late_{k}"] = round(float(v), 2)
        rows.append(row)
    plt.axhline(rnd, ls="--", c="gray", label="random"); plt.axhline(heu, ls="--", c="green", label="heuristic")
    plt.xlabel("episode"); plt.ylabel("탐험 없는 평가: 평균 생존 일수")
    plt.title(f"보상 구조 비교 ({a.algo}, 시드 평균 ± 표준편차)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{out}/reward_compare_curve.png", dpi=150); plt.close()
    table = pd.DataFrame(rows); table.to_csv(f"{out}/reward_compare_table.csv", index=False)
    print(f"[기준선] random={rnd:.2f}일, heuristic={heu:.2f}일"); print(table.to_string(index=False))
    print(f"그래프: {out}/reward_compare_curve.png, 표: {out}/reward_compare_table.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True, help="run_seeds 에서 사용한 tag 들 (뒤의 _s<seed>는 제외)")
    p.add_argument("--algo", default="qlearning")
    p.add_argument("--name", default="reward_study", help="결과 저장 폴더 이름 (results/<name>/)")
    main(p.parse_args())
