"""사용법:  python -m scripts.analyze --tag base --algo qlearning
저장 위치: results/<tag>/figs/
 1) <algo>_curve.png     학습 곡선 + 기준선
 2) <algo>_strategy.png  학습 단계(초기/중기/후기)별 시간대 x 행동 분포
 3) <algo>_death.png     학습 단계별 사망 원인 비율 (굶주림 / 기력 고갈 / 몬스터 / 생존)
 4) <algo>_behavior.png  학습 단계별 행동 지표 (밤 전 쉼터 준비율, 하루당 식사, 식사 시점 배고픔, 무의미한 행동 비율)
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from envs.survival_env import ACTION_NAMES, PHASE_NAMES
from agents.baselines import evaluate, random_policy, heuristic_policy

plt.rcParams["font.family"] = "Malgun Gothic"   # Windows 한글 폰트 (Mac: AppleGothic)
plt.rcParams["axes.unicode_minus"] = False
CAUSES = [("hunger", "굶주림"), ("exhaustion", "기력 고갈"), ("monster", "몬스터 습격"), ("survived", "목표 달성(생존)")]


def stage_chunks(df, n_stage):
    """에피소드를 n_stage 개 구간으로 나눠 (구간 이름, 데이터프레임) 리스트 반환"""
    size = max(1, len(df) // n_stage)
    out = []
    for i in range(n_stage):
        c = df.iloc[i * size: (i + 1) * size if i < n_stage - 1 else len(df)]
        out.append((f"{int(i * 100 / n_stage)}~{int((i + 1) * 100 / n_stage)}%", c))
    return out


def main(a):
    d = f"results/{a.tag}"
    log = pd.read_csv(f"{d}/{a.algo}_log.csv")
    acts = np.load(f"{d}/{a.algo}_actions.npy")          # (episodes, phase, action)
    os.makedirs(f"{d}/figs", exist_ok=True)

    # 1) 학습 곡선 + baseline
    rnd, heu = evaluate(random_policy)[0], evaluate(heuristic_policy)[0]
    plt.figure(figsize=(7, 4))
    plt.plot(log.days.rolling(500).mean(), label=a.algo)
    plt.axhline(rnd, ls="--", c="gray", label="random")
    plt.axhline(heu, ls="--", c="green", label="heuristic")
    plt.xlabel("episode"); plt.ylabel("생존 일수 (이동평균 500)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{d}/figs/{a.algo}_curve.png", dpi=150); plt.close()

    # 2) 학습 단계별 행동 분포 (초기 / 중기 / 후기) -> 전략 변화 관찰
    n = len(acts)
    chunks = {"초기": acts[: n // 10],
              "중기": acts[n // 2 - n // 20: n // 2 + n // 20],
              "후기": acts[-(n // 10):]}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (name, c) in zip(axes, chunks.items()):
        m = c.sum(axis=0).astype(float)
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)       # 시간대별 행동 비율
        im = ax.imshow(m, vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_xticks(range(len(ACTION_NAMES)))
        ax.set_xticklabels([x.split("(")[0] for x in ACTION_NAMES], rotation=30)
        ax.set_yticks(range(len(PHASE_NAMES))); ax.set_yticklabels(PHASE_NAMES)
        ax.set_title(f"{name} 학습")
    fig.colorbar(im, ax=axes, shrink=0.8)
    plt.savefig(f"{d}/figs/{a.algo}_strategy.png", dpi=150); plt.close()

    # 3), 4) 새 로그 컬럼이 있을 때만 (이전 버전으로 학습한 결과에는 없음)
    if "death_cause" not in log.columns:
        print("사망 원인/행동 지표 컬럼이 없습니다. 최신 train.py로 다시 학습하면 3), 4)번 그래프가 생성됩니다.")
        print("저장 완료:", f"{d}/figs/"); return

    stages = stage_chunks(log, a.stages)
    names = [s for s, _ in stages]

    # 3) 사망 원인 비율 (누적 막대)
    ratios = np.array([[ (c.death_cause == k).mean() for k, _ in CAUSES] for _, c in stages])
    plt.figure(figsize=(9, 4.5))
    bottom = np.zeros(len(stages))
    for j, (k, label) in enumerate(CAUSES):
        plt.bar(names, ratios[:, j], bottom=bottom, label=label)
        bottom += ratios[:, j]
    plt.ylabel("비율"); plt.xlabel("학습 진행 구간"); plt.title("학습 단계별 종료 원인")
    plt.xticks(rotation=45); plt.legend(loc="upper left", bbox_to_anchor=(1, 1)); plt.tight_layout()
    plt.savefig(f"{d}/figs/{a.algo}_death.png", dpi=150); plt.close()

    # 4) 행동 지표
    def agg(c):
        steps = max(c.steps.sum(), 1)
        return dict(
            shelter_ratio=c.shelter_nights.sum() / max(c.nights.sum(), 1),
            eat_per_day=c.eat_actions.sum() / (steps / 6.0),
            hunger_at_eat=c.hunger_at_eat.mean(),
            invalid_rate=c.invalid_actions.sum() / steps)
    m = pd.DataFrame([agg(c) for _, c in stages])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (col, title) in zip(axes.ravel(), [("shelter_ratio", "밤 시작 시 쉼터가 있는 비율"),
                                               ("eat_per_day", "하루(6 step)당 식사 횟수"),
                                               ("hunger_at_eat", "식사할 때의 평균 배고픔 수치 (낮을수록 늦게 먹음)"),
                                               ("invalid_rate", "무의미한 행동 비율")]):
        ax.plot(names, m[col], marker="o"); ax.set_title(title); ax.tick_params(axis="x", rotation=45)
    plt.tight_layout(); plt.savefig(f"{d}/figs/{a.algo}_behavior.png", dpi=150); plt.close()
    m.insert(0, "stage", names); m.to_csv(f"{d}/{a.algo}_behavior.csv", index=False)
    print(m.round(3).to_string(index=False))
    print("저장 완료:", f"{d}/figs/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="base"); p.add_argument("--algo", default="qlearning")
    p.add_argument("--stages", type=int, default=10, help="학습을 몇 구간으로 나눠 볼지")
    main(p.parse_args())