"""사용법:  python -m scripts.train --algo qlearning --episodes 100000 --tag base
- 학습 진행률 0/10/30/60/100% 시점의 Q-table을 results/<tag>/ckpt/ 에 저장 (--ckpt)
- --eval_every 에피소드마다 '탐험 없이(greedy)' 고정 시드로 평가해서 results/<tag>/<algo>_eval.csv 에 기록
"""
import argparse, os, json
import numpy as np
import pandas as pd
from envs.survival_env import MinecraftSurvivalEnv, SLOTS_PER_DAY, EAT
from agents.tabular import TabularAgent

EVAL_SEED0 = 10_000_000     # 평가용 시드 (학습 시드와 겹치지 않게 분리, 모든 알고리즘/시점에서 동일)


def greedy_eval(Q, n_episodes, max_days):
    """탐험 없이 argmax 정책으로 n_episodes 판을 돌려 평균 생존일 / 표준편차 / 사망률 반환"""
    env = MinecraftSurvivalEnv(max_days=max_days)
    rng = np.random.default_rng(0)               # 동점(방문 안 한 상태) 처리용 별도 난수
    days, died = [], []
    for i in range(n_episodes):
        s, _ = env.reset(seed=EVAL_SEED0 + i)
        done = False
        while not done:
            q = Q[s]
            a = int(rng.choice(np.flatnonzero(q == q.max())))
            s, _, term, trunc, info = env.step(a)
            done = term or trunc
        days.append(info["day"])
        died.append(int(term))
    return float(np.mean(days)), float(np.std(days)), float(np.mean(died))


def run(args):
    reward_cfg = dict(step=args.step_reward, day_bonus=args.day_bonus, death=args.death_penalty,
                      invalid=args.invalid_penalty, step_growth=args.step_growth, day_growth=args.day_growth)
    env = MinecraftSurvivalEnv(max_days=args.max_days, reward_cfg=reward_cfg)
    agent = TabularAgent(env.n_states, env.action_space.n, algo=args.algo, alpha=args.alpha,
                         gamma=args.gamma, eps_decay_episodes=int(args.episodes * 0.6), seed=args.seed)

    out = os.path.join("results", args.tag)
    ckpt_dir = os.path.join(out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_at = {}
    for pc in [int(x) for x in args.ckpt.split(",") if x.strip()]:
        if pc <= 0:
            np.save(f"{ckpt_dir}/{args.algo}_Q_p0.npy", agent.Q.copy())
        else:
            ckpt_at.setdefault(max(1, int(args.episodes * pc / 100)), []).append(pc)

    eval_rows = []
    def do_eval(ep_count):
        m, sd, dr = greedy_eval(agent.Q, args.eval_episodes, args.max_days)
        eval_rows.append(dict(episode=ep_count, eval_days_mean=m, eval_days_std=sd, eval_death_rate=dr))
        return m, dr

    do_eval(0)                                     # 학습 전 성능
    logs, action_log = [], []
    for ep in range(args.episodes):
        eps = agent.epsilon(ep)
        s, info = env.reset(seed=args.seed * 100000 + ep)
        a = agent.act(s, eps)
        traj, total = [], 0.0
        counts = np.zeros((SLOTS_PER_DAY, env.action_space.n), dtype=int)
        # 행동 지표: 식사 횟수, 무의미한 행동, 밤 대비(쉼터), 식사 시점 배고픔
        n_steps = eat_cnt = invalid_cnt = nights = shelter_nights = 0
        hunger_at_eat = []
        done = False
        while not done:
            phase = info["phase"]
            if a == EAT and info["food"] > 0:
                eat_cnt += 1
                hunger_at_eat.append(info["hunger"])
            s2, r, term, trunc, info = env.step(a)
            n_steps += 1
            invalid_cnt += int(info["invalid"])
            if info["phase"] == 4 and not term:      # 밤이 막 시작된 시점에 쉼터가 있는가?
                nights += 1
                shelter_nights += int(info["shelter"])
            done = term or trunc
            a2 = agent.act(s2, eps)
            counts[phase, a] += 1
            if args.algo == "mc":
                traj.append((s, a, r))
            else:
                agent.update_td(s, a, r, s2, a2, term)
            total += r
            s, a = s2, a2
        if args.algo == "mc":
            agent.update_mc(traj)
        logs.append(dict(episode=ep, days=info["day"], total_reward=total, eps=eps, died=int(term),
                         death_cause=info["death_cause"] if term else "survived", steps=n_steps,
                         eat_actions=eat_cnt, invalid_actions=invalid_cnt, nights=nights,
                         shelter_nights=shelter_nights,
                         hunger_at_eat=float(np.mean(hunger_at_eat)) if hunger_at_eat else np.nan))
        action_log.append(counts)

        for pc in ckpt_at.get(ep + 1, []):
            np.save(f"{ckpt_dir}/{args.algo}_Q_p{pc}.npy", agent.Q.copy())
            print(f"[{args.algo}] 체크포인트 저장: 진행률 {pc}% (episode {ep+1})")

        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            m, dr = do_eval(ep + 1)
            recent = pd.DataFrame(logs[-args.eval_every:])
            print(f"[{args.algo}|seed{args.seed}] ep {ep+1}  train_days={recent.days.mean():.2f}  "
                  f"eval_days(greedy)={m:.2f}  eval_death={dr:.2f}  eps={eps:.2f}")

    pd.DataFrame(logs).to_csv(f"{out}/{args.algo}_log.csv", index=False)
    pd.DataFrame(eval_rows).to_csv(f"{out}/{args.algo}_eval.csv", index=False)
    np.save(f"{out}/{args.algo}_actions.npy", np.array(action_log))
    np.save(f"{out}/{args.algo}_Q.npy", agent.Q)
    json.dump(vars(args), open(f"{out}/{args.algo}_args.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["qlearning", "sarsa", "mc"], default="qlearning")
    p.add_argument("--episodes", type=int, default=100000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--max_days", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="base")
    p.add_argument("--ckpt", default="0,10,30,60,100", help="체크포인트 저장 시점 (학습 진행률 %%, 쉼표 구분)")
    # --- 보상 설정 (보상 구조 비교 실험용) ---
    p.add_argument("--step_reward", type=float, default=1.0, help="매 step 기본 보상")
    p.add_argument("--day_bonus", type=float, default=5.0, help="하루 생존 보너스")
    p.add_argument("--death_penalty", type=float, default=-20.0)
    p.add_argument("--invalid_penalty", type=float, default=-0.5)
    p.add_argument("--step_growth", type=float, default=0.0, help="step 보상 증가율 (step*(1+g*day))")
    p.add_argument("--day_growth", type=float, default=0.0, help="하루 보너스 증가율 (bonus*(1+g*day))")
    p.add_argument("--eval_every", type=int, default=1000, help="탐험 없는 평가 주기(에피소드)")
    p.add_argument("--eval_episodes", type=int, default=100, help="평가 때 돌리는 판 수")
    run(p.parse_args())