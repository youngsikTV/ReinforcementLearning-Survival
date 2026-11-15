"""비교 기준(baseline) 정책: 학습한 에이전트가 정말 똑똑한지 확인하는 용도"""
import numpy as np
from envs.survival_env import MinecraftSurvivalEnv, REST, EAT, HUNT, WOOD, SHELTER


def random_policy(env, rng):
    return int(rng.integers(5))


def heuristic_policy(env, rng):
    """사람이 짠 규칙 기반 정책 (최소한 이 정도는 넘어야 '학습했다'고 말할 수 있음)"""
    if env.hunger <= 2 and env.food > 0:
        return EAT
    if env.energy <= 1:
        return REST
    if env.phase in (2, 3) and not env.shelter:   # 해지기 전 쉼터 준비
        return SHELTER if env.wood >= 1 else WOOD
    if env.food < 2:
        return HUNT
    return REST


def evaluate(policy, n_episodes=200, seed=0, **env_kwargs):
    env = MinecraftSurvivalEnv(**env_kwargs)
    rng = np.random.default_rng(seed)
    days = []
    for i in range(n_episodes):
        env.reset(seed=seed + i)
        done = False
        while not done:
            a = policy(env, rng)
            _, _, term, trunc, info = env.step(a)
            done = term or trunc
        days.append(info["day"])
    return float(np.mean(days)), float(np.std(days))


if __name__ == "__main__":
    print("random    (평균 생존일, 표준편차):", evaluate(random_policy))
    print("heuristic (평균 생존일, 표준편차):", evaluate(heuristic_policy))
