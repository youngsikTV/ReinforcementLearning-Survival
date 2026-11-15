"""학습된 Q-table로 한 에피소드를 텍스트로 시연:  python -m scripts.play --tag base --algo qlearning"""
import argparse
import numpy as np
from envs.survival_env import MinecraftSurvivalEnv, ACTION_NAMES

p = argparse.ArgumentParser()
p.add_argument("--tag", default="base"); p.add_argument("--algo", default="qlearning")
a = p.parse_args()
Q = np.load(f"results/{a.tag}/{a.algo}_Q.npy")
env = MinecraftSurvivalEnv(render_mode="ansi")
s, _ = env.reset(seed=123)
done = False
while not done:
    act = int(np.argmax(Q[s]))
    print(env.render(), "->", ACTION_NAMES[act])
    s, r, term, trunc, _ = env.step(act)
    done = term or trunc
print(env.render(), "| 종료(사망)" if term else "| 목표 일수 달성")
