"""수업 시간에 배운 Tabular 알고리즘 (Q-learning, SARSA, Monte Carlo)"""
import numpy as np


class TabularAgent:
    def __init__(self, n_states, n_actions, algo="qlearning", alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=3000, seed=0):
        self.Q = np.zeros((n_states, n_actions))
        self.N = np.zeros((n_states, n_actions))   # Monte Carlo용 방문 횟수
        self.algo, self.alpha, self.gamma = algo, alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_episodes
        self.nA = n_actions
        self.rng = np.random.default_rng(seed)

    def epsilon(self, ep):
        frac = min(1.0, ep / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, s, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.nA))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))   # 동점이면 무작위

    # --- TD 업데이트 (Q-learning / SARSA) ---
    def update_td(self, s, a, r, s2, a2, terminated):
        if self.algo == "qlearning":
            target = r + (0 if terminated else self.gamma * self.Q[s2].max())
        else:  # sarsa
            target = r + (0 if terminated else self.gamma * self.Q[s2, a2])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    # --- Monte Carlo (first-visit) : 에피소드 종료 후 호출 ---
    def update_mc(self, traj):
        """traj = [(s, a, r), ...]"""
        G, returns = 0.0, []
        for s, a, r in reversed(traj):
            G = r + self.gamma * G
            returns.append((s, a, G))
        seen = set()
        for s, a, G in reversed(returns):       # 시간 순서대로 first-visit
            if (s, a) in seen:
                continue
            seen.add((s, a))
            self.N[s, a] += 1
            self.Q[s, a] += (G - self.Q[s, a]) / self.N[s, a]
