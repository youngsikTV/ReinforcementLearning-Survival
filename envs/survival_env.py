"""
Minecraft-style Survival Environment (Gymnasium 호환, 순수 Python/Numpy)
- 1 step = 하루의 한 시간대(slot). 하루 = 6 slots (아침~늦은밤)
- 목표: 최대한 많은 날(day)을 생존
- 상태는 이산(Discrete)이라 Q-learning / SARSA / Monte Carlo 같은 수업 알고리즘을 바로 적용 가능

보상 설정(reward_cfg):
  step         : 매 step 기본 보상
  day_bonus    : 하루를 버틸 때마다 받는 보너스
  death        : 사망 시 보상(보통 음수)
  invalid      : 의미 없는 행동(음식 없이 식사, 쉼터 중복 건설 등) 페널티
  step_growth  : step 보상 증가율 -> step * (1 + step_growth * day)   (오래 살수록 step 보상 증가)
  day_growth   : 보너스 증가율  -> day_bonus * (1 + day_growth * day)
step 반환 info 에는 상태와 함께 invalid(bool), death_cause(사망 시: monster/hunger/exhaustion) 가 포함된다.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

REST, EAT, HUNT, WOOD, SHELTER = range(5)
ACTION_NAMES = ["REST(휴식)", "EAT(식사)", "HUNT(사냥/채집)", "WOOD(목재)", "SHELTER(쉼터 건설)"]
PHASE_NAMES = ["아침", "점심", "오후", "저녁", "밤", "늦은밤"]
SLOTS_PER_DAY = 6
NIGHT_PHASES = (4, 5)

MAX_STAT = 5      # hunger / health / energy : 0~5
MAX_FOOD = 3      # 음식 인벤토리 : 0~3
MAX_WOOD = 2      # 목재 인벤토리 : 0~2
# 상태 = (hunger, health, energy, food, wood, phase, shelter)
DIMS = (MAX_STAT + 1, MAX_STAT + 1, MAX_STAT + 1, MAX_FOOD + 1, MAX_WOOD + 1, SLOTS_PER_DAY, 2)

DEFAULT_REWARD = dict(step=1.0, day_bonus=5.0, death=-20.0, invalid=-0.5, step_growth=0.0, day_growth=0.0)


class MinecraftSurvivalEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_days=30, reward_cfg=None, render_mode=None):
        super().__init__()
        self.max_days = max_days
        self.render_mode = render_mode
        self.cfg = dict(DEFAULT_REWARD)
        if reward_cfg:
            self.cfg.update(reward_cfg)

        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.n_states = int(np.prod(DIMS))
        self.observation_space = spaces.Discrete(self.n_states)

    # ---------- 상태 인코딩 ----------
    def _encode(self):
        return int(np.ravel_multi_index(
            (self.hunger, self.health, self.energy, self.food, self.wood, self.phase, self.shelter), DIMS))

    @staticmethod
    def decode(s):
        return tuple(int(x) for x in np.unravel_index(s, DIMS))

    def _info(self):
        return dict(day=self.day, hunger=self.hunger, health=self.health, energy=self.energy,
                    food=self.food, wood=self.wood, phase=self.phase, shelter=self.shelter)

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.hunger, self.health, self.energy = 5, 5, 5
        self.food, self.wood = 1, 0
        self.phase, self.shelter, self.day = 0, 0, 0
        return self._encode(), self._info()

    def step(self, action):
        rng = self.np_random
        r = self.cfg["step"] * (1.0 + self.cfg["step_growth"] * self.day)
        night = self.phase in NIGHT_PHASES
        invalid = False

        # 1) 행동 효과
        if action == REST:
            self.energy += 3 if (self.shelter and night) else 2
        elif action == EAT:
            if self.food > 0:
                self.food -= 1
                self.hunger += 3
            else:
                invalid = True
        elif action == HUNT:
            self.energy -= 1
            if rng.random() < (0.3 if night else 0.7):
                self.food += 1
        elif action == WOOD:
            self.energy -= 1
            if rng.random() < 0.8:
                self.wood += 1
        elif action == SHELTER:
            if self.wood >= 1 and not self.shelter:
                self.wood -= 1
                self.shelter = 1
                self.energy -= 1
            else:
                invalid = True
        if invalid:
            r += self.cfg["invalid"]

        # 2) 자연 감소 (배고픔)
        if rng.random() < 0.5:
            self.hunger -= 1

        # 3) 피해 계산 (원인별로 기록 -> 사망 원인 분석용)
        dmg = {"monster": 0, "hunger": 0, "exhaustion": 0}
        if night and rng.random() < (0.05 if self.shelter else 0.5):
            dmg["monster"] = 2                  # 밤에는 몬스터 습격 (쉼터가 있으면 거의 안전)

        # 4) 능력치 범위 제한 후 체력 변화
        self.hunger = int(np.clip(self.hunger, 0, MAX_STAT))
        self.energy = int(np.clip(self.energy, 0, MAX_STAT))
        self.food = int(np.clip(self.food, 0, MAX_FOOD))
        self.wood = int(np.clip(self.wood, 0, MAX_WOOD))
        if self.hunger == 0:
            dmg["hunger"] = 1
        if self.energy == 0:
            dmg["exhaustion"] = 1
        self.health -= sum(dmg.values())
        if self.hunger >= 4 and self.energy >= 2 and rng.random() < 0.3:
            self.health += 1
        self.health = int(np.clip(self.health, 0, MAX_STAT))

        # 5) 종료 판정 / 시간 진행
        terminated = self.health <= 0
        truncated = False
        cause = None
        if terminated:
            r += self.cfg["death"]
            cause = max(dmg, key=dmg.get)       # 마지막 step에서 가장 큰 피해 원인
        else:
            self.phase = (self.phase + 1) % SLOTS_PER_DAY
            if self.phase == 0:                 # 새 하루 시작
                self.day += 1
                self.shelter = 0                # 쉼터는 하루만 유지
                r += self.cfg["day_bonus"] * (1.0 + self.cfg["day_growth"] * self.day)
                truncated = self.day >= self.max_days

        info = self._info()
        info["invalid"] = invalid
        info["death_cause"] = cause
        return self._encode(), float(r), terminated, truncated, info

    def render(self):
        i = self._info()
        txt = (f"Day {i['day']:>2} [{PHASE_NAMES[i['phase']]}] "
               f"HP:{i['health']} 배고픔:{i['hunger']} 체력:{i['energy']} "
               f"음식:{i['food']} 목재:{i['wood']} 쉼터:{'O' if i['shelter'] else 'X'}")
        if self.render_mode == "ansi":
            return txt
        print(txt)