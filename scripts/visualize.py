"""
pygame 시각화: 학습된 에이전트(또는 규칙 기반 정책)가 생존하는 모습을 보여줌
사용법:
  python -m scripts.visualize --policy q --tag base --algo qlearning              # 최종 학습 결과
  python -m scripts.visualize --policy q --tag base --algo qlearning --ckpt 30    # 학습 30% 시점
  python -m scripts.visualize --policy random                                     # 무작위
  python -m scripts.visualize --policy heuristic                                  # 규칙 기반
  python -m scripts.visualize --policy q --tag base --algo sarsa --ckpt 100 --record results/base/videos/sarsa_p100.gif
조작(창 모드): SPACE 일시정지 / R 새 에피소드 / UP, DOWN 속도 조절 / ESC 종료
--record 를 주면 창을 띄우지 않고 한 판을 GIF로 저장 (같은 --seed 로 시점별 영상을 만들면 비교하기 좋음)
"""
import argparse
import os
import numpy as np
import pygame
from envs.survival_env import (MinecraftSurvivalEnv, ACTION_NAMES, PHASE_NAMES, NIGHT_PHASES,
                               REST, EAT, HUNT, WOOD, SHELTER, MAX_STAT, MAX_FOOD, MAX_WOOD)
from agents.baselines import heuristic_policy

W, H = 720, 460
SKY = {0: (135, 206, 235), 1: (110, 190, 245), 2: (240, 170, 110), 3: (200, 110, 90),
       4: (25, 30, 70), 5: (15, 18, 45)}
ACTION_COLOR = {REST: (120, 120, 220), EAT: (230, 120, 60), HUNT: (200, 60, 60),
                WOOD: (140, 95, 50), SHELTER: (160, 160, 160)}


def q_path(args):
    if args.ckpt == "final":
        return f"results/{args.tag}/{args.algo}_Q.npy"
    return f"results/{args.tag}/ckpt/{args.algo}_Q_p{args.ckpt}.npy"


def make_label(args):
    if args.policy == "q":
        when = "최종" if args.ckpt == "final" else f"학습 {args.ckpt}%"
        return f"{args.algo} ({when})"
    return args.policy


def make_policy(args, env):
    rng = np.random.default_rng(args.seed)
    if args.policy == "q":
        path = q_path(args)
        if not os.path.exists(path):
            raise SystemExit(f"파일이 없습니다: {path}\n먼저 train.py로 학습하세요 (체크포인트는 --ckpt 값과 학습 때 저장한 진행률이 같아야 함).")
        Q = np.load(path)
        return lambda s: int(np.argmax(Q[s]))
    if args.policy == "heuristic":
        return lambda s: heuristic_policy(env, rng)
    return lambda s: int(rng.integers(5))


def bar(screen, font, x, y, label, value, vmax, color):
    screen.blit(font.render(f"{label} {value}/{vmax}", True, (255, 255, 255)), (x, y))
    pygame.draw.rect(screen, (50, 50, 50), (x, y + 22, 160, 14))
    pygame.draw.rect(screen, color, (x, y + 22, int(160 * value / vmax), 14))
    pygame.draw.rect(screen, (255, 255, 255), (x, y + 22, 160, 14), 1)


def draw(screen, fonts, env, last_action, reward, total, label):
    f, fb = fonts
    i = env._info()
    phase = i["phase"]
    screen.fill(SKY[phase])
    night = phase in NIGHT_PHASES
    pygame.draw.circle(screen, (240, 240, 200) if night else (255, 220, 60), (60 + phase * 110, 60), 24)
    pygame.draw.rect(screen, (70, 140, 60), (0, 330, W, H - 330))
    pygame.draw.rect(screen, (110, 80, 50), (0, 380, W, H - 380))
    if i["shelter"]:
        pygame.draw.rect(screen, (150, 110, 70), (420, 270, 120, 60))
        pygame.draw.polygon(screen, (110, 70, 40), [(410, 270), (480, 225), (550, 270)])
        pygame.draw.rect(screen, (40, 25, 15), (470, 295, 24, 35))
    col = ACTION_COLOR.get(last_action, (60, 120, 200))
    pygame.draw.rect(screen, col, (300, 285, 40, 45))
    pygame.draw.rect(screen, (240, 200, 160), (304, 260, 32, 28))
    if night:
        shade = pygame.Surface((W, 330), pygame.SRCALPHA)
        shade.fill((0, 0, 30, 70))
        screen.blit(shade, (0, 0))
    screen.blit(fb.render(f"Day {i['day']}  -  {PHASE_NAMES[phase]}", True, (255, 255, 255)), (W - 230, 15))
    screen.blit(f.render(label, True, (255, 255, 255)), (W - 230, 50))
    bar(screen, f, 20, 110, "HP(체력)", i["health"], MAX_STAT, (220, 50, 50))
    bar(screen, f, 20, 155, "배고픔", i["hunger"], MAX_STAT, (230, 150, 40))
    bar(screen, f, 20, 200, "기력", i["energy"], MAX_STAT, (80, 160, 230))
    screen.blit(f.render(f"음식 {i['food']}/{MAX_FOOD}   목재 {i['wood']}/{MAX_WOOD}", True, (255, 255, 255)), (20, 250))
    act_txt = ACTION_NAMES[last_action] if last_action is not None else "-"
    screen.blit(fb.render(f"행동: {act_txt}", True, (255, 255, 255)), (20, 400))
    screen.blit(f.render(f"보상 {reward:+.1f}  누적 {total:.1f}", True, (255, 255, 255)), (20, 435))


def make_fonts():
    fam = "malgungothic,applegothic,nanumgothic,notosanscjkkr"
    return pygame.font.SysFont(fam, 18), pygame.font.SysFont(fam, 24, bold=True)


def end_text(screen, fonts, env):
    msg = "사망..." if env.health <= 0 else "목표 일수 달성!"
    screen.blit(fonts[1].render(msg, True, (255, 255, 0)), (270, 180))


def record(args):
    """창 없이 한 판을 실행해 GIF로 저장"""
    from PIL import Image
    pygame.init()
    screen = pygame.Surface((W, H))
    fonts = make_fonts()
    label = make_label(args)
    env = MinecraftSurvivalEnv(max_days=args.max_days)
    policy = make_policy(args, env)
    s, _ = env.reset(seed=args.seed)

    frames, durations = [], []

    def grab(duration):
        frames.append(Image.frombytes("RGB", (W, H), pygame.image.tostring(screen, "RGB")))
        durations.append(duration)

    last_action, reward, total, done = None, 0.0, 0.0, False
    draw(screen, fonts, env, last_action, reward, total, label)
    grab(args.delay * 2)
    while not done:
        last_action = policy(s)
        s, reward, term, trunc, _ = env.step(last_action)
        total += reward
        done = term or trunc
        draw(screen, fonts, env, last_action, reward, total, label)
        if done:
            end_text(screen, fonts, env)
        grab(args.delay * (4 if done else 1))

    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
    frames[0].save(args.record, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"저장 완료: {args.record}  ({label}, {env.day}일 생존, {len(frames)} 프레임)")
    pygame.quit()


def main(args):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Minecraft Survival RL")
    fonts = make_fonts()
    label = make_label(args)
    clock = pygame.time.Clock()
    if args.save_frames:
        os.makedirs("frames", exist_ok=True)

    env = MinecraftSurvivalEnv(max_days=args.max_days)
    policy = make_policy(args, env)
    s, _ = env.reset(seed=args.seed)
    last_action, reward, total, done, paused, frame = None, 0.0, 0.0, False, False, 0
    delay = args.delay
    last_step = pygame.time.get_ticks()

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_UP:
                    delay = max(50, delay - 100)
                elif e.key == pygame.K_DOWN:
                    delay += 100
                elif e.key == pygame.K_r:
                    args.seed += 1
                    s, _ = env.reset(seed=args.seed)
                    last_action, reward, total, done = None, 0.0, 0.0, False

        now = pygame.time.get_ticks()
        if not paused and not done and now - last_step >= delay:
            last_action = policy(s)
            s, reward, term, trunc, _ = env.step(last_action)
            total += reward
            done = term or trunc
            last_step = now

        draw(screen, fonts, env, last_action, reward, total, label)
        if done:
            end_text(screen, fonts, env)
            screen.blit(fonts[0].render("R: 다시 시작", True, (255, 255, 0)), (285, 215))
        pygame.display.flip()
        if args.save_frames:
            pygame.image.save(screen, f"frames/{frame:05d}.png")
            frame += 1
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=["q", "heuristic", "random"], default="heuristic")
    p.add_argument("--tag", default="base")
    p.add_argument("--algo", default="qlearning")
    p.add_argument("--ckpt", default="final", help="final 또는 학습 진행률(예: 0, 10, 30, 60, 100)")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max_days", type=int, default=30)
    p.add_argument("--delay", type=int, default=600, help="한 스텝당 시간(ms)")
    p.add_argument("--record", default=None, help="지정하면 창 없이 한 판을 GIF로 저장 (예: results/base/videos/a.gif)")
    p.add_argument("--save_frames", action="store_true")
    a = p.parse_args()
    record(a) if a.record else main(a)
