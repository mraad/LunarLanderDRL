import gym
import cv2
from gym import wrappers

if __name__ == '__main__':
    frames = []
    env = gym.make("LunarLander-v2")
    # env = wrappers.Monitor(env, "./gym-results", force=True)
    state = env.reset()
    done = False
    while not done:
        # env.render()
        frames.append(env.render(mode='rgb_array'))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
    env.close()

    size = (96, 96)

    out = cv2.VideoWriter('LunarLander.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for frame in frames:
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(rgb_img)
    out.release()