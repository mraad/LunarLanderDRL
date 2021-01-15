import click
import gym
import torch
import cv2
import numpy as np
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

from rl.model import MLP


@click.command()
@click.option('--env_name', type=str, default="LunarLander-v2", show_default=True, help='Environment name.')
@click.option('--pth_path', type=str, default="2021_01_13_06_17.pth", help='Path of pytorch weight file.')
@click.option('--mp4_path', type=str, default="lunarlander.mp4", help='Path of MP4 file.')
def main(env_name: str,
         pth_path: str,
         mp4_path: str
         ) -> None:
    frames = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with gym.make(env_name) as env:
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        model = MLP(n_states, n_actions)
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.eval()

        for i in range(5):
            state = env.reset()
            done = False
            while not done:
                frame = env.render(mode="rgb_array")
                frames.append(frame)
                state = torch.from_numpy(state).to(device)
                action = model(state).argmax().item()
                # action = env.action_space.sample()
                state, reward, done, _ = env.step(action)
    w, h, c = frames[0].shape
    s = (h - w) // 2
    out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 48, (w, w))
    for f in frames:
        out.write(f[:, s:h - s + 1, :])
    out.release()


if __name__ == '__main__':
    main()