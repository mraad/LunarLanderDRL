from typing import Optional

import click
import cv2
import gymnasium as gym
import numpy as np
import torch

from rl.model import MLP
from rl.utils import env_shape


def write_frame(writer: Optional[cv2.VideoWriter],
                mp4_path: str,
                fps: int,
                frame: np.ndarray
                ) -> cv2.VideoWriter:
    """Append one RGB frame, centre cropped to a square, opening the writer on first use."""
    height, width = frame.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    frame = frame[top:top + side, left:left + side, ::-1]  # OpenCV wants BGR
    if writer is None:
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (side, side))
    writer.write(np.ascontiguousarray(frame))
    return writer


@click.command()
@click.option('--env_name', type=str, default="LunarLander-v3", show_default=True, help='Environment name.')
@click.option('--pth_path', type=str, default="models/lunarlander-v3.pth", show_default=True, help='Path of pytorch weight file.')
@click.option('--mp4_path', type=str, default="LunarLander.mp4", show_default=True, help='Path of MP4 file.')
@click.option('--save/--no-save', default=True, show_default=True, help="Save mp4 at the end of the run.")
@click.option('--num-sims', type=int, default=5, show_default=True, help="Number of simulations.")
@click.option('--fps', type=int, default=50, show_default=True, help="Frame rate of the MP4.")
@click.option('--seed', type=int, default=None, help="Seed for the environment, random when unset.")
def main(env_name: str,
         pth_path: str,
         mp4_path: str,
         save: bool,
         num_sims: int,
         fps: int,
         seed: Optional[int]
         ) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = None
    with gym.make(env_name, render_mode="rgb_array" if save else None) as env:
        model = MLP(*env_shape(env)).to(device)
        model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
        model.eval()

        for sim in range(num_sims):
            state, _ = env.reset(seed=seed if sim == 0 else None)
            done = False
            total_reward = 0.0
            while not done:
                if save:
                    writer = write_frame(writer, mp4_path, fps, np.asarray(env.render()))
                with torch.no_grad():
                    pt_state = torch.as_tensor(state, dtype=torch.float32, device=device)
                    action = int(model(pt_state).argmax().item())
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += float(reward)
            click.echo(f"simulation {sim}: reward {total_reward:.1f}")

    if writer is not None:
        writer.release()
        click.echo(f"wrote {mp4_path}")


if __name__ == '__main__':
    main()
