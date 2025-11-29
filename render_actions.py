import argparse
import pickle
import imageio
import gymnasium as gym
import time

def render_episode_from_pickle(pickle_path, env_name, gif_path, episode=None, delay=0.0, fps=None, frame_stride=1):
    # Load actions data
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Pickle file must contain a non-empty list of episode action entries.")

    # Select episode entry
    entry = None
    if episode is not None:
        for e in data:
            if isinstance(e, dict) and e.get("episode") == episode:
                entry = e
                break
        if entry is None:
            raise ValueError(f"Episode {episode} not found in pickle.")
    else:
        # Default to last entry
        entry = data[-1]

    actions = entry.get("actions")
    seed = entry.get("seed", None)
    print("Acquired actions for rendering")

    if actions is None:
        raise ValueError("Selected entry does not contain 'actions'.")

    # Create env in rgb_array mode for GIF
    env = gym.make(env_name, render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()
    print("Environment created")

    duration = (1.0 / fps) if (fps and fps > 0) else (delay if delay and delay > 0 else 0.02)
    writer = imageio.get_writer(gif_path, mode="I", duration=duration)
    try:
        from tqdm import tqdm
        frame_idx = 0
        for a in tqdm(actions, desc="Rendering frames", unit="step"):
            _, _, terminated, truncated, _ = env.step(int(a))
            frame = env.render()
            if frame_idx % frame_stride == 0:
                writer.append_data(frame)
            frame_idx += 1
            if terminated or truncated:
                break
    finally:
        writer.close()
        env.close()

if __name__ == "__main__":
    pickleLocation = "./runs_data/20/actions.pkl"
    envName = "LunarLander-v3"
    gifLocToSave = "./runs_data/20/sampleRun.gif"
    episode = 471
    render_episode_from_pickle(pickleLocation, envName, gifLocToSave, episode, frame_stride=2)