import numpy as np
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import os

class RaceWorld:
    """
    A class representing the racer world
    """

    SIZE = 24 # pixels grid edge length
    ACTIONS = 3
    R, HALF_W, SPEED = 8, 2.2, 1.5

    # RGB colors for different elements in the world
    GRASS = np.array([0.16, 0.34, 0.16], np.float32)
    ROAD = np.array([0.44, 0.44, 0.46], np.float32)
    KERB = np.array([0.75, 0.72, 0.62], np.float32)
    CAR = np.array([0.92, 0.20, 0.15], np.float32)
 
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        c = (self.SIZE - 1) /2 
        yy, xx = np.meshgrid(np.arange(self.SIZE), np.arange(self.SIZE))
        rho = np.sqrt((xx - c)**2 + (yy - c)**2)
        self.track = np.tile(self.GRASS, (self.SIZE, self.SIZE, 1)).astype(np.float32)
        self.track[np.abs(rho - self.R) <= self.HALF_W] = self.ROAD
        self.track[np.abs(np.abs(rho - self.R) - self.HALF_W) <= 0.45] = self.KERB
        self._c = c
        self.reset()

    def reset(self):
        self.theta = self.rng.uniform(0, 2 * np.pi) 
        self.d = self.rng.uniform(-1, 1)
        self.phi = self.rng.uniform(-0.2, 0.2)
        self.direction = int(self.rng.choice([-1, 1]))
        self.t = 0
        return self.observe()

    @property
    def state(self):
        return np.array([self.theta, self.d, self.phi, self.direction], np.float32)

    def observe(self):
        """
        Returns the current observation of the world
        """
        obs = self.track.copy()
        r = self.R + self.d
        x = self._c + r * np.cos(self.theta)
        y = self._c + r * np.sin(self.theta)
        xi, yi = int(round(x)), int(round(y))
        for dx in (0,1):
            for dy in (0,1):
                if 0 <= xi + dx -1 < self.SIZE and 0 <= yi + dy -1 < self.SIZE:
                    obs[xi + dx, yi + dy] = self.CAR

        return obs                   

    def step(self, action):
        """
        Updates the world state based on the action taken
        """
        self.phi = float(np.clip(self.phi + 0.16 * (action - 1), -0.7, 0.7))
        self.d = float(self.d + 1.1 * np.sin(self.phi))
        if abs(self.d) > self.HALF_W:                    # scraping the kerb
            self.d = float(np.clip(self.d, -self.HALF_W, self.HALF_W))
            self.phi *= 0.4
        self.theta = (self.theta + self.direction * self.SPEED / (self.R + self.d)) % (2 * np.pi)
        reward = 1.0 - abs(self.d) / self.HALF_W
        self.t += 1
        return self.observe(), float(reward), self.t >= 150


def generate_dataset(seed, episodes, steps):
    """
    Generates a dataset of observations, actions, and rewards
    """
    world = RaceWorld(seed)
    rng = np.random.default_rng(seed)
    observations = np.zeros((episodes, steps, world.SIZE, world.SIZE, 3), np.float32)
    actions = np.zeros((episodes, steps), np.int16)
    reward = np.zeros((episodes, steps), np.float32)
    for e in range(episodes):
        obs = world.reset()
        for t in range(steps):
            a = int(rng.integers(0, world.ACTIONS))
            observations[e,t], actions[e,t] = obs, a
            obs, r, _ = world.step(a)
            reward[e, t] = r
    
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" 
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / "Examples.npz" 
    np.savez_compressed(file_path, frames = observations, actions = actions, rewards = reward)
    size = os.path.getsize(file_path)/ (1024**2) # MB
    print (f"Dataset created: {episodes} episodes x {steps} time steps")
    print (f"FILESIZE: {size:.2f} MB")
    
    return file_path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 5)
    parser.add_argument("--episodes", type = int, default = 150)
    parser.add_argument("--steps", type=int, default=150)
    args= parser.parse_args()

    file = generate_dataset(**vars(args))
    print(f"Dataset currently saved in: {file}")
    
    data = np.load(file)
    frames = data["frames"]
    fig, axes = plt.subplots(1, 8, figsize=(12, 1.9))
    for i in range(8):
        axes[i].imshow(frames[i, i * 9]); axes[i].axis("off")
        fig.suptitle("Eight raw observations from the dataset", y=1.12)

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()