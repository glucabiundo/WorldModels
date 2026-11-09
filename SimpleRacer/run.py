import random, math, time, os
import numpy as np
import torch
from matplotlib import rcParams
from utils import models, world
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Parameters
SEED = 5
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEV}")

FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / "data" / "Examples.npz"
RESULT_PATH = FILE_PATH / "results"
RESULT_PATH.mkdir(exist_ok=True)

# Colors and styles
PAPER, INK, MUTED = "#ffffff", "#000000", "#71602CE4"
DARKRED, OCRA, CYAN = "#B00020", "#D9A066", "#00B0F0"

rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 12,
    "font.family": "serif", "axes.titlesize": 13, "axes.titleweight": "bold",
})

def train_vae(vae, data, plot = False):
    opt = torch.optim.Adam(vae.parameters(), lr=2e-3)
    frames = data["frames"]
    flat = torch.tensor(frames.reshape(-1, 24, 24, 3))
    env = world.RaceWorld(SEED)
    track_t = torch.tensor(env.track).to(DEV)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(flat),
                                     batch_size=256, shuffle=True)
    t0 = time.time()
    for epoch in range(30):
        tot = 0.0
        for (x,) in loader:
            x= x.to(DEV)
            mu, logvar = vae.encode(x)
            z = mu + torch.randn_like(mu)*torch.exp(0.5*logvar)
            z = z + 0.10 * torch.randn_like(z) # extra noise
            xh = vae.decode(z)
            w = 1.0 + 30.0 * (x - track_t).abs().mean(-1, keepdim=True)  # weight what moves
            recon = (w * F.binary_cross_entropy(xh,x, reduction="none")).sum() / len(x)
            kl = -0.5*torch.mean(torch.sum(1+logvar-mu**2-logvar.exp(), dim=1))
            loss = recon + kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(x)
        if (epoch+1) % 3 == 0:
            print(f"epoch {epoch+1:2d}  loss {tot/len(flat):8.2f}")
            print(f"V trained in {time.time()-t0:.0f}s   "
                  f"({sum(p.numel() for p in vae.parameters()):,} parameters)")
        
    if plot == True:
        vae.eval()
        with torch.no_grad():
            test = flat[:20000:2611][:8].to(DEV)
            recon, _, _ = vae(test)

            fig, axes = plt.subplots(2, 8, figsize=(12, 3.4))
            for i in range(8):
                axes[0, i].imshow(test[i].cpu()); axes[0, i].axis("off")
                axes[1, i].imshow(recon[i].cpu().clamp(0, 1)); axes[1, i].axis("off")
            axes[0, 0].set_title("original", loc="left", color=OCRA)
            axes[1, 0].set_title("V's reconstruction (from 16 numbers)", loc="left", color=CYAN)
            fig.suptitle("V — 1,728 pixels → 16 numbers → 1,728 pixels", y=1.02)
            plt.tight_layout(); plt.savefig(RESULT_PATH / "plot_recon.png", dpi=150, bbox_inches="tight"); plt.show()

            # latent walk between two frames of the SAME episode, half a lap apart
            e_walk = 2
            z1, _ = vae.encode(torch.tensor(frames[e_walk, 0:1]).to(DEV))
            z2, _ = vae.encode(torch.tensor(frames[e_walk, 26:27]).to(DEV))
            steps = torch.linspace(0, 1, 9, device=DEV).view(-1, 1)
            walk = vae.decode(z1 + steps * (z2 - z1))

            fig, axes = plt.subplots(1, 9, figsize=(12, 1.8))
            for i in range(9):
                axes[i].imshow(walk[i].cpu().clamp(0, 1)); axes[i].axis("off")
            axes[0].set_title("code A", loc="left", color=DARKRED)
            axes[-1].set_title("code B", loc="right", color=CYAN)
            fig.suptitle("The smooth walk — the car glides around the track between two codes", y=1.15)
            plt.tight_layout(); plt.savefig(RESULT_PATH / "plot_walk.png", dpi=150, bbox_inches="tight"); plt.show()



def main():
    vae = models.VAE().to(DEV)
    data = np.load(DATA_PATH)

    train_vae(vae,data, plot = True)


if __name__ == "__main__":
    main()