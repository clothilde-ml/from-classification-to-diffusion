import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager 

SAVE_DIR = "/content/drive/MyDrive/model_checkpoints"

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay  = decay
        self.shadow = {k: v.clone().float() 
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply_to(self, model):
        """Charge les poids EMA dans le modèle pour la génération."""
        model.load_state_dict(
            {k: v.to(next(model.parameters()).device) 
             for k, v in self.shadow.items()}
        )

class Scheduler:
    """
    Schedule linéaire (Ho et al., 2020) OU cosinus (Nichol & Dhariwal, 2021).
    Paramètre schedule : "cosine" (recommandé) ou "linear".
    """
    def __init__(self, T: int = 1000, schedule: str = "cosine",
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 device=None):
        self.T      = T
        self.device = device or torch.device("cpu")
        if schedule == "cosine":
            betas_, alphas_, alpha_bars_ = self._cosine(T, s=8e-3)
        else:
            betas_, alphas_, alpha_bars_ = self._linear(T, beta_start, beta_end)

        self.betas      = betas_.to(self.device)
        self.alphas     = alphas_.to(self.device)
        self.alpha_bars = alpha_bars_.to(self.device)
        self.sqrt_alpha_bars           = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    @staticmethod
    def _cosine(T: int, s: float):
        steps = torch.arange(T + 1, dtype=torch.float64)
        f     = torch.cos(((steps / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        ab    = f / f[0]
        betas = torch.clamp(1.0 - ab[1:] / ab[:-1], 0.0, 0.999)
        return betas.float(), (1.0 - betas).float(), ab[1:].float()

    @staticmethod
    def _linear(T: int, beta_start: float, beta_end: float):
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        return betas, alphas, torch.cumprod(alphas, dim=0)

    def _coefs(self, t: torch.Tensor):
        i = t[0]
        return (self.betas[i].view(1,1,1,1),
                self.alphas[i].view(1,1,1,1),
                self.alpha_bars[i].view(1,1,1,1))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab   = self.sqrt_alpha_bars[t].view(-1,1,1,1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bars[t].view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def mu_cfg(self, model, xt, t, y, null_token:int, w=3.0):
        null_y = torch.full_like(y, null_token)
        eps_c, eps_u = model(
            torch.cat([xt, xt]),
            torch.cat([t,  t]),
            torch.cat([y, null_y]),
        ).chunk(2)
        eps = (1 + w) * eps_c - w * eps_u
        b, a, ab = self._coefs(t)
        mu = (1.0 / torch.sqrt(a)) * (xt - (b / torch.sqrt(1.0 - ab)) * eps)
        return mu, b

    def p_step(self, mu_fn, *args):
        mu, beta_t = mu_fn(*args)
        t = args[2]
        if t[0].item() > 0:
            return mu + torch.sqrt(beta_t) * torch.randn_like(mu)
        return mu
    


def make_norm(num_ch):
    for g in [32,16,8,4,2,1]:
        if num_ch % g == 0:
            return nn.GroupNorm(g, num_ch)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.arange(half) * -(math.log(10000) / (half - 1)))
        self.register_buffer('freqs', freqs)
    def forward(self, t):
        emb = t[:,None].float() * self.freqs[None,:]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class TimeEmbedding(nn.Module):
    def __init__(self, sin_dim, out_dim):
        super().__init__()
        self.sin_emb = SinusoidalEmbedding(sin_dim)
        self.mlp = nn.Sequential(
            nn.Linear(sin_dim, out_dim * 4), nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim),
        )
    def forward(self, t):
        return self.mlp(self.sin_emb(t))

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = make_norm(out_ch)
        self.norm2 = make_norm(out_ch)
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, emb):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = F.silu(h + self.time_proj(emb).view(emb.size(0), -1, 1, 1))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = make_norm(channels)
    def forward(self, x):
        B, C, H, W = x.shape
        h, _ = self.attn(*[x.flatten(2).permute(0,2,1)]*3)
        return self.norm(x + h.permute(0,2,1).view(B,C,H,W))

class ConditionalMiniUNet(nn.Module):
    """
    U-Net conditionnel à n_levels niveaux d'encodeur/décodeur.
    Self-attention optionnelle au bottleneck (attn=True).

    Canaux à chaque niveau : base_ch, 2*base_ch, 4*base_ch, ...
    L'interpolation dans le décodeur corrige les décalages de taille
    pour les résolutions impaires (ex: MNIST 28x28).

    Résolutions recommandées selon n_levels :
      n_levels=2 -> MNIST  (28x28, bottleneck 7x7)
      n_levels=3 -> CIFAR  (32x32, bottleneck 4x4  |  64x64, bottleneck 8x8)
    """
    def __init__(self, in_channels: int = 1, time_emb_dim: int = 128,
                 num_classes: int = 10, base_ch: int = 64, n_levels: int = 2,
                 attn: bool = False, attn_heads: int = 4):
        super().__init__()
        C             = base_ch
        self.n_levels = n_levels

        self.time_emb  = TimeEmbedding(sin_dim=time_emb_dim, out_dim=time_emb_dim)
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)  # +1 pour NULL_TOKEN

        # Canaux à chaque niveau : C, 2C, 4C, ... (plafonné à 8C)
        ch = [min(C * (2 ** i), C * 8) for i in range(n_levels + 1)]

        # ----- Encodeurs ---------------------------------------------
        self.encoders = nn.ModuleList()
        self.downs    = nn.ModuleList()
        in_ch = in_channels
        for i in range(n_levels):
            # 2 ResBlock par niveau du UNet 
            #  -> nette amélioration des performances mais training plus long
            self.encoders.append(nn.ModuleList([
                ResBlock(in_ch,  ch[i], time_emb_dim),
                ResBlock(ch[i],  ch[i], time_emb_dim),
            ]))
            self.downs.append(nn.Conv2d(ch[i], ch[i], 4, stride=2, padding=1))
            in_ch = ch[i]

        # ----- Bottleneck avec attention optionnelle -----------------
        self.bot1 = ResBlock(ch[n_levels - 1], ch[n_levels], time_emb_dim)
        self.attn = SelfAttention2d(ch[n_levels], num_heads=attn_heads) if attn else nn.Identity()
        self.bot2 = ResBlock(ch[n_levels], ch[n_levels], time_emb_dim)

        # ----- Décodeurs (entrée = up + skip -> canaux doublés) ------
        self.ups      = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(n_levels)):
            self.ups.append(nn.ConvTranspose2d(ch[i + 1], ch[i + 1], 4, stride=2, padding=1))
            self.decoders.append(nn.ModuleList([
                ResBlock(ch[i + 1] + ch[i], ch[i], time_emb_dim),  # prend le skip
                ResBlock(ch[i], ch[i], time_emb_dim),
            ]))

        self.final = nn.Conv2d(ch[0], in_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.time_emb(t)
        if y is not None:
            emb = emb + self.class_emb(y)

        # ----- Encodeur + stockage des skip connections --------------
        skips, h = [], x
        for enc_blocks, down in zip(self.encoders, self.downs):
            for enc in enc_blocks:
                h = enc(h, emb)
            skips.append(h)
            h = down(h)

        # ----- Bottleneck -------------------------------------------
        h = self.bot2(self.attn(self.bot1(h, emb)), emb)

        # ----- Décodeur + interpolate sur résolutions impaires ------
        for up, dec_blocks, skip in zip(self.ups, self.decoders, reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode='nearest')
            h = torch.cat([h, skip], dim=1)
            for dec in dec_blocks:
                h = dec(h, emb)

        return self.final(h)
    
@contextmanager
def eval_mode(model):
    model.eval()    
    try:
        yield      
    finally:
        model.train() 
    
def diffusion_loss_cond(model, x0, y, scheduler, null_token, p_uncond=0.2):
    t     = torch.randint(0, scheduler.T, (x0.size(0),), device=x0.device)
    noise = torch.randn_like(x0)
    xt    = scheduler.q_sample(x0, t, noise)
    keep  = torch.rand(x0.size(0), device=x0.device) > p_uncond
    y_in  = torch.where(keep, y, torch.full_like(y, null_token))
    return F.mse_loss(model(xt, t, y_in), noise)

@torch.no_grad()
def sample_cond(model: nn.Module, scheduler: Scheduler, label: int,
                n_samples: int = 16, w: float = 3.0,
                in_ch: int = 1, img_size: int = 28) -> torch.Tensor:
    """Génère n_samples images du label demandé avec CFG."""
    dev = next(model.parameters()).device
    xt  = torch.randn(n_samples, in_ch, img_size, img_size, device=dev)
    y   = torch.full((n_samples,), label, device=dev, dtype=torch.long)
    with eval_mode(model):
        for t_step in reversed(range(scheduler.T)):
            t  = torch.full((n_samples,), t_step, device=dev, dtype=torch.long)
            xt = scheduler.p_step(scheduler.mu_cfg, model, xt, t, y, w)
    return xt


def train_model(model, loader, loss_fn, epochs=20, lr=2e-4, clip_grad=True,
                checkpoint_every=5, checkpoint_name="checkpoint",
                retrain=False, ema=None, **loss_kwargs):
    """
    Boucle d'entraînement avec support EMA optionnel.
    Passe `ema=ema_instance` pour activer la mise à jour automatique.
    """
    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr)
    # Ajout d'un cosine annealing pour contrer la loss stagnante
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )
    loss_history = []
    start_epoch  = 0

    ckpt_path = os.path.join(SAVE_DIR, f"{checkpoint_name}_ckpt.pt")
    if os.path.exists(ckpt_path) and not retrain :
        print(f"  v Checkpoint trouvé, reprise depuis : {ckpt_path}")
        ckpt   = torch.load(ckpt_path, map_location=next(model.parameters()).device)
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        target.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch  = ckpt["epoch"] + 1
        loss_history = ckpt["loss_history"]
        print(f"  -> Reprise à l'epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        for x0, y in loader:
            dev    = next(model.parameters()).device
            x0, y  = x0.to(dev), y.to(dev)
            loss   = loss_fn(model, x0=x0, y=y, **loss_kwargs)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            epoch_loss += loss.item()

        scheduler_lr.step()
        avg = epoch_loss / len(loader)
        loss_history.append(avg)
        print(f"  Epoch {epoch+1:2d}/{epochs}  Loss : {avg:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            target = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_data = {
                "epoch":           epoch,
                "model_state":     target.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss_history":    loss_history,
            }
            torch.save(ckpt_data, ckpt_path)
            print(f"  v Checkpoint sauvegardé - epoch {epoch+1}")

    return loss_history

def save_model(model, name, rewards_history=None):
    path = os.path.join(SAVE_DIR, f"{name}.pt")
    state = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    torch.save({"model_state_dict": state, "rewards_history": rewards_history}, path)
    print(f"Modèle '{name}' sauvegardé : {path}")

def load_model(base_model, name, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path       = os.path.join(SAVE_DIR, f"{name}.pt")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Récupère le modèle de base non compilé
    target = base_model._orig_mod if hasattr(base_model, "_orig_mod") else base_model
    model  = copy.deepcopy(target)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Modèle '{name}' rechargé depuis : {path}")
    return model, checkpoint.get("rewards_history", None)