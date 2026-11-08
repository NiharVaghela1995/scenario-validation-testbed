"""CPU-feasible methods from later "eras", on the same synthetic / esmini test bed.

Sequence autoencoder (1-D convolutional) on multichannel windows; reconstruction error as a detector;
    latent embeddings clustered with DBSCAN / HDBSCAN.
Prediction error as a detector: a kinematic (constant relative speed) gap predictor and a learned linear
    (ridge) predictor trained on calm data - the simplest form of "surprise". NOT a world model.
Scene graph per timestep + a temporal-logic query with quantitative robustness (STL-style, numpy).
A small denoising diffusion model (DDPM, MLP denoiser) that generates (ego speed, gap) pairs.

Everything is CPU, seeded, and small. Results are "synthetic test bed, illustrative".
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

import detectors as D

torch.set_num_threads(2)
WIN, STEP = 40, 5           # 4 s windows, 0.5 s stride at 10 Hz


# ------------------------------------------------------------------ shared: channels and windows
def channels(ser):
    """[1/TTC, THW/4 (clipped, NaN -> 1), ego deceleration/9 (>=0)] per timestep."""
    from series import detector_signal
    inv = detector_signal(ser)
    thw = np.clip(ser.thw.to_numpy(float), 0, 4) / 4
    thw = np.where(np.isnan(thw), 1.0, thw)
    v = ser.v_ego.to_numpy(float)
    dec = np.clip(-np.gradient(v, 0.1), 0, 9) / 9
    dec = np.convolve(dec, np.ones(5) / 5, mode="same")        # 0.5 s smoothing of a noisy derivative
    return np.stack([inv / 5.0, thw, dec]).astype(np.float32)  # inv scaled by its cap


def windows(ser, win=WIN, step=STEP):
    X = channels(ser)
    starts = np.arange(0, X.shape[1] - win + 1, step)
    return np.stack([X[:, s:s + win] for s in starts]), starts


# ------------------------------------------------------------------ autoencoder
class ConvAE(nn.Module):
    def __init__(self, c=3, latent=4, win=WIN):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(c, 16, 5, stride=2, padding=2), nn.ReLU(),
                                 nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(16 * (win // 4), latent))
        self.dec = nn.Sequential(nn.Linear(latent, 16 * (win // 4)), nn.ReLU(),
                                 nn.Unflatten(1, (16, win // 4)),
                                 nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1), nn.ReLU(),
                                 nn.ConvTranspose1d(16, c, 4, stride=2, padding=1))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_ae(Xtr, Xva, lr=1e-3, epochs=60, batch=128, seed=0, latent=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = ConvAE(latent=latent)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    tr, va = torch.tensor(Xtr), torch.tensor(Xva)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(tr), generator=g)
        for i in range(0, len(tr), batch):
            xb = tr[perm[i:i + batch]]
            loss = ((m(xb)[0] - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            hist.append((float(((m(tr)[0] - tr) ** 2).mean()), float(((m(va)[0] - va) ** 2).mean())))
    return m, hist


def ae_scores(model, ser):
    """Per-timestep score = max reconstruction MSE of the windows covering the timestep; latent per window."""
    W, starts = windows(ser)
    with torch.no_grad():
        rec, z = model(torch.tensor(W))
    err = ((rec - torch.tensor(W)) ** 2).mean(dim=(1, 2)).numpy()
    score = np.zeros(len(ser))
    for e, s in zip(err, starts):
        score[s:s + WIN] = np.maximum(score[s:s + WIN], e)
    return score, err, z.numpy(), starts


def score_to_intervals(score, t, thr):
    return D.runs_to_intervals(score > thr, t, min_dur=0.3, merge_gap=1.0)


# ------------------------------------------------------------------ prediction error
H = 10   # 1 s horizon


def kinematic_pred_error(ser):
    """|gap(t+1s) - [gap(t) - (v_ego - v_lead)(t) * 1 s]|, same lead only; 0 where undefined."""
    g = ser.gap.to_numpy(float)
    dv = (ser.v_ego - ser.v_lead).to_numpy(float)
    lead = ser.lead_id.to_numpy(float)
    e = np.zeros(len(ser))
    pred = g[:-H] - dv[:-H] * H * 0.1
    same = (lead[:-H] == lead[H:]) & np.isfinite(g[:-H]) & np.isfinite(g[H:])
    e[H:] = np.where(same, np.abs(g[H:] - pred), 0.0)            # aligned at the time the outcome is seen
    return e


def ar_features(ser):
    """Features at t for predicting gap(t+1s): gap, closing speed, its 0.5 s change, ego speed."""
    g = ser.gap.to_numpy(float)
    dv = (ser.v_ego - ser.v_lead).to_numpy(float)
    v = ser.v_ego.to_numpy(float)
    ddv = np.r_[np.zeros(5), dv[5:] - dv[:-5]]
    return np.column_stack([g, dv, ddv, v])


def fit_ridge(series_list, alpha=1.0):
    from sklearn.linear_model import Ridge
    Xs, ys = [], []
    for ser in series_list:
        F = ar_features(ser)
        lead = ser.lead_id.to_numpy(float)
        y = ser.gap.to_numpy(float)
        ok = np.isfinite(F[:-H]).all(1) & np.isfinite(y[H:]) & (lead[:-H] == lead[H:])
        Xs.append(F[:-H][ok])
        ys.append(y[H:][ok] - F[:-H][ok, 0])                     # predict the change of gap
    return Ridge(alpha=alpha).fit(np.vstack(Xs), np.concatenate(ys))


def ridge_pred_error(model, ser):
    F = ar_features(ser)
    lead = ser.lead_id.to_numpy(float)
    y = ser.gap.to_numpy(float)
    e = np.zeros(len(ser))
    ok = np.isfinite(F[:-H]).all(1) & np.isfinite(y[H:]) & (lead[:-H] == lead[H:])
    pred = np.full(len(ser) - H, np.nan)
    pred[ok] = F[:-H][ok, 0] + model.predict(F[:-H][ok])
    e[H:] = np.where(ok, np.abs(y[H:] - pred), 0.0)
    return e


# ------------------------------------------------------------------ scene graph + temporal-logic query
def scene_graph(frames, t, ego_id=0):
    """Nodes = vehicles at time t; edges ego->other with relation (lead / same_lane_behind / adjacent / other)."""
    f = frames[frames.t.round(2) == round(t, 2)]
    ego = f[f.vid == ego_id].iloc[0]
    nodes = [dict(id=int(r.vid), lane=int(r.lane), s=float(r.s_front), v=float(r.v)) for r in f.itertuples()]
    edges = []
    ahead = f[(f.vid != ego_id) & (f.lane == ego.lane) & (f.s_rear > ego.s_front)]
    lead = int(ahead.loc[ahead.s_rear.idxmin(), "vid"]) if len(ahead) else None
    for r in f[f.vid != ego_id].itertuples():
        rel = "lead" if r.vid == lead else ("same_lane" if r.lane == ego.lane else
                                           ("adjacent" if abs(r.lane - ego.lane) == 1 else "other"))
        edges.append(dict(src=ego_id, dst=int(r.vid), rel=rel, ds=float(r.s_rear - ego.s_front),
                          dv=float(ego.v - r.v)))
    return nodes, edges


def eventually(rho, k):
    """F_[0,k] : max of rho over the next k samples (inclusive)."""
    out = np.full(len(rho), -np.inf)
    for i in range(len(rho)):
        out[i] = np.max(rho[i:i + k + 1])
    return out


def always(rho, k):
    out = np.full(len(rho), np.inf)
    for i in range(len(rho)):
        out[i] = np.min(rho[i:i + k + 1])
    return out


def query_tight_cut_in(ser, thw_max=0.6, within_s=3.0):
    """phi(t) = lead_switch(t) AND F_[0, within] (THW < thw_max). Robustness = min(switch, max(thw_max - THW))."""
    thw = ser.thw.to_numpy(float)
    rho_thw = np.where(np.isnan(thw), -1.0, thw_max - thw)
    sw = np.where(ser.lead_switch.to_numpy(bool), 1.0, -1.0)
    return np.minimum(sw, eventually(rho_thw, int(within_s * 10)))


def query_sustained_closing(ser, ttc_max=3.0, hold_s=1.0):
    """phi(t) = G_[0, hold] (TTC < ttc_max). Robustness in 1/s: min over the window of (1/TTC - 1/ttc_max)."""
    from series import detector_signal
    rho = detector_signal(ser) - 1.0 / ttc_max
    return always(rho, int(hold_s * 10))


def query_intervals(rho, t, extend_s):
    """Time points with rho > 0 -> intervals [t, t + extend] (the witness window of the query)."""
    idx = np.flatnonzero(rho > 0)
    iv = [[float(t[i]), float(min(t[i] + extend_s, t[-1]))] for i in idx]
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 2), round(b, 2)] for a, b in out]


# ------------------------------------------------------------------ small DDPM on parameter vectors
class Denoiser(nn.Module):
    def __init__(self, d=2, hidden=128, T=100):
        super().__init__()
        self.T = T
        self.net = nn.Sequential(nn.Linear(d + 16, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, d))

    def temb(self, t):
        f = torch.exp(-np.log(1000.0) * torch.arange(8) / 8)
        a = t[:, None].float() * f[None]
        return torch.cat([torch.sin(a), torch.cos(a)], 1)

    def forward(self, x, t):
        return self.net(torch.cat([x, self.temb(t)], 1))


SCHEDULE = "cosine"   # 5 Nov 2026: was linear 1e-4..0.02, which with T = 100 leaves abar_T = 0.36 (x_T not noise)
CLIP_FACTOR = 1.2     # x0_hat clipped to +-1.2 x max |x0| of the standardised training data (design choice)


def ddpm_schedule(T=100, kind=SCHEDULE, b0=1e-4, b1=0.02, s=0.008):
    """Noise schedule. kind="cosine": Nichol & Dhariwal 2021, abar(t) = f(t)/f(0), f(t) = cos^2(((t/T)+s)/(1+s) * pi/2),
    beta_t = 1 - abar_t/abar_{t-1} clipped at 0.999 (abar_T ~ 0). kind="linear": linspace(b0, b1, T) (Ho et al. 2020;
    their range is meant for T = 1000)."""
    if kind == "cosine":
        tt = torch.arange(T + 1, dtype=torch.float64) / T
        f = torch.cos((tt + s) / (1 + s) * np.pi / 2) ** 2
        ab = f / f[0]
        beta = torch.clamp(1 - ab[1:] / ab[:-1], max=0.999).float()
    elif kind == "linear":
        beta = torch.linspace(b0, b1, T)
    else:
        raise ValueError(kind)
    alpha = 1 - beta
    abar = torch.cumprod(alpha, 0)
    return beta, alpha, abar


def train_ddpm(X, steps=3000, lr=1e-3, batch=256, T=100, seed=0, log_every=100, schedule=SCHEDULE):
    """Simplified DDPM objective: E || eps - eps_theta(sqrt(abar) x0 + sqrt(1-abar) eps, t) ||^2."""
    torch.manual_seed(seed)
    beta, alpha, abar = ddpm_schedule(T, schedule)
    m = Denoiser(d=X.shape[1], T=T)
    m.schedule = schedule
    m.clip_x0 = CLIP_FACTOR * float(np.abs(X).max())    # sampling clips x0_hat to the (scaled) training range
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    hist = []
    run = 0.0
    for it in range(1, steps + 1):
        x0 = Xt[torch.randint(len(Xt), (batch,), generator=g)]
        t = torch.randint(T, (batch,), generator=g)
        eps = torch.randn(x0.shape, generator=g)
        ab = abar[t][:, None]
        loss = ((m(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += loss.item()
        if it % log_every == 0:
            hist.append((it, run / log_every))
            run = 0.0
    return m, hist


@torch.no_grad()
def sample_ddpm(m, n, d=2, seed=1, clip_x0="model"):
    """Ancestral DDPM sampling in the x0-form: x0_hat = (x_t - sqrt(1-abar_t) eps)/sqrt(abar_t), optionally clipped to
    +-clip_x0, then the posterior mean of q(x_{t-1} | x_t, x0_hat); variance beta_t. Without clipping this equals the
    eps-form of Ho et al. 2020. Clipping (as diffusers' clip_sample) keeps the first steps of the cosine schedule,
    where beta_T = 0.999 and 1/sqrt(alpha_T) ~ 32, from amplifying denoiser errors. clip_x0=None disables it."""
    g = torch.Generator().manual_seed(seed)
    beta, alpha, abar = ddpm_schedule(m.T, getattr(m, "schedule", SCHEDULE))
    c = getattr(m, "clip_x0", None) if clip_x0 == "model" else clip_x0
    x = torch.randn((n, d), generator=g)
    for t in reversed(range(m.T)):
        eps = m(x, torch.full((n,), t))
        ab, abp = abar[t], (abar[t - 1] if t > 0 else torch.tensor(1.0))
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        if c is not None:
            x0 = x0.clamp(-c, c)
        mean = beta[t] * abp.sqrt() / (1 - ab) * x0 + (1 - abp) * alpha[t].sqrt() / (1 - ab) * x
        x = mean + (beta[t].sqrt() * torch.randn((n, d), generator=g) if t > 0 else 0)
    return x.numpy()


def car_following_pairs(series_by_seg):
    """(ego speed, gap) at 1 Hz from 'ok'/'opening' steps."""
    P = []
    for s in series_by_seg.values():
        s = s[(s.status.isin(["ok", "opening"])) & s.gap.notna()]
        s = s[(s.t * 10).round() % 10 == 0]
        P += list(zip(s.v_ego, s.gap))
    return np.array(P, dtype=float)
