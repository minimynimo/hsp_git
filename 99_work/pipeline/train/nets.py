"""LSTM / TCN in all three strategies (plan 1.3, 3.4-3.7, 4, 5).

  regional   one model for the pool; best epoch by median VAL NSE over basins (early stopping)
  adapted    regional model (one per seed) fine-tuned per basin: lr x ADAPT.lr_factor, <= ADAPT.max_epochs,
             early stopping on that basin's VAL NSE, best epoch kept (epoch 0 = regional weights)
  individual one model per basin; early stopping on that basin's VAL NSE

Loss: NSE* (models.networks.NSELoss) everywhere. Inputs: Precip/Temp/PET z-scored per basin (TRAIN
stats); target: discharge z-scored per basin; statics z-scored across the pool. Every input window
lies within DATA_START.., so 1987 is warm-up for every model.

RUN e.g.
  python train/nets.py regional   --model LSTM --seed 42
  python train/nets.py regional   --model TCN  --seed 42 --num_filters 64 --seq_length 180
  python train/nets.py adapted    --model LSTM --seed 42
  python train/nets.py individual --model TCN  --seed 42 --worker 0 --n_workers 4   (then --finalize)
"""
import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import experiment as E  # noqa: E402
from hsp.io import basin_scalers, load_basin_pool, load_forcing, load_static_attributes, scenario_frame  # noqa: E402
from hsp.metrics import nse  # noqa: E402
from hsp import runio  # noqa: E402
from models.networks import EALSTM, LSTMIndiv, NSELoss, TCN  # noqa: E402

warnings.filterwarnings("ignore", message="RNN module weights are not part")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------------------------------
# data (GPU resident)
# ------------------------------------------------------------------------------------------
class Data:
    """Scaled forcing of `basins` for one scenario on a common date axis starting at DATA_START."""

    def __init__(self, basins, scenario="og", scalers=None, statics=None):
        self.basins = list(basins)
        frames, self.scalers = [], scalers or {}
        for b in self.basins:
            df = load_forcing(b)
            if b not in self.scalers:
                self.scalers[b] = basin_scalers(b, df)
            frames.append(scenario_frame(df, scenario))
        self.dates = frames[0].index
        assert self.dates[0] == E.DATA_START and all(f.index.equals(self.dates) for f in frames)
        dm = np.stack([self.scalers[b]["dyn_mean"] for b in self.basins])[:, None, :]
        ds = np.stack([self.scalers[b]["dyn_std"] for b in self.basins])[:, None, :]
        X = (np.stack([f[E.DYN_COLS].to_numpy(float) for f in frames]) - dm) / ds
        self.q_mean = np.array([self.scalers[b]["q_mean"] for b in self.basins])
        self.q_std = np.array([self.scalers[b]["q_std"] for b in self.basins])
        Q = np.stack([f[E.Q_COL].to_numpy(float) for f in frames])
        self.Q = Q                                                     # physical, for metrics
        self.X = torch.tensor(X, dtype=torch.float32, device=DEV)
        self.Y = torch.tensor((Q - self.q_mean[:, None]) / self.q_std[:, None], dtype=torch.float32, device=DEV)
        self.qstd_t = torch.tensor(self.q_std, dtype=torch.float32, device=DEV)
        self.S = None if statics is None else torch.tensor(statics.loc[self.basins].to_numpy(float),
                                                           dtype=torch.float32, device=DEV)

    def index(self, period, seq, require_obs=True, basin_idx=None):
        """(basin index, time index) of every target day in `period` with a full input window."""
        t = np.where((self.dates >= period[0]) & (self.dates <= period[1]))[0]
        t = t[t >= seq - 1]
        bs = range(len(self.basins)) if basin_idx is None else [basin_idx]
        bb, tt = [], []
        for b in bs:
            ok = t[np.isfinite(self.Q[b, t])] if require_obs else t
            bb.append(np.full(len(ok), b)); tt.append(ok)
        return (torch.tensor(np.concatenate(bb), device=DEV), torch.tensor(np.concatenate(tt), device=DEV))


def static_table(basins):
    a = load_static_attributes(basins)
    return (a - a.mean()) / a.std().replace(0, 1)


def batch(data, b, t, seq):
    offs = torch.arange(-seq + 1, 1, device=DEV)
    x = data.X[b[:, None], t[:, None] + offs]
    s = None if data.S is None else data.S[b]
    return x, s


@torch.no_grad()
def predict_z(model, data, b, t, seq, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(b), bs):
        x, s = batch(data, b[i:i + bs], t[i:i + bs], seq)
        out.append(model(x, s).squeeze(1))
    return torch.cat(out) if out else torch.empty(0, device=DEV)


def val_score(model, data, vb, vt, seq):
    """Median over basins of VAL NSE (physical units)."""
    pz = predict_z(model, data, vb, vt, seq).cpu().numpy()
    b, t = vb.cpu().numpy(), vt.cpu().numpy()
    q = pz * data.q_std[b] + data.q_mean[b]
    df = pd.DataFrame({"b": b, "o": data.Q[b, t], "s": np.maximum(q, 0)})
    return float(df.groupby("b").apply(lambda g: nse(g["o"], g["s"]), include_groups=False).median())


def train(model, data, tb, tt, vb, vt, seq, lr, batch_size, max_epochs, patience, lr_plateau=False,
          log=None, start_score=None, tag=""):
    """Generic loop: NSE* loss, Adam, grad-clip 1, best epoch by VAL score, early stopping.
    start_score = VAL score of the initial weights (adapted: epoch 0 = regional model)."""
    loss_fn = NSELoss(E.NSE_LOSS_EPS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3) if lr_plateau else None
    best_state = copy.deepcopy(model.state_dict()) if start_score is not None else None
    best, best_epoch, bad, hist = (start_score if start_score is not None else -np.inf), 0, 0, []
    n = len(tb)
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x, s = batch(data, tb[idx], tt[idx], seq)
            y = data.Y[tb[idx], tt[idx]].unsqueeze(1)
            pred = model(x, s)
            loss = loss_fn(pred, y, data.qstd_t[tb[idx]].unsqueeze(1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        score = val_score(model, data, vb, vt, seq)
        hist.append({"epoch": epoch, "train_loss": tot / n, "val_median_NSE": score,
                     "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        if log is not None:
            print(f"{tag} epoch {epoch}: loss {tot / n:.4f} val median NSE {score:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if sched is not None:
            sched.step(score)
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, best, hist


# ------------------------------------------------------------------------------------------
# model construction
# ------------------------------------------------------------------------------------------
def regional_cfg(model, args):
    if model == "LSTM":
        c = dict(E.LSTM_REGIONAL)
        if getattr(args, "hidden_size", None):
            c["hidden_size"] = args.hidden_size
        if getattr(args, "dropout", None):
            c["dropout"] = args.dropout
    else:
        c = dict(E.TCN_REGIONAL)
        if args.num_filters is not None:
            c["num_filters"] = args.num_filters
    if args.seq_length:
        c["seq_length"] = args.seq_length
    if args.max_epochs:
        c["max_epochs"] = args.max_epochs
    return c


def regional_tag(model, c, args):
    if model == "TCN":
        return f"f{c['num_filters']}_w{c['seq_length']}"
    # LSTM production calls pass neither override, so tag stays "" and the one production run
    # directory is untouched; a hyperparameter-search run (3.5 re-check) is tagged so it writes
    # to its own directory instead of colliding with the production run.
    if getattr(args, "hidden_size", None) or getattr(args, "dropout", None):
        return f"h{c['hidden_size']}_d{c['dropout']}_w{c['seq_length']}"
    return ""


def build(model, c, n_static):
    if model == "LSTM" and n_static:
        return EALSTM(len(E.DYN_COLS), n_static, c["hidden_size"], c["initial_forget_bias"], c["dropout"]).to(DEV)
    if model == "LSTM":
        return LSTMIndiv(len(E.DYN_COLS), c["hidden_size"], c["dropout"]).to(DEV)
    return TCN(len(E.DYN_COLS), c["num_filters"], c["num_levels"], c["kernel_size"], c["dropout"],
               n_static=n_static, static_embedding=c.get("static_embedding", 32)).to(DEV)


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simulate(model, basins, seq, scalers, statics, pairs=None):
    """{scenario: {basin: Series}} for og (TRAIN[0]..TEST[1]) and every synthetic scenario."""
    sims = {}
    for scen in runio.SIM_COLS:
        d = Data(basins, scen, scalers, statics)
        win = (E.TRAIN[0], E.TEST[1]) if scen == "og" else E.SCENARIO_WINDOW[scen]
        b, t = d.index(win, seq, require_obs=False)
        q = predict_z(model, d, b, t, seq).cpu().numpy()
        b, t = b.cpu().numpy(), t.cpu().numpy()
        q = np.maximum(q * d.q_std[b] + d.q_mean[b], 0)
        sims[scen] = {}
        for i, bas in enumerate(d.basins):
            m = b == i
            sims[scen][bas] = pd.Series(q[m], index=d.dates[t[m]])
    return sims


# ------------------------------------------------------------------------------------------
# strategies
# ------------------------------------------------------------------------------------------
def run_regional(args):
    c = regional_cfg(args.model, args)
    tag = regional_tag(args.model, c, args)
    rd = E.run_dir(args.model, "regional", args.seed, tag)
    if runio.is_done(rd) and not args.force:
        print(f"done already: {rd}")
        return
    seed_all(args.seed)
    basins = load_basin_pool()[: args.n_basins]
    statics = static_table(basins)
    data = Data(basins, "og", statics=statics)
    seq = c["seq_length"]
    tb, tt = data.index(E.TRAIN, seq)
    vb, vt = data.index(E.VAL, seq)
    model = build(args.model, c, statics.shape[1])
    runio.init_run(rd, {"model": args.model, "strategy": "regional", "seed": args.seed, "hyperparameters": c,
                        "loss": E.LOSS_NAME, "architecture": type(model).__name__, "statics": E.STATIC_ATTRS,
                        "n_basins": len(basins), "n_train_samples": len(tb),
                        "receptive_field": getattr(model, "receptive_field", None)})
    print(f"[{args.model} regional seed {args.seed} {tag}] loss: {E.LOSS_NAME} | {len(basins)} basins, "
          f"{len(tb)} train samples | {c}", flush=True)
    best_epoch, best, hist = train(model, data, tb, tt, vb, vt, seq, c["learning_rate"], c["batch_size"],
                                   c["max_epochs"], c["patience"], lr_plateau=(args.model == "TCN"),
                                   log=True, tag=f"[{args.model} regional s{args.seed}]")
    pd.DataFrame(hist).to_csv(rd / "train_log.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "cfg": c, "best_epoch": best_epoch, "val_median_NSE": best,
                "statics_mean_std": "pool z-score", "basins": basins}, rd / "best_model.pt")
    sims = simulate(model, basins, seq, data.scalers, statics)
    summary = runio.finalize(rd, sims)
    (rd / "run_summary.txt").write_text(
        f"{args.model} regional seed {args.seed} {tag}\nloss {E.LOSS_NAME}\nconfig {c}\n"
        f"best epoch {best_epoch} (val median NSE {best:.4f}) of {len(hist)} run\nmedian metrics\n{summary}\n")
    print((rd / "run_summary.txt").read_text())


def run_adapted(args):
    reg_tag = args.regional_tag if args.model == "TCN" else ""
    reg_dir = E.run_dir(args.model, "regional", args.seed, reg_tag)
    if not runio.is_done(reg_dir):
        raise FileNotFoundError(f"regional run not finished: {reg_dir}")
    ck = torch.load(reg_dir / "best_model.pt", map_location=DEV, weights_only=False)
    c = ck["cfg"]
    seq = c["seq_length"]
    rd = E.run_dir(args.model, "adapted", args.seed, reg_tag)
    if runio.is_done(rd) and not args.force:
        print(f"done already: {rd}")
        return
    basins = ck["basins"]
    statics = static_table(basins)
    lr = c["learning_rate"] * E.ADAPT["lr_factor"]
    if not (rd / "config.json").exists():
        runio.init_run(rd, {"model": args.model, "strategy": "adapted", "seed": args.seed, "regional_run": reg_dir,
                            "regional_hyperparameters": c, "fine_tune": {**E.ADAPT, "learning_rate": lr,
                                                                         "batch_size": c["batch_size"]},
                            "loss": E.LOSS_NAME})
    print(f"[{args.model} adapted seed {args.seed}] loss: {E.LOSS_NAME} | fine-tune lr {lr} from {reg_dir}", flush=True)
    data = Data(basins, "og", statics=statics)
    todo = [i for i in range(len(basins))[args.worker::args.n_workers] if not runio.part_done(rd, basins[i])]
    base = build(args.model, c, statics.shape[1])
    base.load_state_dict(ck["state_dict"])
    for i in todo:
        b = basins[i]
        seed_all(args.seed)
        model = copy.deepcopy(base)
        tb, tt = data.index(E.TRAIN, seq, basin_idx=i)
        vb, vt = data.index(E.VAL, seq, basin_idx=i)
        s0 = val_score(model, data, vb, vt, seq)
        best_epoch, best, hist = train(model, data, tb, tt, vb, vt, seq, lr, c["batch_size"], E.ADAPT["max_epochs"],
                                       E.ADAPT["patience"], start_score=s0)
        sims = simulate(model, [b], seq, {b: data.scalers[b]}, statics)
        runio.save_part(rd, b, {k: v[b] for k, v in sims.items()},
                        {"chosen_epoch": best_epoch, "val_NSE": best, "val_NSE_regional": s0, "epochs_run": len(hist)})
        print(f"  {b}: epoch {best_epoch} val NSE {s0:.3f} -> {best:.3f}", flush=True)
    if args.n_workers == 1:
        finalize_parts(args, rd, "adapted")


def individual_tag(model, args):
    """Empty for the production call (no overrides passed); a grid-search point gets its own tag so
    it writes to its own run dir instead of colliding with production or with other grid points."""
    if args.hidden_size or args.dropout or (model == "TCN" and args.num_filters is not None) or args.seq_length:
        bits = []
        if model == "LSTM":
            bits.append(f"h{args.hidden_size}") if args.hidden_size else None
        else:
            bits.append(f"f{args.num_filters}")
        bits.append(f"d{args.dropout}") if args.dropout else None
        bits.append(f"w{args.seq_length}") if args.seq_length else None
        return "_".join(bits)
    return ""


def run_individual(args):
    # Individual gets its own hyperparameter selection (separate from regional) -- only "adapted"
    # inherits regional's architecture, because it literally starts from the regional model's trained
    # weights. Individual trains from scratch per basin, so it is tuned on its own (user correction,
    # 2026-11-03; see run/select_lstm_individual.py / run/select_tcn_individual.py).
    c = dict(E.LSTM_INDIVIDUAL if args.model == "LSTM" else E.TCN_INDIVIDUAL)
    if args.hidden_size:
        c["hidden_size"] = args.hidden_size
    if args.dropout:
        c["dropout"] = args.dropout
    if args.model == "TCN" and args.num_filters is not None:
        c["num_filters"] = args.num_filters
    if args.seq_length:
        c["seq_length"] = args.seq_length
    if args.max_epochs:
        c["max_epochs"] = args.max_epochs
    seq = c["seq_length"]
    tag = individual_tag(args.model, args)
    rd = E.run_dir(args.model, "individual", args.seed, tag)
    if runio.is_done(rd) and not args.force:
        print(f"done already: {rd}")
        return
    basins = load_basin_pool()[: args.n_basins]
    if not (rd / "config.json").exists():
        runio.init_run(rd, {"model": args.model, "strategy": "individual", "seed": args.seed, "hyperparameters": c,
                            "loss": E.LOSS_NAME, "statics": None})
    todo = [b for b in basins[args.worker::args.n_workers] if not runio.part_done(rd, b)]
    print(f"[{args.model} individual seed {args.seed} worker {args.worker}] loss: {E.LOSS_NAME} | "
          f"{len(todo)} basins | {c}", flush=True)
    for b in todo:
        seed_all(args.seed)
        data = Data([b], "og")
        tb, tt = data.index(E.TRAIN, seq)
        vb, vt = data.index(E.VAL, seq)
        model = build(args.model, c, 0)
        best_epoch, best, hist = train(model, data, tb, tt, vb, vt, seq, c["learning_rate"], c["batch_size"],
                                       c["max_epochs"], c["patience"])
        sims = simulate(model, [b], seq, data.scalers, None)
        runio.save_part(rd, b, {k: v[b] for k, v in sims.items()},
                        {"best_epoch": best_epoch, "val_NSE": best, "epochs_run": len(hist)})
        print(f"  {b}: best epoch {best_epoch}/{len(hist)} val NSE {best:.3f}", flush=True)
    if args.n_workers == 1:
        finalize_parts(args, rd, "individual")


def finalize_parts(args, rd, strategy):
    basins = load_basin_pool()[: args.n_basins]
    missing = [b for b in basins if not runio.part_done(rd, b)]
    if missing:
        raise RuntimeError(f"{len(missing)} basins not finished in {rd}: {missing[:5]}...")
    summary = runio.finalize(rd)
    info = pd.DataFrame([{"basin": f.stem, **json.loads(f.read_text())} for f in sorted((rd / "parts").glob("*.json"))])
    name = "chosen_epochs.csv" if strategy == "adapted" else "best_epochs.csv"
    info.insert(1, "seed", args.seed)
    info.to_csv(rd / name, index=False)
    (rd / "run_summary.txt").write_text(f"{args.model} {strategy} seed {args.seed}\nloss {E.LOSS_NAME}\n"
                                        f"epochs: {info.iloc[:, 2].describe().to_dict()}\nmedian metrics\n{summary}\n")
    print((rd / "run_summary.txt").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", choices=["regional", "adapted", "individual"])
    ap.add_argument("--model", choices=["LSTM", "TCN"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_filters", type=int, default=None)
    ap.add_argument("--seq_length", type=int, default=None)
    ap.add_argument("--hidden_size", type=int, default=None, help="LSTM regional/individual hyperparameter search")
    ap.add_argument("--dropout", type=float, default=None, help="LSTM/TCN regional/individual hyperparameter search")
    ap.add_argument("--regional_tag", default="", help="TCN adapted: tag of the regional run, e.g. f64_w180")
    ap.add_argument("--max_epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--n_basins", type=int, default=None, help="smoke tests only, or individual HP search subset")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--finalize", action="store_true", help="merge parts of a multi-worker run")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.finalize:
        if a.strategy == "adapted" and a.model == "TCN":
            tag = a.regional_tag
        elif a.strategy == "individual":
            tag = individual_tag(a.model, a)
        else:
            tag = ""
        finalize_parts(a, E.run_dir(a.model, a.strategy, a.seed, tag), a.strategy)
    else:
        {"regional": run_regional, "adapted": run_adapted, "individual": run_individual}[a.strategy](a)
