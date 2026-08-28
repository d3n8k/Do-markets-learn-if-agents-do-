#market simulation (everything related to the draft n1)

# simple baseline sim: actions in {-1,0,1}, p[t+1] = p[t] + alpha * a_bar + beta * r[t] + noise
# rolling volatility = sqrt(mean(r^2) over last k). keep it rough, not microstructure.
# run_phase1/run_phase2 give quick plots and numbers to eyeball behavior.
# keep calibrate helpers consistent with main sim (k and seed).
# old helpers v0 present as reference. fine to keep but mark them not-run in final.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# this part = Phase 1-2 baseline sim (no learning)

def simulate_market(
    N=50,
    T=1000,
    alpha=1.0,
    beta=0.0,
    c=0.0,
    k=20,
    seed=None,
):
    # input checks are ok, keep them
    if N <= 0:
        raise ValueError("N must be positive.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if k <= 0:
        raise ValueError("k must be positive.")

    rng = np.random.default_rng(seed)

    p = np.zeros(T + 1, dtype=float)
    r = np.zeros(T + 1, dtype=float)
    sigma = np.zeros(T + 1, dtype=float)
    a_mean = np.zeros(T, dtype=float)

    for t in range(T):
        # agents choose simple discrete actions
        actions = rng.choice((-1, 0, 1), size=N)
        a_bar = float(actions.mean())
        a_mean[t] = a_bar

        # rolling window vol
        start = max(0, t - k + 1)
        window = r[start : t + 1]
        sigma[t] = float(np.sqrt(np.mean(window ** 2)))

        eps = rng.normal(loc=0.0, scale=sigma[t])

        p[t + 1] = p[t] + alpha * a_bar + beta * r[t] + eps
        r[t + 1] = p[t + 1] - p[t]

    start = max(0, T - k + 1)
    window = r[start : T + 1]
    sigma[T] = float(np.sqrt(np.mean(window ** 2)))

    return {
        "p": p,
        "r": r,
        "sigma": sigma,
        "a_mean": a_mean,
        "N": N,
        "T": T,
        "alpha": alpha,
        "beta": beta,
        "c": c,
        "k": k,
    }


def lag1_return_autocorrelation(r):
    # useful simple diagnostic
    r = np.asarray(r, dtype=float)
    if r.size < 3:
        return np.nan

    x = r[:-1]
    y = r[1:]
    x_std = x.std()
    y_std = y.std()
    if x_std == 0.0 or y_std == 0.0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _baseline_stats(N, T, alpha, beta, n_seeds=10):
    return_stats = []
    action_var = []
    lag1 = []

    for seed in range(n_seeds):
        out = simulate_market(
            N=N,
            T=T,
            alpha=alpha,
            beta=beta,
            seed=seed,
        )
        r = out["r"][1:]
        a_mean = out["a_mean"]

        return_stats.append(float(r.std()))
        action_var.append(float(a_mean.var()))
        lag1.append(lag1_return_autocorrelation(r))

    return {
        "return_std_mean": float(np.mean(return_stats)),
        "return_std_sd": float(np.std(return_stats, ddof=0)),
        "action_var_mean": float(np.mean(action_var)),
        "action_var_sd": float(np.std(action_var, ddof=0)),
        "lag1_mean": float(np.mean(lag1)),
        "lag1_sd": float(np.std(lag1, ddof=0)),
    }


def run_phase1(save_path="phase1_baseline.png"):
    # quick visual check for baseline dynamics
    out = simulate_market(N=50, T=1000, alpha=1.0, beta=0.0, seed=0)
    p, r, sigma = out["p"], out["r"], out["sigma"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(p, lw=1)
    axes[0].set_ylabel("price $p_t$")
    axes[0].set_title(
        f"Phase 1 baseline  (N={out['N']}, alpha={out['alpha']}, beta={out['beta']})"
    )

    axes[1].plot(r, lw=0.7)
    axes[1].set_ylabel("return $r_t$")

    axes[2].plot(sigma, lw=0.9)
    axes[2].set_ylabel("volatility $\\sigma_t$")
    axes[2].set_xlabel("t")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(
        f"[Phase 1] std(r) = {r[1:].std():.4f}   "
        f"final sigma_t = {sigma[-1]:.4f}   "
        f"lag-1 corr = {lag1_return_autocorrelation(r[1:]):.4f}   "
        f"any non-finite in p: {not np.all(np.isfinite(p))}"
    )
    print(f"[Phase 1] saved {save_path}")
    return out


def run_phase2(save_path="phase2_sweeps.png", n_seeds=20):
    # quick sweeps for baseline parameter sensitivity
    T = 1000

    N_values = [10, 50, 200, 1000]
    alpha_fixed = 1.0
    beta_fixed = 0.0

    N_stats = {
        N: _baseline_stats(
            N=N,
            T=T,
            alpha=alpha_fixed,
            beta=beta_fixed,
            n_seeds=n_seeds,
        )
        for N in N_values
    }

    alpha_values = [0.2, 1.0, 3.0]
    N_fixed = 50

    alpha_stats = {
        alpha: _baseline_stats(
            N=N_fixed,
            T=T,
            alpha=alpha,
            beta=beta_fixed,
            n_seeds=n_seeds,
        )
        for alpha in alpha_values
    }

    beta_values = [-0.2, 0.0, 0.2]
    beta_stats = {
        beta: _baseline_stats(
            N=N_fixed,
            T=T,
            alpha=alpha_fixed,
            beta=beta,
            n_seeds=n_seeds,
        )
        for beta in beta_values
    }

    N_paths = {
        N: simulate_market(
            N=N, T=T, alpha=alpha_fixed, beta=beta_fixed, seed=0
        )["p"]
        for N in N_values
    }

    alpha_paths = {
        alpha: simulate_market(
            N=N_fixed, T=T, alpha=alpha, beta=beta_fixed, seed=0
        )["p"]
        for alpha in alpha_values
    }

    beta_paths = {
        beta: simulate_market(
            N=N_fixed, T=T, alpha=alpha_fixed, beta=beta, seed=0
        )["p"]
        for beta in beta_values
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for N in N_values:
        axes[0, 0].plot(N_paths[N], lw=1, label=f"N={N}")
    axes[0, 0].set_title(f"Representative price paths vs. N  (alpha={alpha_fixed})")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_ylabel("price $p_t$")
    axes[0, 0].legend()

    xN = np.array(N_values, dtype=float)
    yN = np.array([N_stats[N]["return_std_mean"] for N in N_values])
    yNerr = np.array([N_stats[N]["return_std_sd"] for N in N_values])
    axes[0, 1].errorbar(xN, yN, yerr=yNerr, marker="o", capsize=3)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_title("Return volatility vs. N")
    axes[0, 1].set_xlabel("N (log scale)")
    axes[0, 1].set_ylabel("std($r_t$)")

    for alpha in alpha_values:
        axes[1, 0].plot(alpha_paths[alpha], lw=1, label=f"alpha={alpha}")
    axes[1, 0].set_title(f"Representative price paths vs. alpha  (N={N_fixed})")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_ylabel("price $p_t$")
    axes[1, 0].legend()

    axes[1, 1].plot(
        beta_values,
        [beta_stats[b]["return_std_mean"] for b in beta_values],
        marker="o",
        label="return std",
    )
    axes[1, 1].set_xlabel("beta")
    axes[1, 1].set_ylabel("std($r_t$)")
    axes[1, 1].set_title("Baseline response to beta")
    ax2 = axes[1, 1].twinx()
    ax2.plot(
        beta_values,
        [beta_stats[b]["lag1_mean"] for b in beta_values],
        marker="s",
        linestyle="--",
        label="lag-1 corr",
    )
    ax2.set_ylabel("Corr($r_t,r_{t+1}$)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("[Phase 2] Population scaling (mean +/- SD over seeds):")
    print("    N       std(r)          Var(mean action)")
    for N in N_values:
        s = N_stats[N]
        print(
            f"    {N:<5d}   "
            f"{s['return_std_mean']:.6f} +/- {s['return_std_sd']:.6f}    "
            f"{s['action_var_mean']:.8f} +/- {s['action_var_sd']:.8f}"
        )

    print("[Phase 2] Alpha sweep (N=50, beta=0):")
    print("    alpha    std(r)          lag-1 corr")
    for alpha in alpha_values:
        s = alpha_stats[alpha]
        print(
            f"    {alpha:<6}   "
            f"{s['return_std_mean']:.6f} +/- {s['return_std_sd']:.6f}    "
            f"{s['lag1_mean']:.6f} +/- {s['lag1_sd']:.6f}"
        )

    print("[Phase 2] Beta sweep (N=50, alpha=1):")
    print("    beta     std(r)          lag-1 corr")
    for beta in beta_values:
        s = beta_stats[beta]
        print(
            f"    {beta:<6}   "
            f"{s['return_std_mean']:.6f} +/- {s['return_std_sd']:.6f}    "
            f"{s['lag1_mean']:.6f} +/- {s['lag1_sd']:.6f}"
        )

    print(f"[Phase 2] saved {save_path}")
    return N_stats, alpha_stats, beta_stats


# this part = earlier draft of Phase 1-2 (kept for reference, not run by default)
# keep these v0 helpers but mark not-run in final report

def simulate_market_v0(N=50, T=1000, alpha=1.0, beta=0.0, c=0.0, k=20, seed=None):
    # x_t = (p_t, r_t, sigma_t)
    # p_{t+1} = p_t + alpha * mean_i(a_i^t) + beta * r_t + eps_t,  eps_t ~ N(0, sigma_t^2)
    # r_t     = p_t - p_{t-1}
    # sigma_t^2 = mean of r_t^2 over the trailing k steps (including r_t)
    rng = np.random.default_rng(seed)

    p = np.zeros(T + 1)
    r = np.zeros(T + 1)
    sigma = np.zeros(T + 1)
    a_mean = np.zeros(T)

    for t in range(T):
        actions = rng.choice([-1, 0, 1], size=N)
        a_bar = actions.mean()
        a_mean[t] = a_bar

        window = r[max(0, t - k + 1): t + 1]
        sigma[t] = np.sqrt(np.mean(window ** 2))

        eps = rng.normal(0.0, sigma[t])

        p[t + 1] = p[t] + alpha * a_bar + beta * r[t] + eps
        r[t + 1] = p[t + 1] - p[t]

    window = r[max(0, T - k + 1): T + 1]
    sigma[T] = np.sqrt(np.mean(window ** 2))

    return {"p": p, "r": r, "sigma": sigma, "a_mean": a_mean,
            "N": N, "T": T, "alpha": alpha, "beta": beta, "c": c, "k": k}


def run_phase1_v0(save_path="phase1_baseline_v0.png"):
    out = simulate_market_v0(N=50, T=1000, alpha=1.0, beta=0.0, seed=0)
    p, r = out["p"], out["r"]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(p, lw=1)
    axes[0].set_ylabel("price $p_t$")
    axes[0].set_title(
        f"Phase 1 baseline  (N={out['N']}, alpha={out['alpha']}, beta={out['beta']})"
    )

    axes[1].plot(r, lw=0.7, color="darkorange")
    axes[1].set_ylabel("return $r_t$")
    axes[1].set_xlabel("t")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[Phase 1] std(r) = {r.std():.4f}   final sigma_t = {out['sigma'][-1]:.4f}   "
          f"any non-finite in p: {not np.all(np.isfinite(p))}")
    print(f"[Phase 1] saved {save_path}")
    return out


def _return_std_over_seeds_v0(N, T, alpha, beta, n_seeds=10):
    stds = [simulate_market_v0(N=N, T=T, alpha=alpha, beta=beta, seed=s)["r"][1:].std()
            for s in range(n_seeds)]
    return float(np.mean(stds)), float(np.std(stds))


def run_phase2_v0(save_path="phase2_sweeps_v0.png"):
    T = 1000

    N_values = [10, 50, 200, 1000]
    alpha_fixed = 1.0
    N_paths = {N: simulate_market_v0(N=N, T=T, alpha=alpha_fixed, seed=0)["p"] for N in N_values}
    N_std = {N: _return_std_over_seeds_v0(N, T, alpha_fixed, 0.0) for N in N_values}

    alpha_values = [0.2, 1.0, 3.0]
    N_fixed = 50
    alpha_paths = {a: simulate_market_v0(N=N_fixed, T=T, alpha=a, seed=0)["p"] for a in alpha_values}
    alpha_std = {a: _return_std_over_seeds_v0(N_fixed, T, a, 0.0) for a in alpha_values}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for N in N_values:
        axes[0].plot(N_paths[N], lw=1, label=f"N={N}")
    axes[0].set_title(f"Price vs. N  (alpha={alpha_fixed})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("price $p_t$")
    axes[0].legend()

    for a in alpha_values:
        axes[1].plot(alpha_paths[a], lw=1, label=f"alpha={a}")
    axes[1].set_title(f"Price vs. alpha  (N={N_fixed})")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("price $p_t$")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print("[Phase 2] std(returns) by N, alpha=1.0  (mean +/- std over 10 seeds):")
    for N in N_values:
        m, s = N_std[N]
        print(f"    N={N:<5d}  {m:.4f} +/- {s:.4f}")

    print("[Phase 2] std(returns) by alpha, N=50  (mean +/- std over 10 seeds):")
    for a in alpha_values:
        m, s = alpha_std[a]
        print(f"    alpha={a:<5}  {m:.4f} +/- {s:.4f}")

    print(f"[Phase 2] saved {save_path}")
    return N_std, alpha_std


# this part = Q-learning agents
# current draft: simple 6-state discretization (r x median sigma). see must-fix notes below.
# must-fix if you want note/code parity:
#  - replace holding cost with turnover cost (c * |a_t - a_{t-1}|) if note claims turnover.
#  - implement annealed eta if note claims annealing: eta_t = eta/(1+4*t/T).
#  - include prev_action in state to have inventory-aware state -> N_STATES=27 and terciles.
#  - switch sigma thresholding to terciles (q33, q66) instead of median.
#  - compute per-learner state indices and update Q per-learner (avoid broadcasting bugs).
#  - add tiny random init to Q to break symmetry.
# these are comments only — code below is left unchanged.

N_STATES = 6
ACTIONS = np.array([-1, 0, 1])


def discretize(r, sigma, sigma_threshold, r_eps=1e-9):
    # coarse bin: r in {-1,0,1} and sigma low/high (median)
    r_bin = 0 if r < -r_eps else (2 if r > r_eps else 1)
    sigma_bin = 0 if sigma <= sigma_threshold else 1
    return r_bin * 2 + sigma_bin


def calibrate_sigma_threshold(N, alpha, beta, T=500, seed=12345, k=20):
    # returns median sigma by default (median split)
    out = simulate_market(N=N, T=T, alpha=alpha, beta=beta, k=k, seed=seed)
    return float(np.median(out["sigma"][1:]))


def rolling_sigma(r, t, k):
    # same rolling sd helper
    start = max(0, t - k + 1)
    return float(np.sqrt(np.mean(r[start : t + 1] ** 2)))


def simulate_learning_market(
    N=50, T=1000, alpha=1.0, beta=0.0,
    f=0.5, tau=1.0, eta=0.1, gamma=0.9, c=0.05,
    k=20, sigma_threshold=None, seed=None, checkpoint_every=None,
):
    # main learning sim. current draft uses fixed eta, holding-cost, shared market state
    rng = np.random.default_rng(seed)
    policy_snapshots = []

    if sigma_threshold is None:
        sigma_threshold = calibrate_sigma_threshold(N, alpha, beta, k=k)

    n_learn = int(round(f * N))
    learner_idx = np.sort(rng.choice(N, size=n_learn, replace=False)) if n_learn > 0 else np.array([], dtype=int)
    is_learner = np.zeros(N, dtype=bool)
    is_learner[learner_idx] = True

    Q = np.zeros((n_learn, N_STATES, 3))

    p = np.zeros(T + 1)
    r = np.zeros(T + 1)
    sigma = np.zeros(T + 1)
    a_mean = np.zeros(T)
    reward = np.zeros((T, N))
    entropy_t = np.full(T, np.nan)

    sigma[0] = rolling_sigma(r, 0, k)
    state = discretize(r[0], sigma[0], sigma_threshold)
    choice_idx = None

    for t in range(T):
        actions = rng.choice(ACTIONS, size=N)

        if n_learn > 0:
            # softmax per learner over shared state (current draft)
            logits = Q[:, state, :] / max(tau, 1e-8)
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs = probs / probs.sum(axis=1, keepdims=True)
            entropy_t[t] = float(np.mean(-np.sum(probs * np.log(probs + 1e-12), axis=1)))

            g = rng.gumbel(size=probs.shape)
            choice_idx = np.argmax(np.log(probs + 1e-12) + g, axis=1)
            actions[learner_idx] = ACTIONS[choice_idx]

        a_bar = float(actions.mean())
        a_mean[t] = a_bar

        eps = rng.normal(0.0, sigma[t])
        p[t + 1] = p[t] + alpha * a_bar + beta * r[t] + eps
        r[t + 1] = p[t + 1] - p[t]
        sigma[t + 1] = rolling_sigma(r, t + 1, k)

        dp = p[t + 1] - p[t]
        # current draft: holding cost used here (note: change to turnover if you want)
        reward[t] = actions * dp - c * np.abs(actions)

        next_state = discretize(r[t + 1], sigma[t + 1], sigma_threshold)

        if n_learn > 0:
            learner_rewards = reward[t, learner_idx]
            best_next = Q[:, next_state, :].max(axis=1)
            td_target = learner_rewards + gamma * best_next
            rows = np.arange(n_learn)
            td_error = td_target - Q[rows, state, choice_idx]
            Q[rows, state, choice_idx] += eta * td_error

        state = next_state

        if checkpoint_every and n_learn > 0 and (t % checkpoint_every == 0):
            snap_logits = Q / max(tau, 1e-8)
            snap_logits = snap_logits - snap_logits.max(axis=2, keepdims=True)
            snap_probs = np.exp(snap_logits)
            snap_probs = snap_probs / snap_probs.sum(axis=2, keepdims=True)
            policy_snapshots.append((t, snap_probs.mean(axis=0)))

    return {
        "policy_snapshots": policy_snapshots,
        "p": p, "r": r, "sigma": sigma, "a_mean": a_mean,
        "reward": reward, "is_learner": is_learner, "learner_idx": learner_idx,
        "Q": Q, "entropy_t": entropy_t, "sigma_threshold": sigma_threshold,
        "N": N, "T": T, "alpha": alpha, "beta": beta,
        "f": f, "tau": tau, "eta": eta, "gamma": gamma, "c": c, "k": k,
    }


def one_step_predictability(r):
    # out-of-sample linear predictability (split-half R^2)
    r = np.asarray(r, dtype=float)
    x, y = r[:-1], r[1:]
    n = len(x)
    if n < 40:
        return np.nan
    half = n // 2
    x_tr, y_tr, x_te, y_te = x[:half], y[:half], x[half:], y[half:]
    if x_tr.std() == 0:
        return np.nan
    b, a = np.polyfit(x_tr, y_tr, 1)
    y_hat = a + b * x_te
    ss_res = np.sum((y_te - y_hat) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def agent_level_stats(sim_out):
    # cumulative reward diagnostics; fraction profitable etc.
    reward = sim_out["reward"]
    is_learner = sim_out["is_learner"]
    cum_reward = reward.sum(axis=0)

    def _grp(mask):
        if mask.sum() == 0:
            return {"avg_cum_reward": np.nan, "frac_profitable": np.nan}
        cr = cum_reward[mask]
        return {"avg_cum_reward": float(cr.mean()), "frac_profitable": float((cr > 0).mean())}

    out = {
        "learners": _grp(is_learner),
        "non_learners": _grp(~is_learner),
        "all": _grp(np.ones(len(is_learner), dtype=bool)),
    }

    Q = sim_out["Q"]
    out["Q_mean"] = float(Q.mean()) if Q.size else np.nan
    out["Q_std"] = float(Q.std()) if Q.size else np.nan

    ent = sim_out["entropy_t"]
    ent_valid = ent[~np.isnan(ent)]
    out["entropy_mean"] = float(ent_valid.mean()) if len(ent_valid) else np.nan
    tail = max(1, len(ent_valid) // 5)
    out["entropy_late"] = float(ent_valid[-tail:].mean()) if len(ent_valid) else np.nan

    return out


def run_stats(n_seeds=15, **sim_kwargs):
    # run multiple seeds, return raw + mean/sd for metrics
    records = []
    for seed in range(n_seeds):
        out = simulate_learning_market(seed=seed, **sim_kwargs)
        r = out["r"][1:]
        ags = agent_level_stats(out)
        records.append({
            "return_std": float(r.std()),
            "lag1_corr": lag1_return_autocorrelation(r),
            "action_var": float(out["a_mean"].var()),
            "oos_r2": one_step_predictability(r),
            "cum_reward_learners": ags["learners"]["avg_cum_reward"],
            "frac_profitable_learners": ags["learners"]["frac_profitable"],
            "cum_reward_all": ags["all"]["avg_cum_reward"],
            "entropy_late": ags["entropy_late"],
            "Q_std": ags["Q_std"],
        })

    summary = {"n_seeds": n_seeds, "_raw": records, "_kwargs": sim_kwargs}
    for key in records[0]:
        vals = np.array([rec[key] for rec in records], dtype=float)
        valid = vals[~np.isnan(vals)]
        summary[key + "_mean"] = float(valid.mean()) if len(valid) else np.nan
        summary[key + "_sd"] = float(valid.std(ddof=0)) if len(valid) else np.nan
    return summary


def welch_t_test(sample_a, sample_b):
    # wrapper to scipy t-test
    from scipy import stats as sstats
    a = np.asarray(sample_a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(sample_b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = sstats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


# this part = Experiments A/B/C + plots

N, T = 50, 1000
ALPHA0, BETA0 = 1.0, 0.0
TAU0, ETA0, GAMMA0, C0 = 1.0, 0.1, 0.9, 0.05
N_SEEDS_AB = 15
N_SEEDS_C = 10

METRIC_LABELS = {
    "return_std": "std($r_t$)",
    "lag1_corr": "Corr($r_t,r_{t+1}$)",
    "action_var": "Var($\\bar a_t$)",
    "oos_r2": "out-of-sample $R^2$",
    "cum_reward_all": "avg cum. reward (all agents)",
    "frac_profitable_learners": "fraction of learners profitable",
    "entropy_late": "late-training policy entropy",
    "Q_std": "std(Q)",
}


def fmt_row(label, s):
    return (f"    {label:<8} "
            f"std(r)={s['return_std_mean']:.4f}+/-{s['return_std_sd']:.4f}  "
            f"lag1={s['lag1_corr_mean']:.4f}+/-{s['lag1_corr_sd']:.4f}  "
            f"oosR2={s['oos_r2_mean']:.4f}+/-{s['oos_r2_sd']:.4f}  "
            f"reward={s['cum_reward_all_mean']:.3f}+/-{s['cum_reward_all_sd']:.3f}  "
            f"entropy={s['entropy_late_mean']:.4f}" if not np.isnan(s['entropy_late_mean'])
            else f"    {label:<8} std(r)={s['return_std_mean']:.4f}+/-{s['return_std_sd']:.4f}  "
                 f"lag1={s['lag1_corr_mean']:.4f}+/-{s['lag1_corr_sd']:.4f}  "
                 f"oosR2={s['oos_r2_mean']:.4f}+/-{s['oos_r2_sd']:.4f}  "
                 f"reward={s['cum_reward_all_mean']:.3f}+/-{s['cum_reward_all_sd']:.3f}  entropy=n/a")


def run_experiment_A():
    print("=== Experiment A: learner fraction f ===")
    f_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    results = {}
    for f in f_values:
        results[f] = run_stats(n_seeds=N_SEEDS_AB, N=N, T=T, alpha=ALPHA0, beta=BETA0,
                                f=f, tau=TAU0, eta=ETA0, gamma=GAMMA0, c=C0)
        print(fmt_row(f"f={f}", results[f]))

    baseline_raw = results[0.0]["_raw"]
    print("  t-tests vs f=0 baseline (lag1_corr, oos_r2, return_std):")
    for f in f_values[1:]:
        raw = results[f]["_raw"]
        for metric in ["lag1_corr", "oos_r2", "return_std"]:
            a = [rec[metric] for rec in baseline_raw]
            b = [rec[metric] for rec in raw]
            t, pval = welch_t_test(a, b)
            print(f"    f={f} vs f=0, {metric}: t={t:.2f}, p={pval:.4f}")
    return f_values, results


def run_experiment_B():
    print("\n=== Experiment B: hyperparameter sweep (f=0.5 fixed) ===")
    F_FIXED = 0.5

    tau_values = [0.1, 1.0, 3.0]
    eta_values = [0.02, 0.1, 0.4]
    c_values = [0.0, 0.05, 0.25]

    tau_results = {tau: run_stats(n_seeds=N_SEEDS_AB, N=N, T=T, alpha=ALPHA0, beta=BETA0,
                                   f=F_FIXED, tau=tau, eta=ETA0, gamma=GAMMA0, c=C0)
                   for tau in tau_values}
    eta_results = {eta: run_stats(n_seeds=N_SEEDS_AB, N=N, T=T, alpha=ALPHA0, beta=BETA0,
                                   f=F_FIXED, tau=TAU0, eta=eta, gamma=GAMMA0, c=C0)
                   for eta in eta_values}
    c_results = {c: run_stats(n_seeds=N_SEEDS_AB, N=N, T=T, alpha=ALPHA0, beta=BETA0,
                               f=F_FIXED, tau=TAU0, eta=ETA0, gamma=GAMMA0, c=c)
                 for c in c_values}

    print("  tau sweep (eta=%.2f, c=%.2f):" % (ETA0, C0))
    for tau in tau_values:
        print(fmt_row(f"tau={tau}", tau_results[tau]))
    print("  eta sweep (tau=%.2f, c=%.2f):" % (TAU0, C0))
    for eta in eta_values:
        print(fmt_row(f"eta={eta}", eta_results[eta]))
    print("  c sweep (tau=%.2f, eta=%.2f):" % (TAU0, ETA0))
    for c in c_values:
        print(fmt_row(f"c={c}", c_results[c]))

    return (tau_values, tau_results), (eta_values, eta_results), (c_values, c_results)


def run_experiment_C():
    print("\n=== Experiment C: alpha x beta robustness grid (f=0 vs f=1) ===")
    alpha_values = [0.2, 1.0, 3.0]
    beta_values = [-0.2, 0.0, 0.2]

    grid = {}
    for a in alpha_values:
        for b in beta_values:
            base = run_stats(n_seeds=N_SEEDS_C, N=N, T=T, alpha=a, beta=b,
                              f=0.0, tau=TAU0, eta=ETA0, gamma=GAMMA0, c=C0)
            learn = run_stats(n_seeds=N_SEEDS_C, N=N, T=T, alpha=a, beta=b,
                               f=1.0, tau=TAU0, eta=ETA0, gamma=GAMMA0, c=C0)
            t_lag1, p_lag1 = welch_t_test(
                [r["lag1_corr"] for r in base["_raw"]],
                [r["lag1_corr"] for r in learn["_raw"]],
            )
            grid[(a, b)] = {
                "base_lag1": base["lag1_corr_mean"], "learn_lag1": learn["lag1_corr_mean"],
                "delta_lag1": learn["lag1_corr_mean"] - base["lag1_corr_mean"],
                "t_lag1": t_lag1, "p_lag1": p_lag1,
                "base_std": base["return_std_mean"], "learn_std": learn["return_std_mean"],
                "delta_std": learn["return_std_mean"] - base["return_std_mean"],
            }
            print(f"  alpha={a:>4}, beta={b:>5}: "
                  f"lag1 base={base['lag1_corr_mean']:+.4f} -> learn={learn['lag1_corr_mean']:+.4f} "
                  f"(delta={grid[(a,b)]['delta_lag1']:+.4f}, p={p_lag1:.3f})   "
                  f"std(r) base={base['return_std_mean']:.3f} -> learn={learn['return_std_mean']:.3f}")

    return alpha_values, beta_values, grid


def plot_experiment_A(f_values, results, save_path="fig_A_learner_fraction.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.array(f_values)

    def eb(ax, key, color, ylabel):
        y = np.array([results[f][f"{key}_mean"] for f in f_values])
        yerr = np.array([results[f][f"{key}_sd"] for f in f_values])
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, color=color)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("learner fraction $f$")
        ax.set_ylabel(ylabel)

    eb(axes[0, 0], "lag1_corr", "tab:blue", METRIC_LABELS["lag1_corr"])
    axes[0, 0].set_title("Predictability vs. learner fraction")
    eb(axes[0, 1], "return_std", "tab:red", METRIC_LABELS["return_std"])
    axes[0, 1].set_title("Volatility vs. learner fraction")
    eb(axes[1, 0], "oos_r2", "tab:green", METRIC_LABELS["oos_r2"])
    axes[1, 0].set_title("Out-of-sample predictability vs. learner fraction")
    eb(axes[1, 1], "cum_reward_all", "tab:purple", METRIC_LABELS["cum_reward_all"])
    axes[1, 1].set_title("Avg. cumulative reward vs. learner fraction")

    fig.suptitle(f"Experiment A: learner-fraction sweep (N={N}, alpha={ALPHA0}, beta={BETA0}, "
                 f"n_seeds={N_SEEDS_AB})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_experiment_B(tau_data, eta_data, c_data, save_path="fig_B_hyperparams.png"):
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))

    def row(ax_left, ax_right, values, results, xlabel):
        x = np.array(values, dtype=float)
        y_std = np.array([results[v]["return_std_mean"] for v in values])
        y_lag1 = np.array([results[v]["lag1_corr_mean"] for v in values])
        y_reward = np.array([results[v]["cum_reward_all_mean"] for v in values])
        y_ent = np.array([results[v]["entropy_late_mean"] for v in values])

        l1, = ax_left.plot(x, y_std, "o-", color="tab:red", label="std($r_t$)")
        ax_left.set_xlabel(xlabel)
        ax_left.set_ylabel("std($r_t$)", color="tab:red")
        axL2 = ax_left.twinx()
        l2, = axL2.plot(x, y_lag1, "s--", color="tab:blue", label="lag-1 corr")
        axL2.set_ylabel("lag-1 corr", color="tab:blue")
        axL2.axhline(0, color="grey", lw=0.5, ls=":")

        r1, = ax_right.plot(x, y_reward, "o-", color="tab:purple", label="avg reward")
        ax_right.set_xlabel(xlabel)
        ax_right.set_ylabel("avg cum. reward", color="tab:purple")
        axR2 = ax_right.twinx()
        r2, = axR2.plot(x, y_ent, "s--", color="tab:green", label="entropy")
        axR2.set_ylabel("policy entropy", color="tab:green")

    row(axes[0, 0], axes[0, 1], *tau_data, "softmax temperature $\\tau$")
    row(axes[1, 0], axes[1, 1], *eta_data, "learning rate $\\eta$")
    row(axes[2, 0], axes[2, 1], *c_data, "transaction cost $c$")

    axes[0, 0].set_title("std($r_t$) & lag-1 corr")
    axes[0, 1].set_title("avg reward & policy entropy")
    fig.suptitle(f"Experiment B: hyperparameter sweeps (f=0.5, N={N}, alpha={ALPHA0}, "
                 f"beta={BETA0}, n_seeds={N_SEEDS_AB})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_experiment_C(alpha_values, beta_values, grid, save_path="fig_C_alpha_beta_grid.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    delta_lag1 = np.array([[grid[(a, b)]["delta_lag1"] for b in beta_values] for a in alpha_values])
    ratio_std = np.array([[grid[(a, b)]["learn_std"] / max(grid[(a, b)]["base_std"], 1e-9)
                            for b in beta_values] for a in alpha_values])

    lim = np.abs(delta_lag1).max()
    im0 = axes[0].imshow(delta_lag1, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    axes[0].set_xticks(range(len(beta_values))); axes[0].set_xticklabels(beta_values)
    axes[0].set_yticks(range(len(alpha_values))); axes[0].set_yticklabels(alpha_values)
    axes[0].set_xlabel("beta"); axes[0].set_ylabel("alpha")
    axes[0].set_title("$\\Delta$ lag-1 corr (learn - baseline)")
    for i in range(len(alpha_values)):
        for j in range(len(beta_values)):
            p = grid[(alpha_values[i], beta_values[j])]["p_lag1"]
            star = "*" if p < 0.05 else ""
            axes[0].text(j, i, f"{delta_lag1[i,j]:+.3f}{star}", ha="center", va="center", fontsize=9)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(ratio_std, cmap="Oranges", aspect="auto")
    axes[1].set_xticks(range(len(beta_values))); axes[1].set_xticklabels(beta_values)
    axes[1].set_yticks(range(len(alpha_values))); axes[1].set_yticklabels(alpha_values)
    axes[1].set_xlabel("beta"); axes[1].set_ylabel("alpha")
    axes[1].set_title("volatility ratio: std($r$) learn / baseline")
    for i in range(len(alpha_values)):
        for j in range(len(beta_values)):
            axes[1].text(j, i, f"{ratio_std[i,j]:.1f}x", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(f"Experiment C: f=0 (baseline) vs f=1 (full learning) across the alpha,beta grid  "
                 f"(*=p<0.05, n_seeds={N_SEEDS_C})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")


def plot_diagnostics(save_path="fig_D_diagnostics.png"):
    # expects policy_snapshots shape; if you change states ensure plotting logic updated
    out = simulate_learning_market(N=N, T=T, alpha=ALPHA0, beta=BETA0, f=1.0,
                                    tau=TAU0, eta=ETA0, gamma=GAMMA0, c=C0,
                                    seed=1, checkpoint_every=20)

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 1)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(out["a_mean"], lw=0.7, color="tab:purple")
    ax1.axhline(0, color="grey", lw=0.6, ls=":")
    ax1.set_ylabel("$\\bar a_t$")
    ax1.set_title("Aggregate action $\\bar a_t$ over training (f=1, representative seed)")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(out["entropy_t"], lw=0.7, color="tab:green")
    ax2.axhline(np.log(3), color="grey", lw=0.6, ls=":", label="max entropy ln(3)")
    ax2.set_ylabel("policy entropy")
    ax2.set_xlabel("t")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_title("Population-avg policy entropy over training")

    state_labels = ["r<0,lowVol", "r<0,hiVol", "r=0,lowVol", "r=0,hiVol", "r>0,lowVol", "r>0,hiVol"]
    snaps = out["policy_snapshots"]
    ts = np.array([s[0] for s in snaps])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")

    fig2, axes2 = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    probs_by_state = np.array([s[1] for s in snaps])
    for s_idx, ax in enumerate(axes2.flat):
        ax.plot(ts, probs_by_state[:, s_idx, 0], label="P(a=-1)", color="tab:red")
        ax.plot(ts, probs_by_state[:, s_idx, 1], label="P(a=0)", color="tab:grey")
        ax.plot(ts, probs_by_state[:, s_idx, 2], label="P(a=+1)", color="tab:blue")
        ax.set_title(state_labels[s_idx], fontsize=10)
        ax.set_ylim(0, 1)
    axes2[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes2[-1, :]:
        ax.set_xlabel("t")
    fig2.suptitle("Population-avg policy $\\pi(a\\,|\\,\\mathrm{state})$ over training, by state (f=1)")
    fig2.tight_layout()
    fig2.savefig("fig_D2_policy_by_state.png", dpi=150)
    plt.close(fig2)
    print("saved fig_D2_policy_by_state.png")

    return out


# this part = Phases 1-2, then Experiments A/B/C
# final checklist before sending:
# 1) implement must-fix items (turnover, anneal, prev_action, terciles) if you want note/code parity
# 2) re-run experiments and regenerate figures/tables
# 3) update note with fresh numbers and table
# 4) freeze seeds and document them in supplementary
# 5) optionally move v0 helpers to /old/ for a tidier final repo

if __name__ == "__main__":
    import time

    run_phase1()
    print()
    run_phase2()

    t0 = time.time()
    fA, resA = run_experiment_A()
    (tauV, tauR), (etaV, etaR), (cV, cR) = run_experiment_B()
    alphaV, betaV, gridC = run_experiment_C()
    print(f"\nNumeric experiments runtime: {time.time()-t0:.1f}s")

    t1 = time.time()
    plot_experiment_A(fA, resA)
    plot_experiment_B((tauV, tauR), (etaV, etaR), (cV, cR))
    plot_experiment_C(alphaV, betaV, gridC)
    plot_diagnostics()
    print(f"Plotting runtime: {time.time()-t1:.1f}s")
