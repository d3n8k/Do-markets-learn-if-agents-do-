""

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# PART 0 -- baseline market + v1 utilities 


def simulate_market(
    N=50,
    T=1000,
    alpha=1.0,
    beta=0.0,
    c=0.0,
    k=20,
    seed=None,
):
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
        actions = rng.choice((-1, 0, 1), size=N)
        a_bar = float(actions.mean())
        a_mean[t] = a_bar

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


def one_step_predictability(r):
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


def welch_t_test(sample_a, sample_b):
    from scipy import stats as sstats
    a = np.asarray(sample_a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(sample_b, dtype=float); b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    t, p = sstats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


N_STATES = 6
ACTIONS = np.array([-1, 0, 1])


def discretize(r, sigma, sigma_threshold, r_eps=1e-9):
    r_bin = 0 if r < -r_eps else (2 if r > r_eps else 1)
    sigma_bin = 0 if sigma <= sigma_threshold else 1
    return r_bin * 2 + sigma_bin


def calibrate_sigma_threshold(N, alpha, beta, T=500, seed=12345, k=20):
    out = simulate_market(N=N, T=T, alpha=alpha, beta=beta, k=k, seed=seed)
    return float(np.median(out["sigma"][1:]))


def rolling_sigma(r, t, k):
    start = max(0, t - k + 1)
    return float(np.sqrt(np.mean(r[start : t + 1] ** 2)))


# =====================================================================
# PART 1 -- v2 extension: annealed learning rate, turnover-based cost,
# inventory-aware state (r_bin(3) x vol_bin(3) x own_prev_action(3) = 27
# states), tercile volatility split
# =====================================================================

N_STATES_V2 = 27


def discretize_v2(r, sigma, sigma_thresholds, own_prev_action, r_eps=1e-9):
    """Scalar version (market-level r, sigma; scalar own action)."""
    r_bin = 0 if r < -r_eps else (2 if r > r_eps else 1)
    lo, hi = sigma_thresholds
    vol_bin = 0 if sigma <= lo else (1 if sigma <= hi else 2)
    act_bin = int(own_prev_action) + 1
    return (r_bin * 3 + vol_bin) * 3 + act_bin


def discretize_v2_batch(r, sigma, sigma_thresholds, own_prev_actions, r_eps=1e-9):
    """Vectorized version: r, sigma are scalars (shared market state);
    own_prev_actions is an array (one entry per learner)."""
    r_bin = 0 if r < -r_eps else (2 if r > r_eps else 1)
    lo, hi = sigma_thresholds
    vol_bin = 0 if sigma <= lo else (1 if sigma <= hi else 2)
    act_bin = own_prev_actions.astype(int) + 1
    base = r_bin * 3 + vol_bin
    return base * 3 + act_bin


def calibrate_sigma_thresholds_v2(N, alpha, beta, T=500, seed=12345, k=20):
    out = simulate_market(N=N, T=T, alpha=alpha, beta=beta, k=k, seed=seed)
    s = out["sigma"][1:]
    lo, hi = np.percentile(s, [33.0, 66.0])
    return float(lo), float(hi)


def eta_schedule(eta0, t, T, decay=4.0):
    """Robbins-Monro-style decay: eta_t ~ 1/t asymptotically.
    decay is a fixed constant of the functional form (like k=20
    elsewhere in the model), not an extra tunable hyperparameter."""
    return eta0 / (1.0 + decay * t / T)


def simulate_learning_market_v2(
    N=50, T=1000, alpha=1.0, beta=0.0,
    f=0.5, tau=1.0, eta0=0.1, eta_decay=4.0, gamma=0.9, c=0.05,
    k=20, sigma_thresholds=None, seed=None, checkpoint_every=None,
):
    rng = np.random.default_rng(seed)
    policy_snapshots = []

    if sigma_thresholds is None:
        sigma_thresholds = calibrate_sigma_thresholds_v2(N, alpha, beta, k=k)

    n_learn = int(round(f * N))
    learner_idx = np.sort(rng.choice(N, size=n_learn, replace=False)) if n_learn > 0 else np.array([], dtype=int)
    is_learner = np.zeros(N, dtype=bool)
    is_learner[learner_idx] = True

    Q = np.zeros((n_learn, N_STATES_V2, 3))

    p = np.zeros(T + 1)
    r = np.zeros(T + 1)
    sigma = np.zeros(T + 1)
    a_mean = np.zeros(T)
    reward = np.zeros((T, N))
    entropy_t = np.full(T, np.nan)
    qstd_t = np.full(T, np.nan)
    eta_t_track = np.full(T, np.nan)

    prev_actions = np.zeros(N)  # every agent starts flat (position 0)
    sigma[0] = rolling_sigma(r, 0, k)

    states = (
        discretize_v2_batch(r[0], sigma[0], sigma_thresholds, prev_actions[learner_idx])
        if n_learn > 0 else np.array([], dtype=int)
    )
    choice_idx = None

    for t in range(T):
        actions = rng.choice(ACTIONS, size=N)

        if n_learn > 0:
            eta_t = eta_schedule(eta0, t, T, decay=eta_decay)
            eta_t_track[t] = eta_t

            logits = Q[np.arange(n_learn), states, :] / max(tau, 1e-8)
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
        reward[t] = actions * dp - c * np.abs(actions - prev_actions)

        if n_learn > 0:
            next_states = discretize_v2_batch(
                r[t + 1], sigma[t + 1], sigma_thresholds, actions[learner_idx]
            )

            learner_rewards = reward[t, learner_idx]
            best_next = Q[np.arange(n_learn), next_states, :].max(axis=1)
            td_target = learner_rewards + gamma * best_next
            rows = np.arange(n_learn)
            td_error = td_target - Q[rows, states, choice_idx]
            Q[rows, states, choice_idx] += eta_t * td_error

            states = next_states
            qstd_t[t] = float(Q.std())

        prev_actions = actions.copy()

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
        "Q": Q, "entropy_t": entropy_t, "qstd_t": qstd_t, "eta_t": eta_t_track,
        "sigma_thresholds": sigma_thresholds,
        "N": N, "T": T, "alpha": alpha, "beta": beta,
        "f": f, "tau": tau, "eta0": eta0, "eta_decay": eta_decay, "gamma": gamma, "c": c, "k": k,
    }


def agent_level_stats_v2(sim_out):
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


def run_stats_v2(n_seeds=20, **sim_kwargs):
    records = []
    for seed in range(n_seeds):
        out = simulate_learning_market_v2(seed=seed, **sim_kwargs)
        r = out["r"][1:]
        ags = agent_level_stats_v2(out)
        records.append({
            "return_std": float(r.std()),
            "lag1_corr": lag1_return_autocorrelation(r),
            "action_var": float(out["a_mean"].var()),
            "oos_r2": one_step_predictability(r),
            "cum_reward_learners": ags["learners"]["avg_cum_reward"],
            "frac_profitable_learners": ags["learners"]["frac_profitable"],
            "cum_reward_all": ags["all"]["avg_cum_reward"],
            "frac_profitable_all": ags["all"]["frac_profitable"],
            "entropy_late": ags["entropy_late"],
            "Q_std": ags["Q_std"],
        })

    summary = {"n_seeds": n_seeds, "_raw": records, "_kwargs": sim_kwargs}
    for key in records[0]:
        vals = np.array([rec[key] for rec in records], dtype=float)
        valid = vals[~np.isnan(vals)]
        summary[key + "_mean"] = float(valid.mean()) if len(valid) else np.nan
        summary[key + "_sd"] = float(valid.std(ddof=0)) if len(valid) else np.nan
        summary[key + "_se"] = float(valid.std(ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else np.nan
    return summary



# PART 2 -- v1 diagnostic wrapper: the ORIGINAL rule (fixed eta, 6-state, holding cost)


def simulate_learning_market_v1_diag(
    N=50, T=1000, alpha=1.0, beta=0.0,
    f=1.0, tau=1.0, eta=0.1, gamma=0.9, c=0.05,
    k=20, sigma_threshold=None, seed=None,
):
    rng = np.random.default_rng(seed)

    if sigma_threshold is None:
        sigma_threshold = calibrate_sigma_threshold(N, alpha, beta, k=k)

    n_learn = int(round(f * N))
    learner_idx = np.sort(rng.choice(N, size=n_learn, replace=False)) if n_learn > 0 else np.array([], dtype=int)

    Q = np.zeros((n_learn, N_STATES, 3))

    p = np.zeros(T + 1)
    r = np.zeros(T + 1)
    sigma = np.zeros(T + 1)
    a_mean = np.zeros(T)
    entropy_t = np.full(T, np.nan)
    qstd_t = np.full(T, np.nan)

    sigma[0] = rolling_sigma(r, 0, k)
    state = discretize(r[0], sigma[0], sigma_threshold)
    choice_idx = None

    for t in range(T):
        actions = rng.choice(ACTIONS, size=N)

        if n_learn > 0:
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
        reward_t = actions * dp - c * np.abs(actions)

        next_state = discretize(r[t + 1], sigma[t + 1], sigma_threshold)

        if n_learn > 0:
            learner_rewards = reward_t[learner_idx]
            best_next = Q[:, next_state, :].max(axis=1)
            td_target = learner_rewards + gamma * best_next
            rows = np.arange(n_learn)
            td_error = td_target - Q[rows, state, choice_idx]
            Q[rows, state, choice_idx] += eta * td_error
            qstd_t[t] = float(Q.std())

        state = next_state

    return {"p": p, "r": r, "sigma": sigma, "a_mean": a_mean,
            "entropy_t": entropy_t, "qstd_t": qstd_t}



# PART 3 -- Experiments A, B, C + final table + plotting + main


RNG_SEED_BASE = 0
N, T = 50, 1000
ALPHA0, BETA0 = 1.0, 0.0
TAU0, ETA0, ETA_DECAY0, GAMMA0, C0 = 1.0, 0.1, 4.0, 0.9, 0.05
N_SEEDS_A = 20
N_SEEDS_B = 15

OUT = {}


# Experiment A: learner-fraction sweep (core result)


def run_experiment_A():
    print("=== Experiment A: learner fraction f (v2 model) ===")
    f_values = [0.0, 0.1, 0.25, 0.5, 1.0]
    results = {}
    for f in f_values:
        results[f] = run_stats_v2(
            n_seeds=N_SEEDS_A, N=N, T=T, alpha=ALPHA0, beta=BETA0,
            f=f, tau=TAU0, eta0=ETA0, eta_decay=ETA_DECAY0, gamma=GAMMA0, c=C0,
        )
        s = results[f]
        prof = s["frac_profitable_learners_mean"]
        prof_str = f"{prof*100:5.1f}%" if not np.isnan(prof) else "   n/a"
        ent = s["entropy_late_mean"]
        ent_str = f"{ent:.4f}" if not np.isnan(ent) else "   n/a"
        print(f"  f={f:<4} std(r)={s['return_std_mean']:.4f}+/-{s['return_std_sd']:.4f}  "
              f"lag1={s['lag1_corr_mean']:+.4f}+/-{s['lag1_corr_sd']:.4f}  "
              f"oosR2={s['oos_r2_mean']:+.4f}+/-{s['oos_r2_sd']:.4f}  "
              f"%profit(learners)={prof_str}  entropy={ent_str}")

    baseline_raw = results[0.0]["_raw"]
    print("  Welch t-tests vs f=0 baseline:")
    sig = {}
    for f in f_values[1:]:
        raw = results[f]["_raw"]
        sig[f] = {}
        for metric in ["lag1_corr", "oos_r2", "return_std"]:
            a = [rec[metric] for rec in baseline_raw]
            b = [rec[metric] for rec in raw]
            t, pval = welch_t_test(a, b)
            sig[f][metric] = (t, pval)
            print(f"    f={f} vs f=0, {metric}: t={t:.2f}, p={pval:.4f}")

    OUT["expA_f_values"] = f_values
    OUT["expA_results"] = results
    OUT["expA_sig"] = sig
    return f_values, results, sig


def plot_experiment_A(f_values, results, save_path="fig1_learner_fraction.png"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    x = np.array(f_values)

    def eb(ax, key, color, ylabel, mask_zero_learners=False):
        y = np.array([results[f][f"{key}_mean"] for f in f_values])
        yerr = np.array([results[f][f"{key}_sd"] for f in f_values])
        xx, yy, yyerr = x, y, yerr
        if mask_zero_learners:
            keep = ~np.isnan(y)
            xx, yy, yyerr = x[keep], y[keep], yerr[keep]
        ax.errorbar(xx, yy, yerr=yyerr, marker="o", capsize=3, color=color)
        ax.axhline(0, color="grey", lw=0.6, ls=":")
        ax.set_xlabel("learner fraction $f$")
        ax.set_ylabel(ylabel)

    eb(axes[0, 0], "lag1_corr", "tab:blue", "Corr($r_t,r_{t+1}$)")
    axes[0, 0].set_title("Predictability vs. learner fraction")
    eb(axes[0, 1], "return_std", "tab:red", "std($r_t$)")
    axes[0, 1].set_title("Volatility vs. learner fraction")
    eb(axes[1, 0], "oos_r2", "tab:green", "out-of-sample $R^2$")
    axes[1, 0].set_title("Out-of-sample predictability vs. learner fraction")
    eb(axes[1, 1], "frac_profitable_learners", "tab:purple", "fraction of learners profitable",
       mask_zero_learners=True)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("% profitable (learners) vs. learner fraction")

    fig.suptitle(f"Experiment A (v2 model): learner-fraction sweep "
                 f"(N={N}, alpha={ALPHA0}, beta={BETA0}, n_seeds={N_SEEDS_A})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")



# Experiment B: sensitivity to alpha (low / medium / high)


def run_experiment_B():
    print("\n=== Experiment B: alpha sensitivity, baseline vs. full learning (v2 model) ===")
    alpha_values = [0.2, 1.0, 3.0]  # low / medium / high
    grid = {}
    for a in alpha_values:
        base = run_stats_v2(n_seeds=N_SEEDS_B, N=N, T=T, alpha=a, beta=BETA0,
                             f=0.0, tau=TAU0, eta0=ETA0, eta_decay=ETA_DECAY0, gamma=GAMMA0, c=C0)
        learn = run_stats_v2(n_seeds=N_SEEDS_B, N=N, T=T, alpha=a, beta=BETA0,
                              f=1.0, tau=TAU0, eta0=ETA0, eta_decay=ETA_DECAY0, gamma=GAMMA0, c=C0)
        t_lag1, p_lag1 = welch_t_test(
            [rec["lag1_corr"] for rec in base["_raw"]],
            [rec["lag1_corr"] for rec in learn["_raw"]],
        )
        t_std, p_std = welch_t_test(
            [rec["return_std"] for rec in base["_raw"]],
            [rec["return_std"] for rec in learn["_raw"]],
        )
        grid[a] = {
            "base_lag1": base["lag1_corr_mean"], "learn_lag1": learn["lag1_corr_mean"],
            "delta_lag1": learn["lag1_corr_mean"] - base["lag1_corr_mean"],
            "t_lag1": t_lag1, "p_lag1": p_lag1,
            "base_std": base["return_std_mean"], "learn_std": learn["return_std_mean"],
            "vol_ratio": learn["return_std_mean"] / max(base["return_std_mean"], 1e-9),
            "t_std": t_std, "p_std": p_std,
        }
        g = grid[a]
        print(f"  alpha={a:>4}: lag1 base={g['base_lag1']:+.4f} -> learn={g['learn_lag1']:+.4f} "
              f"(delta={g['delta_lag1']:+.4f}, p={g['p_lag1']:.3f})   "
              f"std(r) base={g['base_std']:.3f} -> learn={g['learn_std']:.3f} "
              f"(ratio={g['vol_ratio']:.2f}x, p={g['p_std']:.3f})")

    OUT["expB_alpha_values"] = alpha_values
    OUT["expB_grid"] = grid
    return alpha_values, grid


def plot_experiment_B(alpha_values, grid, save_path="fig2_alpha_sensitivity.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.array(alpha_values, dtype=float)

    delta_lag1 = np.array([grid[a]["delta_lag1"] for a in alpha_values])
    axes[0].axhline(0, color="grey", lw=0.6, ls=":")
    axes[0].plot(x, delta_lag1, "o-", color="tab:blue")
    for a in alpha_values:
        star = "*" if grid[a]["p_lag1"] < 0.05 else ""
        axes[0].annotate(f"{grid[a]['delta_lag1']:+.3f}{star}", (a, grid[a]["delta_lag1"]),
                          textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axes[0].set_xlabel("market impact $\\alpha$")
    axes[0].set_ylabel("$\\Delta$ lag-1 corr (learn $-$ baseline)")
    axes[0].set_title("Predictability shift vs. $\\alpha$")

    vol_ratio = np.array([grid[a]["vol_ratio"] for a in alpha_values])
    axes[1].axhline(1, color="grey", lw=0.6, ls=":")
    axes[1].plot(x, vol_ratio, "o-", color="tab:orange")
    for a in alpha_values:
        star = "*" if grid[a]["p_std"] < 0.05 else ""
        axes[1].annotate(f"{grid[a]['vol_ratio']:.2f}x{star}", (a, grid[a]["vol_ratio"]),
                          textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axes[1].set_xlabel("market impact $\\alpha$")
    axes[1].set_ylabel("volatility ratio (learn / baseline)")
    axes[1].set_title("Volatility amplification vs. $\\alpha$")

    fig.suptitle(f"Experiment B (v2 model): full learning (f=1.0) vs. baseline (f=0), "
                 f"beta={BETA0}, n_seeds={N_SEEDS_B}  (*=p<0.05)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")



# Experiment C: learning stability -- v1 (fixed eta) vs v2 (annealed eta)


def run_experiment_C(seed=1, n_check_seeds=5):
    print("\n=== Experiment C: learning stability, v1 (fixed eta) vs v2 (annealed eta) ===")
    v1_out = simulate_learning_market_v1_diag(
        N=N, T=T, alpha=ALPHA0, beta=BETA0, f=1.0, tau=TAU0, eta=ETA0, gamma=GAMMA0, c=C0, seed=seed
    )
    v2_out = simulate_learning_market_v2(
        N=N, T=T, alpha=ALPHA0, beta=BETA0, f=1.0, tau=TAU0, eta0=ETA0, eta_decay=ETA_DECAY0,
        gamma=GAMMA0, c=C0, seed=seed
    )

    def summarize(out, label):
        ent = out["entropy_t"]; ent_v = ent[~np.isnan(ent)]
        qs = out["qstd_t"]; qs_v = qs[~np.isnan(qs)]
        tail = max(1, len(ent_v) // 5)
        print(f"  [{label}] entropy: mean={ent_v.mean():.4f}  late-mean={ent_v[-tail:].mean():.4f}  "
              f"min={ent_v.min():.4f}  final-Qstd={qs_v[-1]:.4f}")
        return {
            "entropy_mean": float(ent_v.mean()), "entropy_late_mean": float(ent_v[-tail:].mean()),
            "entropy_min": float(ent_v.min()), "final_Qstd": float(qs_v[-1]),
        }

    v1_summary = summarize(v1_out, "v1 fixed-eta")
    v2_summary = summarize(v2_out, "v2 annealed-eta")

    # check consistency across a handful of extra seeds (not just the one representative seed)
    print(f"  cross-seed check (n={n_check_seeds}), mean of entropy_late and final Qstd:")
    v1_late, v1_qf, v2_late, v2_qf = [], [], [], []
    for s in range(n_check_seeds):
        o1 = simulate_learning_market_v1_diag(N=N, T=T, alpha=ALPHA0, beta=BETA0, f=1.0,
                                               tau=TAU0, eta=ETA0, gamma=GAMMA0, c=C0, seed=s)
        o2 = simulate_learning_market_v2(N=N, T=T, alpha=ALPHA0, beta=BETA0, f=1.0,
                                          tau=TAU0, eta0=ETA0, eta_decay=ETA_DECAY0,
                                          gamma=GAMMA0, c=C0, seed=s)
        e1 = o1["entropy_t"][~np.isnan(o1["entropy_t"])]; tail1 = max(1, len(e1)//5)
        e2 = o2["entropy_t"][~np.isnan(o2["entropy_t"])]; tail2 = max(1, len(e2)//5)
        q1 = o1["qstd_t"][~np.isnan(o1["qstd_t"])]
        q2 = o2["qstd_t"][~np.isnan(o2["qstd_t"])]
        v1_late.append(e1[-tail1:].mean()); v2_late.append(e2[-tail2:].mean())
        v1_qf.append(q1[-1]); v2_qf.append(q2[-1])
    print(f"    v1 late-entropy: {np.mean(v1_late):.4f}+/-{np.std(v1_late):.4f}   "
          f"v2 late-entropy: {np.mean(v2_late):.4f}+/-{np.std(v2_late):.4f}")
    print(f"    v1 final Qstd:   {np.mean(v1_qf):.4f}+/-{np.std(v1_qf):.4f}   "
          f"v2 final Qstd:   {np.mean(v2_qf):.4f}+/-{np.std(v2_qf):.4f}")

    OUT["expC_v1_summary"] = v1_summary
    OUT["expC_v2_summary"] = v2_summary
    OUT["expC_cross_seed"] = {
        "v1_late_mean": float(np.mean(v1_late)), "v1_late_sd": float(np.std(v1_late)),
        "v2_late_mean": float(np.mean(v2_late)), "v2_late_sd": float(np.std(v2_late)),
        "v1_qf_mean": float(np.mean(v1_qf)), "v1_qf_sd": float(np.std(v1_qf)),
        "v2_qf_mean": float(np.mean(v2_qf)), "v2_qf_sd": float(np.std(v2_qf)),
    }
    return v1_out, v2_out


def plot_experiment_C(v1_out, v2_out, save_path="fig3_learning_stability.png"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")

    axes[0, 0].plot(v1_out["entropy_t"], lw=0.7, color="tab:green")
    axes[0, 0].axhline(np.log(3), color="grey", lw=0.6, ls=":", label="max entropy ln(3)")
    axes[0, 0].set_ylabel("policy entropy")
    axes[0, 0].set_title("v1: fixed $\\eta$, 6-state, holding cost")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[0, 0].set_ylim(0, 1.25)

    axes[0, 1].plot(v2_out["entropy_t"], lw=0.7, color="tab:green")
    axes[0, 1].axhline(np.log(3), color="grey", lw=0.6, ls=":", label="max entropy ln(3)")
    axes[0, 1].set_title("v2: annealed $\\eta$, 27-state, turnover cost")
    axes[0, 1].legend(fontsize=8, loc="lower left")
    axes[0, 1].set_ylim(0, 1.25)

    axes[1, 0].plot(v1_out["qstd_t"], lw=0.7, color="tab:purple")
    axes[1, 0].set_ylabel("std(Q)")
    axes[1, 0].set_xlabel("t")

    axes[1, 1].plot(v2_out["qstd_t"], lw=0.7, color="tab:purple")
    axes[1, 1].set_xlabel("t")

    fig.suptitle(f"Experiment C: learning stability, representative seed "
                 f"(N={N}, alpha={ALPHA0}, beta={BETA0}, f=1.0)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"saved {save_path}")



# Final table (Section 3)


def build_final_table(f_values, results):
    rows = []
    for f in f_values:
        s = results[f]
        prof = s["frac_profitable_learners_mean"]
        ent = s["entropy_late_mean"]
        rows.append({
            "f": f,
            "std_r": s["return_std_mean"],
            "std_r_sd": s["return_std_sd"],
            "lag1": s["lag1_corr_mean"],
            "lag1_sd": s["lag1_corr_sd"],
            "oos_r2": s["oos_r2_mean"],
            "oos_r2_sd": s["oos_r2_sd"],
            "pct_profitable": None if np.isnan(prof) else prof * 100,
            "entropy": None if np.isnan(ent) else ent,
        })
    print("\n=== Final table (Section 3) ===")
    print(f"{'f':>5} {'std(r)':>10} {'lag1':>10} {'oosR2':>10} {'%profit':>10} {'entropy':>10}")
    for row in rows:
        prof_s = f"{row['pct_profitable']:.1f}" if row["pct_profitable"] is not None else "n/a"
        ent_s = f"{row['entropy']:.4f}" if row["entropy"] is not None else "n/a"
        print(f"{row['f']:>5} {row['std_r']:>10.4f} {row['lag1']:>10.4f} {row['oos_r2']:>10.4f} "
              f"{prof_s:>10} {ent_s:>10}")
    OUT["final_table"] = rows
    return rows



# main


if __name__ == "__main__":
    t0 = time.time()
    f_values, resultsA, sigA = run_experiment_A()
    alpha_values, gridB = run_experiment_B()
    v1_out, v2_out = run_experiment_C()
    table_rows = build_final_table(f_values, resultsA)
    print(f"\nTotal numeric-experiment runtime: {time.time()-t0:.1f}s")

    t1 = time.time()
    plot_experiment_A(f_values, resultsA)
    plot_experiment_B(alpha_values, gridB)
    plot_experiment_C(v1_out, v2_out)
    print(f"Plotting runtime: {time.time()-t1:.1f}s")

    # dump a JSON summary (excluding raw per-seed records / heavy arrays) for reuse
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k != "_raw" and k != "_kwargs"}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open("results_summary.json", "w") as fh:
        json.dump(_clean(OUT), fh, indent=2)
    print("saved results_summary.json")
