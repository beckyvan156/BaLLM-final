# -*- coding: utf-8 -*-
"""
Standard MCMC convergence diagnostics for PyMC traces.

Reports R-hat, bulk/tail ESS, divergences, max tree depth, BFMI, and mean
acceptance rate; saves trace, rank, and energy plots. Use `nuts_diagnostics`
after each NUTS `pm.sample()` call; use `metropolis_diagnostics` for RWMH.
"""

import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


def _ensure_dir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def nuts_diagnostics(
    trace,
    model_name,
    var_names=None,
    save_dir=None,
    max_treedepth=10,
    rhat_threshold=1.01,
    ess_threshold=400,
    show_plots=True,
):
    """
    Print and save standard NUTS convergence diagnostics.

    Parameters
    ----------
    trace : arviz.InferenceData
        Output of pm.sample(..., return_inferencedata=True).
    model_name : str
        Identifier used in printed headers and plot filenames (e.g. 'nuts_unconstrained').
    var_names : list[str] or None
        Variables to include in summary/plots. None -> all.
    save_dir : str or None
        Directory for PNG plots. None -> no files saved.
    max_treedepth : int
        Max tree depth used during sampling (PyMC default is 10). Used only to flag hits.
    rhat_threshold, ess_threshold : float
        Thresholds for flagging problematic variables.
    show_plots : bool
        Whether to call plt.show().

    Returns
    -------
    dict
        {'summary': DataFrame, 'divergences': int, 'max_tree_depth_hits': int,
         'mean_acceptance': float, 'bfmi': np.ndarray, 'rhat_bad': DataFrame,
         'ess_bad': DataFrame}
    """
    _ensure_dir(save_dir)
    print("\n" + "=" * 72)
    print(f"NUTS diagnostics — {model_name}")
    print("=" * 72)

    # 1. R-hat and ESS summary
    summary = az.summary(
        trace,
        var_names=var_names,
        stat_focus="mean",
        round_to=4,
    )
    print("\n[Summary: mean, sd, R-hat, ESS]")
    print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())

    # Flag problematic variables
    rhat_bad = summary[summary["r_hat"] > rhat_threshold]
    ess_bad = summary[(summary["ess_bulk"] < ess_threshold) | (summary["ess_tail"] < ess_threshold)]
    print(f"\nVariables with R-hat > {rhat_threshold}: {len(rhat_bad)}")
    if len(rhat_bad):
        print(rhat_bad[["r_hat"]].to_string())
    print(f"Variables with bulk or tail ESS < {ess_threshold}: {len(ess_bad)}")
    if len(ess_bad):
        print(ess_bad[["ess_bulk", "ess_tail"]].to_string())

    # 2. NUTS-specific sample stats
    stats = trace.sample_stats
    divergences = int(stats["diverging"].sum().item()) if "diverging" in stats else 0
    tree_depth = stats["tree_depth"].values if "tree_depth" in stats else None
    max_td_hits = int((tree_depth == max_treedepth).sum()) if tree_depth is not None else 0
    mean_acc = float(stats["acceptance_rate"].mean().item()) if "acceptance_rate" in stats else float("nan")
    bfmi = az.bfmi(trace)

    print(f"\n[NUTS sampler diagnostics]")
    print(f"  Divergent transitions        : {divergences}")
    print(f"  Post-warmup draws at max_treedepth={max_treedepth} : {max_td_hits}")
    print(f"  Mean acceptance rate         : {mean_acc:.3f}")
    print(f"  BFMI per chain               : {np.round(bfmi, 3).tolist()}")
    if np.any(np.asarray(bfmi) < 0.3):
        print("  WARNING: BFMI < 0.3 in at least one chain (posterior may be difficult).")
    if divergences > 0:
        print("  WARNING: Divergences detected. Posterior inference may be biased; "
              "try higher target_accept or reparameterize.")
    if max_td_hits > 0:
        print("  WARNING: Chains hit max tree depth — consider raising max_treedepth.")

    # 3. Plots
    def _save(name):
        if save_dir:
            out = os.path.join(save_dir, f"{model_name}_{name}.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            print(f"  saved: {out}")

    # Trace plot
    az.plot_trace(trace, var_names=var_names, compact=True)
    plt.suptitle(f"Trace — {model_name}", y=1.02)
    plt.tight_layout()
    _save("trace")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Rank plot (better than trace for detecting mixing problems across chains)
    az.plot_rank(trace, var_names=var_names)
    plt.suptitle(f"Rank — {model_name}", y=1.02)
    plt.tight_layout()
    _save("rank")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Energy plot (NUTS pathology detector)
    az.plot_energy(trace)
    plt.suptitle(f"Energy — {model_name}", y=1.02)
    plt.tight_layout()
    _save("energy")
    if show_plots:
        plt.show()
    else:
        plt.close()

    # Save summary CSV too
    if save_dir:
        summary.to_csv(os.path.join(save_dir, f"{model_name}_summary.csv"))

    return {
        "summary": summary,
        "divergences": divergences,
        "max_tree_depth_hits": max_td_hits,
        "mean_acceptance": mean_acc,
        "bfmi": np.asarray(bfmi),
        "rhat_bad": rhat_bad,
        "ess_bad": ess_bad,
    }


def metropolis_diagnostics(
    trace,
    model_name,
    var_names=None,
    save_dir=None,
    rhat_threshold=1.01,
    ess_threshold=400,
    show_plots=True,
):
    """
    Diagnostics for RWMH (Metropolis). Reports R-hat, ESS, acceptance rate,
    autocorrelation plots, and trace plots.
    """
    _ensure_dir(save_dir)
    print("\n" + "=" * 72)
    print(f"RWMH diagnostics — {model_name}")
    print("=" * 72)

    summary = az.summary(trace, var_names=var_names, round_to=4)
    print("\n[Summary: mean, sd, R-hat, ESS]")
    print(summary[["mean", "sd", "r_hat", "ess_bulk", "ess_tail"]].to_string())

    rhat_bad = summary[summary["r_hat"] > rhat_threshold]
    ess_bad = summary[(summary["ess_bulk"] < ess_threshold) | (summary["ess_tail"] < ess_threshold)]
    print(f"\nVariables with R-hat > {rhat_threshold}: {len(rhat_bad)}")
    print(f"Variables with bulk or tail ESS < {ess_threshold}: {len(ess_bad)}")

    # Metropolis acceptance is stored under sample_stats if present
    stats = trace.sample_stats
    acc = None
    for candidate in ["accepted", "accept", "mean_tree_accept"]:
        if candidate in stats:
            acc = float(np.mean(stats[candidate].values))
            break
    if acc is not None:
        print(f"\nMean acceptance rate         : {acc:.3f}")
        if acc < 0.15 or acc > 0.5:
            print("  NOTE: RWMH acceptance outside 0.15-0.5 range — proposal scale may be mis-tuned.")

    def _save(name):
        if save_dir:
            out = os.path.join(save_dir, f"{model_name}_{name}.png")
            plt.savefig(out, dpi=200, bbox_inches="tight")
            print(f"  saved: {out}")

    az.plot_trace(trace, var_names=var_names, compact=True)
    plt.suptitle(f"Trace — {model_name}", y=1.02)
    plt.tight_layout()
    _save("trace")
    if show_plots: plt.show()
    else: plt.close()

    az.plot_autocorr(trace, var_names=var_names, max_lag=100, combined=True)
    plt.suptitle(f"Autocorrelation — {model_name}", y=1.02)
    plt.tight_layout()
    _save("autocorr")
    if show_plots: plt.show()
    else: plt.close()

    if save_dir:
        summary.to_csv(os.path.join(save_dir, f"{model_name}_summary.csv"))

    return {
        "summary": summary,
        "mean_acceptance": acc,
        "rhat_bad": rhat_bad,
        "ess_bad": ess_bad,
    }
