#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tool Window Usage Analysis (final, English)
Compare durations for tool window sessions opened manually vs automatically.

Example:
    python toolwindow_analysis.py --data data/toolwindow_data.csv --double-open-policy close_previous --per-user --survival
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

# survival (optional)
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except Exception:
    HAS_LIFELINES = False

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 5)


# ---------------------- Utilities ----------------------
def cliffs_delta(x, y):
    """Cliff's delta: nonparametric effect size robust to non-normality."""
    x = np.asarray(x)
    y = np.asarray(y)
    gt = 0
    lt = 0
    for a in x:
        gt += np.sum(a > y)
        lt += np.sum(a < y)
    n = len(x) * len(y)
    return (gt - lt) / n if n else np.nan


def bootstrap_ci(data, stat_func=np.median, n_boot=5000, alpha=0.05, random_state=42):
    """Bootstrap two-sided (1-alpha) CI for a given statistic (median by default)."""
    rng = np.random.default_rng(random_state)
    data = np.asarray(data)
    boots = []
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boots.append(stat_func(sample))
    low = np.percentile(boots, 100 * (alpha / 2))
    high = np.percentile(boots, 100 * (1 - alpha / 2))
    return (low, high)


# ---------------------- Matching logic ----------------------
@dataclass
class Episode:
    user_id: str
    open_timestamp: int
    close_timestamp: Optional[int]
    duration_ms: int
    open_type: Optional[str]
    censored: bool


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and values across typical variants."""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    if "event" in df.columns and "event_id" not in df.columns:
        df.rename(columns={"event": "event_id"}, inplace=True)

    for col in ["event_id", "open_type", "user_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # unifying typical aliases
    df["event_id"] = df["event_id"].replace({"opened": "open", "closed": "close"})
    df["open_type"] = df["open_type"].replace(
        {"automatic": "auto", "manually": "manual", "": np.nan, "nan": np.nan}
    )

    required = {"user_id", "timestamp", "event_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # enforce integer timestamps in ms
    df["timestamp"] = df["timestamp"].astype(np.int64)
    return df


def match_episodes(df: pd.DataFrame, double_open_policy: str = "close_previous") -> pd.DataFrame:
    """
    Build open→close episodes per user_id.
    Supports policies for consecutive 'open'. Keeps right-censored episodes (open without a later close).
    """
    assert double_open_policy in {"close_previous", "drop_previous", "keep_first"}

    episodes: List[Episode] = []

    df_sorted = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    end_of_log_by_user = df_sorted.groupby("user_id")["timestamp"].max()

    for user_id, user_events in df_sorted.groupby("user_id", sort=False):
        user_events = user_events.reset_index(drop=True)
        pending_open: Optional[Dict] = None

        for _, row in user_events.iterrows():
            ev = row["event_id"]
            ts = int(row["timestamp"])

            if ev == "open":
                # consecutive open
                if pending_open is not None:
                    if double_open_policy == "close_previous":
                        dur = ts - pending_open["open_timestamp"]
                        if dur >= 0:
                            episodes.append(
                                Episode(
                                    user_id=user_id,
                                    open_timestamp=pending_open["open_timestamp"],
                                    close_timestamp=ts,
                                    duration_ms=dur,
                                    open_type=pending_open["open_type"],
                                    censored=False,
                                )
                            )
                        # start a new pending open
                        pending_open = {
                            "open_timestamp": ts,
                            "open_type": row.get("open_type", None),
                        }
                    elif double_open_policy == "drop_previous":
                        # drop the old pending and replace with the new open
                        pending_open = {
                            "open_timestamp": ts,
                            "open_type": row.get("open_type", None),
                        }
                    elif double_open_policy == "keep_first":
                        # ignore this new open until close arrives
                        pass
                else:
                    pending_open = {
                        "open_timestamp": ts,
                        "open_type": row.get("open_type", None),
                    }

            elif ev == "close":
                if pending_open is not None:
                    dur = ts - pending_open["open_timestamp"]
                    if dur >= 0:
                        episodes.append(
                            Episode(
                                user_id=user_id,
                                open_timestamp=pending_open["open_timestamp"],
                                close_timestamp=ts,
                                duration_ms=dur,
                                open_type=pending_open["open_type"],
                                censored=False,
                            )
                        )
                    pending_open = None
                else:
                    # close without a matching open → skip
                    continue

        # right-censor at user's end-of-log
        if pending_open is not None:
            end_ts = int(end_of_log_by_user.loc[user_id])
            dur = end_ts - pending_open["open_timestamp"]
            if dur >= 0:
                episodes.append(
                    Episode(
                        user_id=user_id,
                        open_timestamp=pending_open["open_timestamp"],
                        close_timestamp=None,
                        duration_ms=dur,
                        open_type=pending_open["open_type"],
                        censored=True,
                    )
                )

    epi_df = pd.DataFrame([e.__dict__ for e in episodes])
    if not epi_df.empty:
        epi_df["duration_seconds"] = epi_df["duration_ms"] / 1000.0
        epi_df["duration_minutes"] = epi_df["duration_ms"] / (1000.0 * 60.0)
    return epi_df


def iqr_filter_by_group(episodes: pd.DataFrame, group_col="open_type",
                        min_duration_ms=100, iqr_multiplier=3.0) -> pd.DataFrame:
    """Remove too-short episodes and extreme outliers using group-wise IQR bounds."""
    df = episodes.copy()
    df = df[df["duration_ms"] >= min_duration_ms]

    def filter_group(g):
        Q1 = g["duration_seconds"].quantile(0.25)
        Q3 = g["duration_seconds"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + iqr_multiplier * IQR
        return g[g["duration_seconds"] <= upper]

    filtered = df.groupby(group_col, group_keys=False).apply(filter_group)
    return filtered


def compute_group_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute summary stats and bootstrap CIs for mean and median per open_type."""
    out = {}
    for t in ["manual", "auto"]:
        s = df.loc[df["open_type"] == t, "duration_seconds"]
        if s.empty:
            continue
        out[t] = {
            "count": float(s.count()),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "std": float(s.std()),
            "min": float(s.min()),
            "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)),
            "max": float(s.max()),
        }
        out[t]["median_ci_low"], out[t]["median_ci_high"] = bootstrap_ci(s, np.median)
        out[t]["mean_ci_low"], out[t]["mean_ci_high"] = bootstrap_ci(s, np.mean)
    return out


def per_user_paired(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, float]]]:
    """Per-user medians and paired Wilcoxon test for users who have both manual and auto."""
    per_user = (df.groupby(["user_id", "open_type"])["duration_seconds"]
                  .median()
                  .unstack())
    per_user = per_user.rename_axis(None, axis=1)
    paired = per_user.dropna(subset=["manual", "auto"], how="any")
    if paired.empty:
        return per_user, None
    try:
        stat, p = stats.wilcoxon(paired["manual"], paired["auto"], zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p = np.nan, np.nan
    res = {
        "n_users": int(len(paired)),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "mean_diff": float(paired["manual"].mean() - paired["auto"].mean())
    }
    return per_user, res


def create_figures(df: pd.DataFrame, outdir="figures", survival=False):
    """Generate plots and return their file paths."""
    os.makedirs(outdir, exist_ok=True)

    # Histogram
    plt.figure()
    for label in ["manual", "auto"]:
        s = df.loc[df["open_type"] == label, "duration_seconds"]
        if not s.empty:
            plt.hist(s, bins=40, alpha=0.6, density=True, label=label)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Density")
    plt.title("Distribution of Tool Window Open Duration")
    plt.legend()
    hist_path = os.path.join(outdir, "hist_duration.png")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # Boxplot
    plt.figure()
    data = [df.loc[df["open_type"] == "manual", "duration_seconds"],
            df.loc[df["open_type"] == "auto", "duration_seconds"]]
    labels = ["manual", "auto"]
    plt.boxplot(data, labels=labels)
    plt.ylabel("Duration (seconds)")
    plt.title("Boxplot: manual vs auto")
    box_path = os.path.join(outdir, "boxplot.png")
    plt.tight_layout()
    plt.savefig(box_path, dpi=200)
    plt.close()

    # ECDF
    plt.figure()
    for label in ["manual", "auto"]:
        s = np.sort(df.loc[df["open_type"] == label, "duration_seconds"].dropna().values)
        if s.size:
            y = np.arange(1, s.size + 1) / s.size
            plt.plot(s, y, label=label)
    plt.xlabel("Duration (seconds)")
    plt.ylabel("ECDF")
    plt.title("ECDF: manual vs auto")
    plt.legend()
    ecdf_path = os.path.join(outdir, "ecdf.png")
    plt.tight_layout()
    plt.savefig(ecdf_path, dpi=200)
    plt.close()

    km_path = None
    if survival and HAS_LIFELINES:
        # Kaplan–Meier
        plt.figure()
        kmf = KaplanMeierFitter()
        for label in ["manual", "auto"]:
            g = df[df["open_type"] == label]
            if not g.empty:
                durations = g["duration_seconds"].values
                event_observed = (~g["censored"]).astype(int).values
                kmf.fit(durations, event_observed=event_observed, label=label)
                kmf.plot(ci_show=True)
        plt.xlabel("Seconds")
        plt.ylabel("Survival probability (open window)")
        plt.title("Kaplan–Meier: manual vs auto")
        km_path = os.path.join(outdir, "km_curves.png")
        plt.tight_layout()
        plt.savefig(km_path, dpi=200)
        plt.close()

    return {"hist": hist_path, "box": box_path, "ecdf": ecdf_path, "km": km_path}


def build_pdf_report(pdf_path: str,
                     group_stats: Dict[str, Dict[str, float]],
                     mw_test: Optional[Dict[str, float]],
                     cliffs: Optional[float],
                     per_user_info: Optional[Dict[str, float]],
                     fig_paths: Dict[str, Optional[str]],
                     assumptions: List[str],
                     data_info: Dict[str, str],
                     survival_info: Optional[Dict[str, float]]):
    """Assemble a concise PDF report with methods, stats, tests, and visuals."""
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    def write_wrapped(text, x, y, max_width=540, leading=14):
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.platypus import Paragraph, Frame
        from reportlab.lib.enums import TA_LEFT
        style = ParagraphStyle(name='Normal', fontSize=10, leading=leading, alignment=TA_LEFT)
        frame = Frame(x, y-200, max_width, 200, showBoundary=0)
        p = Paragraph(text, style)
        frame.addFromList([p], c)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(2*cm, height-2*cm, "Tool Window Usage Analysis: manual vs auto")

    c.setFont("Helvetica", 10)
    y = height - 2.7*cm
    write_wrapped(f"<b>Dataset:</b> {data_info.get('name','')} | "
                  f"events: {data_info.get('n_events','?')} | "
                  f"users: {data_info.get('n_users','?')} | "
                  f"episodes: {data_info.get('n_episodes','?')} | "
                  f"censored episodes kept: {data_info.get('n_censored','?')}", 2*cm, y)

    y -= 2.5*cm
    write_wrapped("<b>Assumptions & Cleaning:</b><br/>" + "<br/>".join(f"• {a}" for a in assumptions), 2*cm, y)

    # Summary stats
    y -= 2.6*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Summary statistics")
    c.setFont("Helvetica", 10)
    y -= 0.5*cm

    def draw_stats(block_title, stats_dict, y_pos):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(2*cm, y_pos, block_title)
        y_pos -= 0.4*cm
        c.setFont("Helvetica", 9)
        for k, v in stats_dict.items():
            line = (f"{k}: "
                    f"n={int(v['count'])}, "
                    f"mean={v['mean']:.2f}s (95% CI [{v['mean_ci_low']:.2f},{v['mean_ci_high']:.2f}]), "
                    f"median={v['median']:.2f}s (95% CI [{v['median_ci_low']:.2f},{v['median_ci_high']:.2f}]), "
                    f"IQR=[{v['q25']:.2f},{v['q75']:.2f}]")
            c.drawString(2.2*cm, y_pos, f"- {k}: {line}")
            y_pos -= 0.4*cm
        return y_pos

    y = draw_stats("By open_type", group_stats, y)

    # Tests
    y -= 0.2*cm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, y, "Statistical tests")
    y -= 0.5*cm
    c.setFont("Helvetica", 10)

    if mw_test:
        c.drawString(2.2*cm, y, f"Mann–Whitney U: U={mw_test['U']:.2f}, p={mw_test['p_value']:.6g}")
        y -= 0.4*cm
        c.drawString(2.2*cm, y, f"Cohen's d={mw_test['cohens_d']:.3f}, mean_diff={mw_test['mean_diff']:.2f}s")
        y -= 0.4*cm
    if cliffs is not None and not math.isnan(cliffs):
        magnitude = ('negligible' if abs(cliffs)<0.147 else 'small' if abs(cliffs)<0.33 else 'medium' if abs(cliffs)<0.474 else 'large')
        c.drawString(2.2*cm, y, f"Cliff's delta={cliffs:.3f} (magnitude: {magnitude})")
        y -= 0.4*cm
    if per_user_info:
        c.drawString(2.2*cm, y, f"Per-user Wilcoxon: n_users={per_user_info['n_users']}, W={per_user_info['wilcoxon_stat']:.2f}, p={per_user_info['wilcoxon_p']:.6g}")
        y -= 0.4*cm
    if survival_info:
        c.drawString(2.2*cm, y, f"Log-rank test: chi2={survival_info['chi2']:.2f}, p={survival_info['p']:.6g}")
        y -= 0.4*cm

    # Figures
    for title, path in [("Histogram", fig_paths.get("hist")),
                        ("ECDF", fig_paths.get("ecdf")),
                        ("Boxplot", fig_paths.get("box")),
                        ("Kaplan–Meier", fig_paths.get("km"))]:
        if path and os.path.exists(path):
            if y < 10*cm:
                c.showPage()
                y = height - 2*cm
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, y, title)
            y -= 0.5*cm
            c.drawImage(path, 2*cm, y-7*cm, width=16*cm, height=7*cm, preserveAspectRatio=True, anchor='nw')
            y -= 7.5*cm

    c.showPage()
    c.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--double-open-policy", type=str, default="close_previous",
                        choices=["close_previous", "drop_previous", "keep_first"])
    parser.add_argument("--min-duration-ms", type=int, default=100)
    parser.add_argument("--iqr-multiplier", type=float, default=3.0)
    parser.add_argument("--survival", action="store_true", help="Build KM curves and log-rank test")
    parser.add_argument("--per-user", action="store_true", help="Per-user medians and paired Wilcoxon")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Load & normalize
    df = pd.read_csv(args.data)
    df = normalize_df(df)

    # Info
    data_info = {
        "name": os.path.basename(args.data),
        "n_events": str(len(df)),
        "n_users": str(df["user_id"].nunique()),
    }

    # Match
    episodes = match_episodes(df, double_open_policy=args.double_open_policy)

    # drop missing open_type
    episodes = episodes[episodes["open_type"].notna()]
    n_censored = int(episodes["censored"].sum()) if not episodes.empty else 0
    data_info.update({"n_episodes": str(len(episodes)), "n_censored": str(n_censored)})

    # Clean
    episodes_clean = iqr_filter_by_group(
        episodes, group_col="open_type",
        min_duration_ms=args.min_duration_ms,
        iqr_multiplier=args.iqr_multiplier
    )

    # Save episodes
    episodes_clean.to_csv("episodes_clean.csv", index=False)

    # Stats
    group_stats = compute_group_stats(episodes_clean)

    # Tests
    manual = episodes_clean.loc[episodes_clean["open_type"] == "manual", "duration_seconds"]
    auto = episodes_clean.loc[episodes_clean["open_type"] == "auto", "duration_seconds"]

    mw_test = None
    cliffs = None
    if not manual.empty and not auto.empty:
        U, p = stats.mannwhitneyu(manual, auto, alternative="two-sided")
        mean_diff = float(manual.mean() - auto.mean())
        pooled_std = math.sqrt(
            ((manual.count()-1)*manual.std(ddof=1)**2 + (auto.count()-1)*auto.std(ddof=1)**2) /
            (manual.count()+auto.count()-2)
        ) if (manual.count()+auto.count()-2) > 0 else np.nan
        cohens_d = mean_diff / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan
        mw_test = {"U": float(U), "p_value": float(p), "cohens_d": float(cohens_d), "mean_diff": mean_diff}
        cliffs = float(cliffs_delta(manual.values, auto.values))

    # Per-user
    per_user_stats, per_user_info = (None, None)
    if args.per_user:
        per_user_stats, per_user_info = per_user_paired(episodes_clean)
        if per_user_stats is not None:
            per_user_stats.to_csv("per_user_stats.csv", index=True)

    # Survival
    survival_info = None
    fig_paths = create_figures(episodes_clean, outdir="figures", survival=args.survival and HAS_LIFELINES)
    if args.survival and HAS_LIFELINES and not manual.empty and not auto.empty:
        g1 = episodes_clean[episodes_clean["open_type"] == "manual"]
        g2 = episodes_clean[episodes_clean["open_type"] == "auto"]
        r = logrank_test(g1["duration_seconds"], g2["duration_seconds"],
                         event_observed_A=(~g1["censored"]).astype(int),
                         event_observed_B=(~g2["censored"]).astype(int))
        survival_info = {"chi2": float(r.test_statistic), "p": float(r.p_value)}

    # Save summary JSON
    with open("summary_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "group_stats": group_stats,
            "mann_whitney": mw_test,
            "cliffs_delta": cliffs,
            "per_user": per_user_info,
            "survival": survival_info
        }, f, ensure_ascii=False, indent=2)

    # Assumptions
    assumptions = [
        "Standardized event_id to {open, close} and open_type to {manual, auto}.",
        f"Consecutive opens policy: {args.double_open_policy}.",
        "Right-censored episodes retained for survival analysis.",
        f"Cleaning: min_duration_ms={args.min_duration_ms}, IQR multiplier={args.iqr_multiplier} (group-wise).",
        "Group comparison: Mann–Whitney U; Cliff’s delta; bootstrap 95% CI for mean/median.",
        "Per-user analysis: per-user medians + paired Wilcoxon (if enabled).",
    ]

    # Build PDF
    build_pdf_report(
        pdf_path=os.path.join("reports", "report.pdf"),
        group_stats=group_stats,
        mw_test=mw_test,
        cliffs=cliffs,
        per_user_info=per_user_info,
        fig_paths=fig_paths,
        assumptions=assumptions,
        data_info=data_info,
        survival_info=survival_info
    )

    print("Done. Report: reports/report.pdf, plots: figures/, summary: summary_stats.json")


if __name__ == "__main__":
    main()
