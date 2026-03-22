#!/usr/bin/env python3
"""Part 2 - Exercise 2 scaling analysis for OpenMP matrix multiplication."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_THREADS = 4
INPUT_FILES = [
    "perf_results_onmultline_ex2_pragma_omp_parallel_for.txt",
    "perf_results_onmultline_ex2_pragma_omp_parallel_for_collapse(2).txt",
    "perf_results_onmultline_ex2_pragma_omp_for_simd.txt",
]

STRATEGY_LABELS = {
    "perf_results_onmultline_ex2_pragma_omp_parallel_for.txt": "omp parallel for",
    "perf_results_onmultline_ex2_pragma_omp_parallel_for_collapse(2).txt": "omp parallel for collapse(2)",
    "perf_results_onmultline_ex2_pragma_omp_for_simd.txt": "omp for simd",
}

METRIC_SUFFIX_MAP = {
    "cycles": "cycles",
    "instructions": "instructions",
    "cache-references": "cache_references",
    "cache-misses": "cache_misses",
    "L1-dcache-loads": "l1_dcache_loads",
    "L1-dcache-load-misses": "l1_dcache_load_misses",
    "LLC-loads": "llc_loads",
    "LLC-load-misses": "llc_load_misses",
}

HEADER_RE = re.compile(
    r"Method:\s*(?P<method>\w+)\s*\|\s*Size:\s*(?P<size>\d+)x\d+\s*\|\s*Threads:\s*(?P<threads>\d+)",
    re.IGNORECASE,
)
PARALLEL_TIME_RE = re.compile(r"^\s*Parallel\s*Time:\s*([0-9]*\.?[0-9]+)\s*s\s*$", re.IGNORECASE)
ELAPSED_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+time\s+elapsed\s*$", re.IGNORECASE)
USER_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+user\s*$", re.IGNORECASE)
SYS_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+sys\s*$", re.IGNORECASE)
PERF_METRIC_RE = re.compile(
    r"^\s*(?P<value><not counted>|<not supported>|[\d,]+(?:\.\d+)?)\s+"
    r"(?P<scope>cpu_(?:core|atom))/(?P<metric>[A-Za-z0-9_.-]+)/(?P<rest>.*)$"
)
IPC_RE = re.compile(r"([0-9]*\.?[0-9]+)\s+insn\s+per\s+cycle", re.IGNORECASE)

NUMERIC_COLUMNS = [
    "matrix_size",
    "threads",
    "parallel_time_seconds",
    "elapsed_seconds",
    "user_seconds",
    "sys_seconds",
    "cpu_core_cycles",
    "cpu_core_instructions",
    "cpu_core_cache_references",
    "cpu_core_cache_misses",
    "cpu_core_l1_dcache_loads",
    "cpu_core_l1_dcache_load_misses",
    "cpu_core_llc_loads",
    "cpu_core_llc_load_misses",
    "cpu_core_ipc",
]


def _to_float(value: str) -> float:
    value = value.strip()
    if value in {"<not counted>", "<not supported>"}:
        return np.nan
    return float(value.replace(",", ""))


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = pd.to_numeric(numer, errors="coerce")
    d = pd.to_numeric(denom, errors="coerce")
    out = np.where((d > 0) & d.notna() & n.notna(), n / d, np.nan)
    return pd.Series(out, index=n.index)


def parse_perf_file(file_path: Path) -> pd.DataFrame:
    """Parse a perf output file and return one row per thread configuration."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    strategy_name = STRATEGY_LABELS.get(file_path.name, file_path.stem)
    records: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    def finalize_record() -> None:
        nonlocal current
        if not current:
            return
        if pd.isna(current.get("threads", np.nan)) or pd.isna(current.get("matrix_size", np.nan)):
            current = None
            return
        records.append(current)
        current = None

    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            header_match = HEADER_RE.search(line)
            if header_match:
                finalize_record()
                current = {
                    "source_file": file_path.name,
                    "strategy_name": strategy_name,
                    "method": header_match.group("method"),
                    "matrix_size": float(header_match.group("size")),
                    "threads": float(header_match.group("threads")),
                }
                continue

            if current is None:
                continue

            parallel_time_match = PARALLEL_TIME_RE.match(line)
            if parallel_time_match:
                current["parallel_time_seconds"] = float(parallel_time_match.group(1))
                continue

            elapsed_match = ELAPSED_RE.match(line)
            if elapsed_match:
                current["elapsed_seconds"] = float(elapsed_match.group(1))
                continue

            user_match = USER_RE.match(line)
            if user_match:
                current["user_seconds"] = float(user_match.group(1))
                continue

            sys_match = SYS_RE.match(line)
            if sys_match:
                current["sys_seconds"] = float(sys_match.group(1))
                continue

            metric_match = PERF_METRIC_RE.match(line)
            if metric_match:
                scope = metric_match.group("scope")
                metric = metric_match.group("metric")
                value = _to_float(metric_match.group("value"))
                rest = metric_match.group("rest") or ""

                suffix = METRIC_SUFFIX_MAP.get(metric)
                if suffix:
                    current[f"{scope}_{suffix}"] = value

                if metric == "instructions" and scope == "cpu_core":
                    ipc_match = IPC_RE.search(rest)
                    if ipc_match:
                        current["cpu_core_ipc"] = float(ipc_match.group(1))

    finalize_record()

    if not records:
        columns = ["source_file", "strategy_name", "method"] + NUMERIC_COLUMNS
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics using only parsed Exercise 2 text data."""
    out = df.copy()

    if out.empty:
        return out

    pt = pd.to_numeric(out["parallel_time_seconds"], errors="coerce")
    n = pd.to_numeric(out["matrix_size"], errors="coerce")
    elapsed = pd.to_numeric(out["elapsed_seconds"], errors="coerce")
    user = pd.to_numeric(out["user_seconds"], errors="coerce")
    sys = pd.to_numeric(out["sys_seconds"], errors="coerce")
    cycles = pd.to_numeric(out["cpu_core_cycles"], errors="coerce")
    instructions = pd.to_numeric(out["cpu_core_instructions"], errors="coerce")

    out["gflops"] = np.where(
        (pt > 0) & n.notna(),
        (2.0 * (n**3)) / (pt * 1e9),
        np.nan,
    )

    out["ipc"] = out["cpu_core_ipc"].combine_first(_safe_div(instructions, cycles))

    out["cache_miss_rate"] = _safe_div(
        out["cpu_core_cache_misses"], out["cpu_core_cache_references"]
    )
    out["l1_miss_rate"] = _safe_div(
        out["cpu_core_l1_dcache_load_misses"], out["cpu_core_l1_dcache_loads"]
    )
    out["llc_miss_rate"] = _safe_div(out["cpu_core_llc_load_misses"], out["cpu_core_llc_loads"])

    out["user_cpu_ratio"] = _safe_div(user, elapsed)
    out["sys_cpu_ratio"] = _safe_div(sys, elapsed)
    out["gflops_per_thread"] = _safe_div(out["gflops"], out["threads"])

    out["speedup"] = np.nan
    out["efficiency"] = np.nan

    for strategy, group in out.groupby("strategy_name"):
        baseline_rows = group[group["threads"] == BASELINE_THREADS]
        if baseline_rows.empty:
            continue

        baseline_time = baseline_rows["parallel_time_seconds"].iloc[0]
        if pd.isna(baseline_time) or baseline_time <= 0:
            continue

        idx = out["strategy_name"] == strategy
        out.loc[idx, "speedup"] = baseline_time / out.loc[idx, "parallel_time_seconds"]
        out.loc[idx, "efficiency"] = out.loc[idx, "speedup"] / (
            out.loc[idx, "threads"] / BASELINE_THREADS
        )

    out.sort_values(["strategy_name", "threads"], inplace=True)
    return out


def add_incremental_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add marginal scaling metrics by strategy across sorted thread counts."""
    out = df.copy()
    out["incremental_speedup_gain"] = np.nan
    out["incremental_time_reduction_pct"] = np.nan

    for strategy, group in out.groupby("strategy_name"):
        g = group.sort_values("threads")
        speed_gain = g["speedup"].diff()
        prev_time = g["parallel_time_seconds"].shift(1)
        time_reduction = (prev_time - g["parallel_time_seconds"]) / prev_time

        out.loc[g.index, "incremental_speedup_gain"] = speed_gain
        out.loc[g.index, "incremental_time_reduction_pct"] = time_reduction

    return out


def plot_line(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    include_ideal_speedup: bool = False,
) -> None:
    """Create a consistent line plot for all strategies."""
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "P", "X"]

    for i, (strategy, group) in enumerate(df.groupby("strategy_name")):
        g = group.sort_values(x_col)
        ax.plot(
            g[x_col],
            g[y_col],
            marker=markers[i % len(markers)],
            linewidth=2.0,
            markersize=6,
            label=strategy,
        )

    if include_ideal_speedup:
        x_vals = sorted(df[x_col].dropna().unique())
        ideal = [x / BASELINE_THREADS for x in x_vals]
        ax.plot(
            x_vals,
            ideal,
            linestyle="--",
            linewidth=1.8,
            color="black",
            label="ideal linear speedup",
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    color_col: Optional[str] = None,
    annotate_threads: bool = False,
) -> None:
    """Create a strategy-colored scatter plot with optional colormap."""
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "P", "X"]

    color_norm = None
    color_values = None
    if color_col is not None and color_col in df.columns:
        color_values = pd.to_numeric(df[color_col], errors="coerce")
        if color_values.notna().any():
            color_norm = (float(color_values.min(skipna=True)), float(color_values.max(skipna=True)))

    last_scatter = None
    for i, (strategy, group) in enumerate(df.groupby("strategy_name")):
        g = group.sort_values("threads")
        kwargs = {
            "marker": markers[i % len(markers)],
            "s": 95,
            "alpha": 0.85,
            "edgecolors": "black",
            "linewidths": 0.5,
            "label": strategy,
        }

        if color_col is not None and color_col in g.columns and color_norm is not None:
            kwargs["c"] = g[color_col]
            kwargs["cmap"] = plt.cm.viridis
            kwargs["vmin"] = color_norm[0]
            kwargs["vmax"] = color_norm[1]

        last_scatter = ax.scatter(g[x_col], g[y_col], **kwargs)

        if annotate_threads and "threads" in g.columns:
            for _, row in g.iterrows():
                if pd.notna(row.get(x_col, np.nan)) and pd.notna(row.get(y_col, np.nan)) and pd.notna(
                    row.get("threads", np.nan)
                ):
                    ax.annotate(
                        f"{int(row['threads'])}",
                        (row[x_col], row[y_col]),
                        textcoords="offset points",
                        xytext=(0, 6),
                        ha="center",
                        fontsize=8,
                    )

    if color_col is not None and color_norm is not None and last_scatter is not None:
        cbar = fig.colorbar(last_scatter, ax=ax)
        cbar.set_label(color_col)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _detect_saturation(group: pd.DataFrame) -> Optional[float]:
    """Estimate where speedup first shows diminishing marginal gain."""
    g = group.sort_values("threads")
    if g["incremental_speedup_gain"].isna().all():
        return None

    for _, row in g.iterrows():
        gain = row.get("incremental_speedup_gain", np.nan)
        if pd.notna(gain) and gain <= 0.2:
            return row.get("threads", np.nan)

    return None


def print_scaling_insights(df: pd.DataFrame) -> None:
    """Print short, data-grounded scaling insights from parsed txt metrics only."""
    if df.empty:
        print("No parsed data available for insights.")
        return

    print("=== Scaling Insights (Exercise 2) ===")

    best_time = df.loc[df["parallel_time_seconds"].idxmin()]
    print(
        "Best strategy by lowest execution time: "
        f"{best_time['strategy_name']} at {int(best_time['threads'])} threads."
    )

    best_gflops = df.loc[df["gflops"].idxmax()]
    print(
        "Best strategy by highest GFLOPS: "
        f"{best_gflops['strategy_name']} at {int(best_gflops['threads'])} threads."
    )

    for strategy, group in df.groupby("strategy_name"):
        g = group.sort_values("threads")

        sat_thread = _detect_saturation(group)
        if sat_thread is None:
            print(
                f"{strategy}: no clear saturation point detected within measured thread range."
            )
        else:
            print(
                f"{strategy}: speedup begins to saturate around {int(sat_thread)} threads."
            )

        # First thread count where time reduction drops below a practical gain threshold.
        weak = g[(g["incremental_time_reduction_pct"].notna()) & (g["incremental_time_reduction_pct"] <= 0.05)]
        if not weak.empty:
            first_weak = weak.iloc[0]
            print(
                f"{strategy}: diminishing elapsed-time returns begin around {int(first_weak['threads'])} threads."
            )

        # User CPU cost trend compared to elapsed-time improvement at the top end.
        if len(g) >= 2:
            prev = g.iloc[-2]
            last = g.iloc[-1]
            if (
                pd.notna(prev["user_seconds"])
                and pd.notna(last["user_seconds"])
                and pd.notna(prev["parallel_time_seconds"])
                and pd.notna(last["parallel_time_seconds"])
            ):
                elapsed_improvement = prev["parallel_time_seconds"] - last["parallel_time_seconds"]
                user_growth = last["user_seconds"] - prev["user_seconds"]
                if user_growth > 0 and elapsed_improvement <= 1.0:
                    print(
                        f"{strategy}: user time keeps rising while elapsed-time gains become small at high thread counts."
                    )

        best_per_thread = g.loc[g["gflops_per_thread"].idxmax()]
        print(
            f"{strategy}: best per-thread return appears at {int(best_per_thread['threads'])} threads."
        )

    parallel_for = df[df["strategy_name"] == "omp parallel for"]
    collapse = df[df["strategy_name"] == "omp parallel for collapse(2)"]
    simd = df[df["strategy_name"] == "omp for simd"]

    max_threads = int(df["threads"].max())

    if not parallel_for.empty and not collapse.empty:
        p_final = parallel_for[parallel_for["threads"] == max_threads]
        c_final = collapse[collapse["threads"] == max_threads]
        if not p_final.empty and not c_final.empty and pd.notna(p_final["speedup"].iloc[0]) and pd.notna(
            c_final["speedup"].iloc[0]
        ):
            if c_final["speedup"].iloc[0] > p_final["speedup"].iloc[0]:
                print("collapse(2): helps scaling versus omp parallel for at the highest measured threads.")
            elif c_final["speedup"].iloc[0] < p_final["speedup"].iloc[0]:
                print("collapse(2): hurts scaling versus omp parallel for at the highest measured threads.")
            else:
                print("collapse(2): similar scaling to omp parallel for at the highest measured threads.")

    if not simd.empty:
        simd_final = simd[simd["threads"] == max_threads]
        others_final = df[(df["strategy_name"] != "omp for simd") & (df["threads"] == max_threads)]
        if not simd_final.empty and not others_final.empty:
            simd_speed = simd_final["speedup"].iloc[0]
            best_other_speed = others_final["speedup"].max(skipna=True)
            if pd.notna(simd_speed) and pd.notna(best_other_speed):
                if simd_speed > best_other_speed:
                    print("simd: helps scaling at the highest measured threads.")
                elif simd_speed < best_other_speed:
                    print("simd: hurts scaling at the highest measured threads compared with the best non-simd strategy.")
                else:
                    print("simd: similar scaling at the highest measured threads.")

    if df["speedup"].notna().any() and df["threads"].notna().any():
        ideal = df["threads"] / BASELINE_THREADS
        below_ideal = (df["speedup"] < ideal).mean(skipna=True)
        if pd.notna(below_ideal) and below_ideal > 0.5:
            print("Overall: speedup remains sub-linear for most points, consistent with overhead and contention.")


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary_cols = [
        "strategy_name",
        "matrix_size",
        "threads",
        "parallel_time_seconds",
        "elapsed_seconds",
        "user_seconds",
        "sys_seconds",
        "gflops",
        "speedup",
        "efficiency",
        "ipc",
        "cache_miss_rate",
        "l1_miss_rate",
        "llc_miss_rate",
        "user_cpu_ratio",
        "sys_cpu_ratio",
        "gflops_per_thread",
        "incremental_speedup_gain",
        "incremental_time_reduction_pct",
    ]

    out = df.copy()
    for col in summary_cols:
        if col not in out.columns:
            out[col] = np.nan

    out = out[summary_cols].sort_values(["strategy_name", "threads"])
    return out


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    plots_dir = script_dir / "plots_ex2"
    plots_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for name in INPUT_FILES:
        frames.append(parse_perf_file(script_dir / name))

    raw_df = pd.concat(frames, ignore_index=True)
    metrics_df = compute_metrics(raw_df)
    metrics_df = add_incremental_metrics(metrics_df)

    metrics_df.sort_values(["strategy_name", "threads"], inplace=True)

    parsed_csv_path = script_dir / "parsed_ex2_perf_data.csv"
    summary_csv_path = script_dir / "ex2_summary_metrics.csv"

    metrics_df.to_csv(parsed_csv_path, index=False)
    _build_summary(metrics_df).to_csv(summary_csv_path, index=False)

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="parallel_time_seconds",
        output_path=plots_dir / "execution_time_vs_threads.png",
        title="Execution Time vs Threads",
        x_label="threads",
        y_label="parallel time (s)",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="speedup",
        output_path=plots_dir / "speedup_vs_threads.png",
        title="Speedup vs Threads (baseline: 4 threads)",
        x_label="threads",
        y_label="speedup",
        include_ideal_speedup=True,
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="efficiency",
        output_path=plots_dir / "efficiency_vs_threads.png",
        title="Parallel Efficiency vs Threads",
        x_label="threads",
        y_label="efficiency",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="gflops",
        output_path=plots_dir / "gflops_vs_threads.png",
        title="GFLOPS vs Threads",
        x_label="threads",
        y_label="GFLOPS",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="ipc",
        output_path=plots_dir / "ipc_vs_threads.png",
        title="IPC vs Threads",
        x_label="threads",
        y_label="IPC",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="cache_miss_rate",
        output_path=plots_dir / "cache_miss_rate_vs_threads.png",
        title="Cache Miss Rate vs Threads",
        x_label="threads",
        y_label="cache miss rate",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="user_seconds",
        output_path=plots_dir / "user_time_vs_threads.png",
        title="Total CPU User Time vs Threads (not wall-clock)",
        x_label="threads",
        y_label="user seconds",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="sys_seconds",
        output_path=plots_dir / "sys_time_vs_threads.png",
        title="System Time vs Threads",
        x_label="threads",
        y_label="sys seconds",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="user_cpu_ratio",
        output_path=plots_dir / "user_cpu_ratio_vs_threads.png",
        title="User CPU Ratio vs Threads",
        x_label="threads",
        y_label="user_seconds / elapsed_seconds",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="gflops_per_thread",
        output_path=plots_dir / "gflops_per_thread_vs_threads.png",
        title="GFLOPS per Thread vs Threads",
        x_label="threads",
        y_label="GFLOPS per thread",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="incremental_speedup_gain",
        output_path=plots_dir / "incremental_speedup_gain_vs_threads.png",
        title="Incremental Speedup Gain vs Threads",
        x_label="threads",
        y_label="speedup_n - speedup_prev",
    )

    plot_line(
        metrics_df,
        x_col="threads",
        y_col="incremental_time_reduction_pct",
        output_path=plots_dir / "incremental_time_reduction_pct_vs_threads.png",
        title="Incremental Time Reduction vs Threads",
        x_label="threads",
        y_label="(time_prev - time_n) / time_prev",
    )

    plot_scatter(
        metrics_df,
        x_col="threads",
        y_col="gflops",
        output_path=plots_dir / "gflops_vs_threads_scatter.png",
        title="GFLOPS vs Threads (color = speedup)",
        x_label="threads",
        y_label="GFLOPS",
        color_col="speedup",
        annotate_threads=True,
    )

    plot_scatter(
        metrics_df,
        x_col="elapsed_seconds",
        y_col="user_seconds",
        output_path=plots_dir / "elapsed_vs_user_time_scatter.png",
        title="Elapsed Time vs Total User CPU Time",
        x_label="elapsed seconds",
        y_label="user seconds",
    )

    plot_scatter(
        metrics_df,
        x_col="ipc",
        y_col="speedup",
        output_path=plots_dir / "speedup_vs_ipc_scatter.png",
        title="Speedup vs IPC",
        x_label="IPC",
        y_label="speedup",
    )

    print(f"Saved parsed performance data to: {parsed_csv_path}")
    print(f"Saved summary metrics to: {summary_csv_path}")
    print(f"Saved plots to: {plots_dir}")
    print_scaling_insights(metrics_df)


if __name__ == "__main__":
    main()
