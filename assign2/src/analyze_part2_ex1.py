#!/usr/bin/env python3
"""Part 2 - Exercise 1 performance parser and plotting utility."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILES = [
    "perf_results_onmult_pragma_omp_parallel_ex1.txt",
    "perf_results_onmult_pragma_omp_parallel_2_ex1.txt",
    "perf_results_onmultline_pragma_omp_parallel_ex1.txt",
]

STRATEGY_LABELS = {
    "perf_results_onmult_pragma_omp_parallel_ex1.txt": "OnMult - omp parallel for",
    "perf_results_onmult_pragma_omp_parallel_2_ex1.txt": "OnMult - omp parallel + omp for",
    "perf_results_onmultline_pragma_omp_parallel_ex1.txt": "OnMultLine - parallel",
}

# Optional baselines. Fill values if you want speedup/efficiency from known sequential times.
SEQ_BASELINES = {
    "OnMult": {},
    "OnMultLine": {},
}

SEQ_BASELINE_FILES = {
    "OnMult": "perf_results_onmult.txt",
    "OnMultLine": "perf_results_onmultline.txt",
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
    "mem_load_retired.l1_miss": "mem_load_retired_l1_miss",
    "mem_load_retired.l2_miss": "mem_load_retired_l2_miss",
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
    "cpu_core_mem_load_retired_l1_miss",
    "cpu_core_mem_load_retired_l2_miss",
    "cpu_core_ipc",
]

SEQ_MATRIX_RE = re.compile(r"Matrix\s+size:\s*(?P<size>\d+)x\d+", re.IGNORECASE)
SEQ_TIME_RE = re.compile(r"^\s*Time:\s*([0-9]*\.?[0-9]+)\s*seconds\s*$", re.IGNORECASE)
SEQ_METHOD_RE = re.compile(r"Running\s+(OnMultLine|OnMult)\s+with\s+size", re.IGNORECASE)


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


def _load_sequential_baselines(search_root: Path) -> Dict[str, Dict[int, float]]:
    """Load sequential baselines from known Part 1 perf files if available."""
    found: Dict[str, Dict[int, float]] = {"OnMult": {}, "OnMultLine": {}}

    candidates = [
        search_root / "assign1" / "src",
        search_root.parent / "assign1" / "src",
        search_root.parent.parent / "assign1" / "src",
        search_root,
    ]

    seq_dir = next((c for c in candidates if c.exists()), None)
    if seq_dir is None:
        return found

    for method, filename in SEQ_BASELINE_FILES.items():
        path = seq_dir / filename
        if not path.exists():
            continue

        current_size: Optional[int] = None
        current_method: Optional[str] = None

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.rstrip("\n")

                matrix_match = SEQ_MATRIX_RE.search(line)
                if matrix_match:
                    current_size = int(matrix_match.group("size"))
                    continue

                method_match = SEQ_METHOD_RE.search(line)
                if method_match:
                    current_method = method_match.group(1)
                    continue

                time_match = SEQ_TIME_RE.match(line)
                if time_match and current_size is not None:
                    parsed_method = current_method if current_method in {"OnMult", "OnMultLine"} else method
                    found[parsed_method][current_size] = float(time_match.group(1))

    return found


def parse_perf_file(file_path: Path) -> pd.DataFrame:
    """Parse one perf text file into a normalized dataframe."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    strategy_name = STRATEGY_LABELS.get(file_path.name, file_path.stem)
    records: List[Dict[str, object]] = []
    current: Optional[Dict[str, object]] = None

    def finalize_record() -> None:
        nonlocal current
        if not current:
            return
        if pd.isna(current.get("matrix_size", np.nan)):
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

            pt_match = PARALLEL_TIME_RE.match(line)
            if pt_match:
                current["parallel_time_seconds"] = float(pt_match.group(1))
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
        return pd.DataFrame(columns=["source_file", "strategy_name", "method"] + NUMERIC_COLUMNS)

    df = pd.DataFrame(records)
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def compute_metrics(
    df: pd.DataFrame,
    effective_baselines: Optional[Dict[str, Dict[int, float]]] = None,
) -> pd.DataFrame:
    """Compute derived metrics, speedup, and efficiency."""
    out = df.copy()

    n = pd.to_numeric(out["matrix_size"], errors="coerce")
    pt = pd.to_numeric(out["parallel_time_seconds"], errors="coerce")
    elapsed = pd.to_numeric(out["elapsed_seconds"], errors="coerce")

    out["gflops"] = np.where((pt > 0) & n.notna(), (2.0 * (n**3)) / (pt * 1e9), np.nan)
    out["gflops_elapsed"] = np.where(
        (elapsed > 0) & n.notna(), (2.0 * (n**3)) / (elapsed * 1e9), np.nan
    )

    out["cache_miss_rate"] = _safe_div(
        out["cpu_core_cache_misses"], out["cpu_core_cache_references"]
    )
    out["l1_miss_rate"] = _safe_div(
        out["cpu_core_l1_dcache_load_misses"], out["cpu_core_l1_dcache_loads"]
    )
    out["llc_miss_rate"] = _safe_div(out["cpu_core_llc_load_misses"], out["cpu_core_llc_loads"])
    # L2 miss rate approximated from retired misses ratio.
    out["l2_miss_rate"] = _safe_div(
        out["cpu_core_mem_load_retired_l2_miss"],
        out["cpu_core_mem_load_retired_l1_miss"],
    )

    ipc_calc = _safe_div(out["cpu_core_instructions"], out["cpu_core_cycles"])
    out["ipc"] = out["cpu_core_ipc"].combine_first(ipc_calc)

    out["normalized_l1_misses_per_element"] = _safe_div(
        out["cpu_core_l1_dcache_load_misses"], n**2
    )

    out["sequential_baseline_seconds"] = np.nan
    out["speedup"] = np.nan
    out["efficiency"] = np.nan

    missing_baselines: List[Tuple[str, int]] = []

    baselines = effective_baselines if effective_baselines is not None else SEQ_BASELINES

    for idx, row in out.iterrows():
        method = str(row.get("method", ""))
        size = row.get("matrix_size", np.nan)
        p_time = row.get("parallel_time_seconds", np.nan)

        if pd.isna(size) or pd.isna(p_time) or p_time <= 0:
            continue

        size_int = int(size)
        baseline = baselines.get(method, {}).get(size_int, np.nan)

        if pd.notna(baseline):
            out.at[idx, "sequential_baseline_seconds"] = baseline
            out.at[idx, "speedup"] = baseline / p_time
            threads = row.get("threads", np.nan)
            if pd.notna(threads) and threads > 0:
                out.at[idx, "efficiency"] = (baseline / p_time) / threads
        else:
            missing_baselines.append((method, size_int))

    if missing_baselines:
        unique_missing = sorted(set(missing_baselines))
        print("Warning: missing sequential baseline for these (method, size) entries:")
        for method, size in unique_missing:
            print(f"  - {method}, N={size}")

    return out


def plot_line(
    df: pd.DataFrame,
    y_col: str,
    output_path: Path,
    title: str,
    y_label: str,
    x_col: str = "matrix_size",
    hue_col: str = "strategy_name",
) -> None:
    """Generic line plot helper."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in sorted(df[hue_col].dropna().unique()):
        subset = df[df[hue_col] == strategy].sort_values(x_col)
        series = subset[[x_col, y_col]].dropna()
        if series.empty:
            continue
        ax.plot(
            series[x_col].astype(int),
            series[y_col],
            marker="o",
            linewidth=2,
            markersize=6,
            label=strategy,
        )

    ax.set_title(title)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.has_data():
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    hue_col: str = "strategy_name",
) -> None:
    """Generic scatter plot helper."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in sorted(df[hue_col].dropna().unique()):
        subset = df[df[hue_col] == strategy][[x_col, y_col]].dropna()
        if subset.empty:
            continue
        ax.scatter(subset[x_col], subset[y_col], s=55, alpha=0.75, label=strategy)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.has_data():
        ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build summary tables and best strategy table by size."""
    summary_cols = [
        "matrix_size",
        "strategy_name",
        "method",
        "threads",
        "parallel_time_seconds",
        "elapsed_seconds",
        "gflops",
        "gflops_elapsed",
        "ipc",
        "cache_miss_rate",
        "l1_miss_rate",
        "llc_miss_rate",
        "l2_miss_rate",
        "normalized_l1_misses_per_element",
        "speedup",
        "efficiency",
    ]

    summary = df[summary_cols].copy()
    summary = summary.sort_values(["matrix_size", "strategy_name"]).reset_index(drop=True)

    best_idx = summary.dropna(subset=["gflops"]).groupby("matrix_size")["gflops"].idxmax()
    best_by_size = summary.loc[best_idx].sort_values("matrix_size").reset_index(drop=True)

    return summary, best_by_size


def _find_fastest_per_size(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["matrix_size", "parallel_time_seconds"])
    if valid.empty:
        return pd.DataFrame(columns=df.columns)
    idx = valid.groupby("matrix_size")["parallel_time_seconds"].idxmin()
    return valid.loc[idx].sort_values("matrix_size")


def _best_gflops_per_size(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.dropna(subset=["matrix_size", "gflops"])
    if valid.empty:
        return pd.DataFrame(columns=df.columns)
    idx = valid.groupby("matrix_size")["gflops"].idxmax()
    return valid.loc[idx].sort_values("matrix_size")


def print_insights(df: pd.DataFrame) -> None:
    """Print concise interpretation and comparisons."""
    fastest = _find_fastest_per_size(df)
    best_gflops = _best_gflops_per_size(df)

    print("\nFastest strategy per matrix size (by parallel time):")
    if fastest.empty:
        print("  No valid data.")
    else:
        for _, row in fastest.iterrows():
            print(
                f"  N={int(row['matrix_size'])}: {row['strategy_name']} "
                f"({row['parallel_time_seconds']:.6f}s)"
            )

    print("\nBest GFLOPS per matrix size:")
    if best_gflops.empty:
        print("  No valid data.")
    else:
        for _, row in best_gflops.iterrows():
            print(
                f"  N={int(row['matrix_size'])}: {row['strategy_name']} "
                f"({row['gflops']:.3f} GFLOPS)"
            )

    v1 = df[df["method"] == "OnMult"].copy()
    v2 = df[df["method"] == "OnMultLine"].copy()

    print("\nBest Version 1 strategy (OnMult):")
    if v1.empty:
        print("  No Version 1 records found.")
    else:
        agg = v1.groupby("strategy_name")["gflops"].mean().sort_values(ascending=False)
        best_name = agg.index[0]
        print(f"  {best_name} (average GFLOPS: {agg.iloc[0]:.3f})")

    print("\nVersion 1 vs Version 2 comparison:")
    if v1.empty or v2.empty:
        print("  Not enough data for direct comparison.")
    else:
        v1_avg = v1["gflops"].mean(skipna=True)
        v2_avg = v2["gflops"].mean(skipna=True)
        if pd.notna(v1_avg) and pd.notna(v2_avg):
            ratio = v2_avg / v1_avg if v1_avg > 0 else np.nan
            print(f"  Avg GFLOPS Version1: {v1_avg:.3f}")
            print(f"  Avg GFLOPS Version2: {v2_avg:.3f}")
            if pd.notna(ratio):
                print(f"  Version2/Version1 GFLOPS ratio: {ratio:.3f}x")

    corr_cols = df[["gflops", "l1_miss_rate", "l2_miss_rate", "ipc"]].copy()
    miss_corr = corr_cols[["gflops", "l1_miss_rate"]].corr().iloc[0, 1]
    l2_miss_corr = corr_cols[["gflops", "l2_miss_rate"]].corr().iloc[0, 1]
    ipc_corr = corr_cols[["gflops", "ipc"]].corr().iloc[0, 1]

    print("\nCorrelation insights:")
    if pd.notna(miss_corr):
        print(f"  corr(GFLOPS, L1 miss rate): {miss_corr:.4f}")
    else:
        print("  corr(GFLOPS, L1 miss rate): not enough valid points")

    if pd.notna(l2_miss_corr):
        print(f"  corr(GFLOPS, L2 miss rate): {l2_miss_corr:.4f}")
    else:
        print("  corr(GFLOPS, L2 miss rate): not enough valid points")

    if pd.notna(ipc_corr):
        print(f"  corr(GFLOPS, IPC): {ipc_corr:.4f}")
    else:
        print("  corr(GFLOPS, IPC): not enough valid points")

    print("\nInterpretation goal checks:")
    print("  - Version 1 tends to show lower IPC and weaker locality than Version 2.")
    print("  - Version 2 generally performs better due to improved access pattern.")
    print("  - OpenMP strategy impacts performance through scheduling/parallel overhead.")
    print("  - Lower miss rates often align with higher IPC and higher GFLOPS.")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    plot_dir = base_dir / "plots_ex1"

    frames: List[pd.DataFrame] = []
    for name in INPUT_FILES:
        file_path = base_dir / name
        try:
            parsed = parse_perf_file(file_path)
            if parsed.empty:
                print(f"Warning: no records parsed from {file_path.name}")
            frames.append(parsed)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    if not frames:
        raise RuntimeError("No input files could be parsed.")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("Parsed dataframe is empty.")

    auto_baselines = _load_sequential_baselines(base_dir)
    effective_baselines = {
        "OnMult": dict(auto_baselines.get("OnMult", {})),
        "OnMultLine": dict(auto_baselines.get("OnMultLine", {})),
    }

    # User-provided values in SEQ_BASELINES override auto-loaded ones.
    for method in ("OnMult", "OnMultLine"):
        effective_baselines[method].update(SEQ_BASELINES.get(method, {}))

    loaded_counts = {k: len(v) for k, v in effective_baselines.items()}
    print(
        "Sequential baselines available:",
        f"OnMult={loaded_counts['OnMult']}, OnMultLine={loaded_counts['OnMultLine']}",
    )

    df = compute_metrics(df, effective_baselines=effective_baselines)
    df = df.sort_values(["matrix_size", "strategy_name"]).reset_index(drop=True)

    summary_df, best_by_size = build_summary(df)

    parsed_csv = base_dir / "parsed_ex1_perf_data.csv"
    summary_csv = base_dir / "ex1_summary_metrics.csv"
    best_csv = base_dir / "ex1_best_strategy_by_size.csv"

    df.to_csv(parsed_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    best_by_size.to_csv(best_csv, index=False)

    # Core line plots
    plot_line(df, "gflops", plot_dir / "gflops_vs_size.png", "GFLOPS vs Matrix Size", "GFLOPS")
    plot_line(
        df,
        "parallel_time_seconds",
        plot_dir / "parallel_time_vs_size.png",
        "Parallel Time vs Matrix Size",
        "Parallel time (s)",
    )
    plot_line(
        df,
        "elapsed_seconds",
        plot_dir / "elapsed_time_vs_size.png",
        "Elapsed Time vs Matrix Size",
        "Elapsed time (s)",
    )
    plot_line(df, "speedup", plot_dir / "speedup_vs_size.png", "Speedup vs Matrix Size", "Speedup")
    plot_line(
        df,
        "efficiency",
        plot_dir / "efficiency_vs_size.png",
        "Efficiency vs Matrix Size",
        "Efficiency",
    )
    plot_line(df, "ipc", plot_dir / "ipc_vs_size.png", "IPC vs Matrix Size", "IPC")
    plot_line(
        df,
        "l1_miss_rate",
        plot_dir / "l1_miss_rate_vs_size.png",
        "L1 Miss Rate vs Matrix Size",
        "L1 miss rate",
    )
    plot_line(
        df,
        "cache_miss_rate",
        plot_dir / "cache_miss_rate_vs_size.png",
        "Cache Miss Rate vs Matrix Size",
        "Cache miss rate",
    )
    plot_line(
        df,
        "llc_miss_rate",
        plot_dir / "llc_miss_rate_vs_size.png",
        "LLC Miss Rate vs Matrix Size",
        "LLC miss rate",
    )
    plot_line(
        df,
        "l2_miss_rate",
        plot_dir / "l2_miss_rate_vs_size.png",
        "L2 Miss Rate vs Matrix Size",
        "L2 miss rate",
    )

    # Scatter plots
    plot_scatter(
        df,
        "l1_miss_rate",
        "gflops",
        plot_dir / "gflops_vs_l1_miss_rate.png",
        "GFLOPS vs L1 Miss Rate",
        "L1 miss rate",
        "GFLOPS",
    )
    plot_scatter(
        df,
        "l1_miss_rate",
        "ipc",
        plot_dir / "ipc_vs_l1_miss_rate.png",
        "IPC vs L1 Miss Rate",
        "L1 miss rate",
        "IPC",
    )
    plot_scatter(
        df,
        "l2_miss_rate",
        "gflops",
        plot_dir / "gflops_vs_l2_miss_rate.png",
        "GFLOPS vs L2 Miss Rate",
        "L2 miss rate",
        "GFLOPS",
    )
    plot_scatter(
        df,
        "l2_miss_rate",
        "ipc",
        plot_dir / "ipc_vs_l2_miss_rate.png",
        "IPC vs L2 Miss Rate",
        "L2 miss rate",
        "IPC",
    )

    # Version 1 strategy comparison
    version1 = df[df["method"] == "OnMult"].copy()
    plot_line(
        version1,
        "gflops",
        plot_dir / "version1_strategy_comparison_gflops.png",
        "Version 1 Strategies: GFLOPS Comparison",
        "GFLOPS",
    )

    # Version 1 best strategy vs Version 2
    best_v1_per_size = (
        version1.sort_values(["matrix_size", "gflops"], ascending=[True, False])
        .dropna(subset=["matrix_size"]) 
        .drop_duplicates(subset=["matrix_size"])
    )
    version2 = df[df["method"] == "OnMultLine"].copy()
    version2 = version2.sort_values("matrix_size").drop_duplicates(subset=["matrix_size"])

    compare_rows: List[pd.DataFrame] = []
    if not best_v1_per_size.empty:
        tmp = best_v1_per_size.copy()
        tmp["strategy_name"] = "Version1 best strategy"
        compare_rows.append(tmp)
    if not version2.empty:
        tmp = version2.copy()
        tmp["strategy_name"] = "Version2 OnMultLine"
        compare_rows.append(tmp)
    if compare_rows:
        compare_df = pd.concat(compare_rows, ignore_index=True)
        plot_line(
            compare_df,
            "gflops",
            plot_dir / "version1_vs_version2_gflops.png",
            "Version 1 (best) vs Version 2 GFLOPS",
            "GFLOPS",
        )

    print(f"Loaded files from: {base_dir}")
    print(f"Parsed records: {len(df)}")
    print(f"Plots saved in: {plot_dir}")
    print(f"CSV outputs: {parsed_csv.name}, {summary_csv.name}, {best_csv.name}")

    print_insights(df)


if __name__ == "__main__":
    main()
