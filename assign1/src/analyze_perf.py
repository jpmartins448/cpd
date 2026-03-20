#!/usr/bin/env python3
"""Parse, compare, and plot matrix multiplication performance results."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MATRIX_RE = re.compile(
    r"Matrix\s+size:\s*(?P<n>\d+)\s*x\s*\d+(?:\s*,\s*Block\s+size:\s*(?P<b>\d+))?",
    re.IGNORECASE,
)
RUN_RE = re.compile(
    r"Running\s+(?P<algo>OnMult(?:Line|Block)?)\s+with\s+size\s+(?P<n>\d+)x\d+(?:\s*,\s*block\s+size\s+(?P<b>\d+))?",
    re.IGNORECASE,
)
TIME_RE = re.compile(r"^\s*Time:\s*([0-9]*\.?[0-9]+)\s*seconds\s*$", re.IGNORECASE)
ELAPSED_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+time\s+elapsed\s*$", re.IGNORECASE)
USER_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+user\s*$", re.IGNORECASE)
SYS_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s+seconds\s+sys\s*$", re.IGNORECASE)

PERF_METRIC_RE = re.compile(
    r"^\s*(?P<value><not counted>|<not supported>|[\d,]+(?:\.\d+)?)\s+"
    r"(?P<scope>cpu_(?:core|atom))/(?P<metric>[A-Za-z0-9_.-]+)/(?P<rest>.*)$"
)
IPC_RE = re.compile(r"([0-9]*\.?[0-9]+)\s+insn\s+per\s+cycle", re.IGNORECASE)

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

BASE_COLUMNS = [
    "algorithm",
    "matrix_size",
    "block_size",
    "time_seconds",
    "elapsed_seconds",
    "user_seconds",
    "sys_seconds",
    "ipc_core",
    "ipc_atom",
]

for scope in ("cpu_core", "cpu_atom"):
    for suffix in METRIC_SUFFIX_MAP.values():
        BASE_COLUMNS.append(f"{scope}_{suffix}")


def _to_float(value: str) -> float:
    v = value.strip()
    if v in {"<not counted>", "<not supported>"}:
        return float("nan")
    return float(v.replace(",", ""))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return np.where((den > 0) & den.notna() & num.notna(), num / den, np.nan)


def _normalize_record(record: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(record)
    for col in BASE_COLUMNS:
        normalized.setdefault(col, np.nan)

    normalized["algorithm"] = str(normalized.get("algorithm", ""))
    if pd.isna(normalized.get("block_size")):
        normalized["block_size"] = np.nan

    return normalized


def parse_perf_file(file_path: Path, default_algorithm: str) -> List[Dict[str, object]]:
    """Parse a perf results file into a list of benchmark records."""
    records: List[Dict[str, object]] = []

    if not file_path.exists():
        raise FileNotFoundError(f"Missing input file: {file_path}")

    current: Optional[Dict[str, object]] = None

    def finalize_current() -> None:
        nonlocal current
        if not current:
            return
        if pd.isna(current.get("matrix_size", np.nan)):
            current = None
            return
        records.append(_normalize_record(current))
        current = None

    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            matrix_match = MATRIX_RE.search(line)
            if matrix_match:
                finalize_current()
                current = {
                    "algorithm": default_algorithm,
                    "matrix_size": float(matrix_match.group("n")),
                    "block_size": (
                        float(matrix_match.group("b"))
                        if matrix_match.group("b") is not None
                        else np.nan
                    ),
                }
                continue

            if current is None:
                continue

            run_match = RUN_RE.search(line)
            if run_match:
                current["algorithm"] = run_match.group("algo")
                current["matrix_size"] = float(run_match.group("n"))
                current["block_size"] = (
                    float(run_match.group("b")) if run_match.group("b") else np.nan
                )
                continue

            time_match = TIME_RE.match(line)
            if time_match:
                current["time_seconds"] = float(time_match.group(1))
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

                if metric == "instructions":
                    ipc_match = IPC_RE.search(rest)
                    if ipc_match:
                        current[f"ipc_{'core' if scope == 'cpu_core' else 'atom'}"] = float(
                            ipc_match.group(1)
                        )

    finalize_current()
    return records


def parse_onmult_file(file_path: Path) -> List[Dict[str, object]]:
    return parse_perf_file(file_path, "OnMult")


def parse_onmultline_file(file_path: Path) -> List[Dict[str, object]]:
    return parse_perf_file(file_path, "OnMultLine")


def parse_onmultblock_file(file_path: Path) -> List[Dict[str, object]]:
    return parse_perf_file(file_path, "OnMultBlock")


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived metrics used in comparisons and plots."""
    out = df.copy()

    out["cache_miss_rate"] = _safe_divide(
        out["cpu_core_cache_misses"], out["cpu_core_cache_references"]
    )
    out["l1_miss_rate"] = _safe_divide(
        out["cpu_core_l1_dcache_load_misses"], out["cpu_core_l1_dcache_loads"]
    )
    out["llc_miss_rate"] = _safe_divide(
        out["cpu_core_llc_load_misses"], out["cpu_core_llc_loads"]
    )

    calc_ipc = _safe_divide(out["cpu_core_instructions"], out["cpu_core_cycles"])
    out["instructions_per_cycle"] = out["ipc_core"].combine_first(pd.Series(calc_ipc, index=out.index))

    n = pd.to_numeric(out["matrix_size"], errors="coerce")
    elapsed = pd.to_numeric(out["elapsed_seconds"], errors="coerce")
    out["gflops_estimate"] = np.where(
        (elapsed > 0) & elapsed.notna() & n.notna(),
        (2.0 * (n**3)) / elapsed / 1e9,
        np.nan,
    )

    n2 = n**2
    out["normalized_cache_misses_per_element"] = _safe_divide(
        out["cpu_core_cache_misses"], n2
    )
    out["normalized_l1_misses_per_element"] = _safe_divide(
        out["cpu_core_l1_dcache_load_misses"], n2
    )

    baseline_onmult = (
        out[out["algorithm"] == "OnMult"]
        .dropna(subset=["elapsed_seconds"])
        .groupby("matrix_size")["elapsed_seconds"]
        .min()
    )
    baseline_onmultline = (
        out[out["algorithm"] == "OnMultLine"]
        .dropna(subset=["elapsed_seconds"])
        .groupby("matrix_size")["elapsed_seconds"]
        .min()
    )

    out["speedup_vs_onmult"] = out["matrix_size"].map(baseline_onmult) / out["elapsed_seconds"]
    out["speedup_vs_onmultline"] = np.where(
        out["algorithm"].eq("OnMultBlock"),
        out["matrix_size"].map(baseline_onmultline) / out["elapsed_seconds"],
        np.nan,
    )

    return out


def _best_per_algorithm(df: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    subset = df[df["algorithm"] == algorithm].copy()
    subset = subset.dropna(subset=["matrix_size", "elapsed_seconds"])
    if subset.empty:
        return subset
    idx = subset.groupby("matrix_size")["elapsed_seconds"].idxmin()
    return subset.loc[idx].sort_values("matrix_size")


def build_summary_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build summary comparison tables and best blocked configs."""
    onmult_best = _best_per_algorithm(df, "OnMult")
    line_best = _best_per_algorithm(df, "OnMultLine")
    block_best = _best_per_algorithm(df, "OnMultBlock")

    all_block_rows = (
        df[df["algorithm"] == "OnMultBlock"]
        .dropna(subset=["matrix_size", "block_size"])
        .sort_values(["matrix_size", "block_size", "elapsed_seconds"])
    )

    onmult_comp = onmult_best[["matrix_size", "elapsed_seconds", "time_seconds", "gflops_estimate", "cache_miss_rate", "l1_miss_rate", "llc_miss_rate"]].rename(
        columns={
            "elapsed_seconds": "onmult_elapsed_seconds",
            "time_seconds": "onmult_time_seconds",
            "gflops_estimate": "onmult_gflops_estimate",
            "cache_miss_rate": "onmult_cache_miss_rate",
            "l1_miss_rate": "onmult_l1_miss_rate",
            "llc_miss_rate": "onmult_llc_miss_rate",
        }
    )

    line_comp = line_best[["matrix_size", "elapsed_seconds", "time_seconds", "gflops_estimate", "cache_miss_rate", "l1_miss_rate", "llc_miss_rate", "speedup_vs_onmult"]].rename(
        columns={
            "elapsed_seconds": "onmultline_elapsed_seconds",
            "time_seconds": "onmultline_time_seconds",
            "gflops_estimate": "onmultline_gflops_estimate",
            "cache_miss_rate": "onmultline_cache_miss_rate",
            "l1_miss_rate": "onmultline_l1_miss_rate",
            "llc_miss_rate": "onmultline_llc_miss_rate",
            "speedup_vs_onmult": "onmultline_speedup_vs_onmult",
        }
    )

    block_comp = block_best[[
        "matrix_size",
        "block_size",
        "elapsed_seconds",
        "time_seconds",
        "gflops_estimate",
        "cache_miss_rate",
        "l1_miss_rate",
        "llc_miss_rate",
        "speedup_vs_onmult",
        "speedup_vs_onmultline",
    ]].rename(
        columns={
            "block_size": "best_block_size",
            "elapsed_seconds": "best_block_elapsed_seconds",
            "time_seconds": "best_block_time_seconds",
            "gflops_estimate": "best_block_gflops_estimate",
            "cache_miss_rate": "best_block_cache_miss_rate",
            "l1_miss_rate": "best_block_l1_miss_rate",
            "llc_miss_rate": "best_block_llc_miss_rate",
            "speedup_vs_onmult": "best_block_speedup_vs_onmult",
            "speedup_vs_onmultline": "best_block_speedup_vs_onmultline",
        }
    )

    summary = pd.merge(onmult_comp, line_comp, on="matrix_size", how="outer")
    summary = pd.merge(summary, block_comp, on="matrix_size", how="outer")
    summary = summary.sort_values("matrix_size").reset_index(drop=True)

    def fastest_label(row: pd.Series) -> str:
        candidates = {
            "OnMult": row.get("onmult_elapsed_seconds", np.nan),
            "OnMultLine": row.get("onmultline_elapsed_seconds", np.nan),
            "OnMultBlock(best)": row.get("best_block_elapsed_seconds", np.nan),
        }
        valid = {k: v for k, v in candidates.items() if pd.notna(v)}
        if not valid:
            return "N/A"
        return min(valid, key=valid.get)

    summary["fastest_algorithm"] = summary.apply(fastest_label, axis=1)

    return summary, block_best, all_block_rows


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _line_plot_from_series(
    series_map: Dict[str, pd.Series],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, series in series_map.items():
        s = series.dropna().sort_index()
        if s.empty:
            continue
        ax.plot(s.index.astype(int), s.values, marker="o", linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.has_data():
        ax.legend()
    _save_plot(fig, out_path)


def plot_elapsed_time_vs_size(summary: pd.DataFrame, out_dir: Path) -> str:
    series = {
        "OnMult": summary.set_index("matrix_size")["onmult_elapsed_seconds"],
        "OnMultLine": summary.set_index("matrix_size")["onmultline_elapsed_seconds"],
        "OnMultBlock (best)": summary.set_index("matrix_size")["best_block_elapsed_seconds"],
    }
    filename = "elapsed_time_vs_size.png"
    _line_plot_from_series(series, "Elapsed Time vs Matrix Size", "Elapsed time (s)", out_dir / filename)
    return filename


def plot_program_time_vs_size(summary: pd.DataFrame, out_dir: Path) -> str:
    series = {
        "OnMult": summary.set_index("matrix_size")["onmult_time_seconds"],
        "OnMultLine": summary.set_index("matrix_size")["onmultline_time_seconds"],
        "OnMultBlock (best)": summary.set_index("matrix_size")["best_block_time_seconds"],
    }
    filename = "program_time_vs_size.png"
    _line_plot_from_series(series, "Program Reported Time vs Matrix Size", "Time (s)", out_dir / filename)
    return filename


def plot_gflops_vs_size(summary: pd.DataFrame, out_dir: Path) -> str:
    series = {
        "OnMult": summary.set_index("matrix_size")["onmult_gflops_estimate"],
        "OnMultLine": summary.set_index("matrix_size")["onmultline_gflops_estimate"],
        "OnMultBlock (best)": summary.set_index("matrix_size")["best_block_gflops_estimate"],
    }
    filename = "gflops_vs_size.png"
    _line_plot_from_series(series, "GFLOPS Estimate vs Matrix Size", "GFLOPS", out_dir / filename)
    return filename


def plot_ipc_vs_size(df: pd.DataFrame, out_dir: Path) -> str:
    filename = "ipc_vs_size.png"
    series = {
        "OnMult": _best_per_algorithm(df, "OnMult").set_index("matrix_size")["instructions_per_cycle"],
        "OnMultLine": _best_per_algorithm(df, "OnMultLine").set_index("matrix_size")["instructions_per_cycle"],
        "OnMultBlock (best)": _best_per_algorithm(df, "OnMultBlock").set_index("matrix_size")["instructions_per_cycle"],
    }
    _line_plot_from_series(series, "Instructions per Cycle vs Matrix Size", "IPC", out_dir / filename)
    return filename


def plot_cache_miss_rate_vs_size(summary: pd.DataFrame, out_dir: Path) -> str:
    filename = "cache_miss_rate_vs_size.png"
    series = {
        "OnMult": summary.set_index("matrix_size")["onmult_cache_miss_rate"],
        "OnMultLine": summary.set_index("matrix_size")["onmultline_cache_miss_rate"],
        "OnMultBlock (best)": summary.set_index("matrix_size")["best_block_cache_miss_rate"],
    }
    _line_plot_from_series(series, "Cache Miss Rate vs Matrix Size", "Miss rate", out_dir / filename)
    return filename


def plot_l1_miss_rate_vs_size(summary: pd.DataFrame, out_dir: Path) -> str:
    filename = "l1_miss_rate_vs_size.png"
    series = {
        "OnMult": summary.set_index("matrix_size")["onmult_l1_miss_rate"],
        "OnMultLine": summary.set_index("matrix_size")["onmultline_l1_miss_rate"],
        "OnMultBlock (best)": summary.set_index("matrix_size")["best_block_l1_miss_rate"],
    }
    _line_plot_from_series(series, "L1 D-Cache Miss Rate vs Matrix Size", "Miss rate", out_dir / filename)
    return filename


def plot_llc_miss_rate_vs_size(summary: pd.DataFrame, out_dir: Path) -> str:
    filename = "llc_miss_rate_vs_size.png"
    series = {
        "OnMult": summary.set_index("matrix_size")["onmult_llc_miss_rate"],
        "OnMultLine": summary.set_index("matrix_size")["onmultline_llc_miss_rate"],
        "OnMultBlock (best)": summary.set_index("matrix_size")["best_block_llc_miss_rate"],
    }
    _line_plot_from_series(series, "LLC Miss Rate vs Matrix Size", "Miss rate", out_dir / filename)
    return filename


def plot_speedup_vs_onmult(summary: pd.DataFrame, out_dir: Path) -> str:
    filename = "speedup_vs_onmult.png"
    series = {
        "OnMultLine vs OnMult": summary.set_index("matrix_size")["onmultline_speedup_vs_onmult"],
        "OnMultBlock(best) vs OnMult": summary.set_index("matrix_size")["best_block_speedup_vs_onmult"],
    }
    _line_plot_from_series(series, "Speedup vs Naive OnMult", "Speedup (x)", out_dir / filename)
    return filename


def _plot_block_metric(all_block_rows: pd.DataFrame, metric_col: str, title: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for block_size in sorted(all_block_rows["block_size"].dropna().unique()):
        subset = all_block_rows[all_block_rows["block_size"] == block_size].sort_values("matrix_size")
        if subset.empty:
            continue
        ax.plot(
            subset["matrix_size"].astype(int),
            subset[metric_col],
            marker="o",
            linewidth=2,
            label=f"Block {int(block_size)}",
        )

    ax.set_title(title)
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.has_data():
        ax.legend()
    _save_plot(fig, out_path)


def plot_block_comparison_elapsed_time(all_block_rows: pd.DataFrame, out_dir: Path) -> str:
    filename = "block_comparison_elapsed_time.png"
    _plot_block_metric(
        all_block_rows,
        "elapsed_seconds",
        "Blocked Algorithm: Elapsed Time by Block Size",
        "Elapsed time (s)",
        out_dir / filename,
    )
    return filename


def plot_block_comparison_cache_miss_rate(all_block_rows: pd.DataFrame, out_dir: Path) -> str:
    filename = "block_comparison_cache_miss_rate.png"
    _plot_block_metric(
        all_block_rows,
        "cache_miss_rate",
        "Blocked Algorithm: Cache Miss Rate by Block Size",
        "Miss rate",
        out_dir / filename,
    )
    return filename


def plot_block_comparison_l1_miss_rate(all_block_rows: pd.DataFrame, out_dir: Path) -> str:
    filename = "block_comparison_l1_miss_rate.png"
    _plot_block_metric(
        all_block_rows,
        "l1_miss_rate",
        "Blocked Algorithm: L1 Miss Rate by Block Size",
        "Miss rate",
        out_dir / filename,
    )
    return filename


def plot_best_block_size_by_size(best_block: pd.DataFrame, out_dir: Path) -> str:
    filename = "best_block_size_by_size.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = best_block.dropna(subset=["matrix_size", "block_size"]).sort_values("matrix_size")
    if not subset.empty:
        ax.bar(subset["matrix_size"].astype(int).astype(str), subset["block_size"].astype(int), color="#3A6EA5")

    ax.set_title("Best Block Size per Matrix Size")
    ax.set_xlabel("Matrix size (N)")
    ax.set_ylabel("Best block size")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    _save_plot(fig, out_dir / filename)
    return filename


def plot_instructions_vs_cycles(df: pd.DataFrame, out_dir: Path) -> str:
    filename = "instructions_vs_cycles.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "OnMult": "#D1495B",
        "OnMultLine": "#2E8B57",
        "OnMultBlock": "#1F77B4",
    }

    for algo, subset in df.groupby("algorithm"):
        clean = subset.dropna(subset=["cpu_core_cycles", "cpu_core_instructions"])
        if clean.empty:
            continue
        ax.scatter(
            clean["cpu_core_cycles"],
            clean["cpu_core_instructions"],
            alpha=0.75,
            s=45,
            label=algo,
            color=palette.get(algo, None),
        )

    ax.set_title("Instructions vs Cycles (cpu_core)")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("Instructions")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.has_data():
        ax.legend()
    _save_plot(fig, out_dir / filename)
    return filename


def plot_roofline_like(df: pd.DataFrame, out_dir: Path) -> str:
    filename = "roofline_like_gflops_vs_missrate.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    palette = {
        "OnMult": "#D1495B",
        "OnMultLine": "#2E8B57",
        "OnMultBlock": "#1F77B4",
    }

    for algo, subset in df.groupby("algorithm"):
        clean = subset.dropna(subset=["gflops_estimate", "l1_miss_rate"])
        if clean.empty:
            continue
        ax.scatter(
            clean["l1_miss_rate"],
            clean["gflops_estimate"],
            alpha=0.75,
            s=45,
            label=algo,
            color=palette.get(algo, None),
        )

    ax.set_title("Roofline-like View: GFLOPS vs L1 Miss Rate")
    ax.set_xlabel("L1 miss rate")
    ax.set_ylabel("GFLOPS estimate")
    ax.grid(True, linestyle="--", alpha=0.35)
    if ax.has_data():
        ax.legend()
    _save_plot(fig, out_dir / filename)
    return filename


def _detect_sharp_changes(series: pd.Series, threshold: float = 1.8) -> List[Tuple[int, float]]:
    changes: List[Tuple[int, float]] = []
    s = series.dropna().sort_index()
    if len(s) < 2:
        return changes

    prev_idx = None
    prev_val = None
    for idx, val in s.items():
        if prev_idx is None or prev_val is None or prev_val == 0:
            prev_idx = idx
            prev_val = val
            continue
        ratio = val / prev_val
        if ratio >= threshold:
            changes.append((int(idx), float(ratio)))
        prev_idx = idx
        prev_val = val
    return changes


def _print_insights(summary: pd.DataFrame, best_block: pd.DataFrame) -> None:
    print("\nFastest algorithm per matrix size:")
    for _, row in summary.dropna(subset=["matrix_size"]).iterrows():
        size = int(row["matrix_size"])
        fastest = row.get("fastest_algorithm", "N/A")
        print(f"  N={size}: {fastest}")

    if not best_block.empty:
        print("\nBest block size per matrix size:")
        for _, row in best_block.iterrows():
            size = int(row["matrix_size"])
            blk = int(row["block_size"]) if pd.notna(row["block_size"]) else None
            elapsed = row.get("elapsed_seconds", np.nan)
            if blk is None:
                print(f"  N={size}: no block size found")
            else:
                print(f"  N={size}: block={blk} (elapsed={elapsed:.6f}s)")

    sharp = _detect_sharp_changes(summary.set_index("matrix_size")["onmult_cache_miss_rate"]) if "onmult_cache_miss_rate" in summary else []
    if sharp:
        print("\nSharp cache-miss-rate increases detected for OnMult:")
        for n, ratio in sharp:
            print(f"  Around N={n}: increase x{ratio:.2f} vs previous size")
    else:
        print("\nNo sharp cache miss rate jumps detected above threshold for OnMult.")

    if {
        "onmult_elapsed_seconds",
        "onmultline_elapsed_seconds",
    }.issubset(summary.columns):
        ratio_series = _safe_divide(
            summary["onmult_elapsed_seconds"], summary["onmultline_elapsed_seconds"]
        )
        ratio_df = pd.DataFrame({"matrix_size": summary["matrix_size"], "ratio": ratio_series}).dropna()
        dramatic = ratio_df[ratio_df["ratio"] >= 3.0]
        if not dramatic.empty:
            print("\nNaive OnMult becomes dramatically worse (>=3x slower than OnMultLine) at:")
            for _, row in dramatic.iterrows():
                print(f"  N={int(row['matrix_size'])}: x{row['ratio']:.2f} slower")
        else:
            print("\nNo matrix size exceeded the dramatic degradation threshold (3x).")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_files = {
        "OnMult": base_dir / "perf_results_onmult.txt",
        "OnMultLine": base_dir / "perf_results_onmultline.txt",
        "OnMultBlock": base_dir / "perf_results_onmultblock.txt",
    }

    missing = [str(path) for path in input_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n  " + "\n  ".join(missing))

    records: List[Dict[str, object]] = []
    records.extend(parse_onmult_file(input_files["OnMult"]))
    records.extend(parse_onmultline_file(input_files["OnMultLine"]))
    records.extend(parse_onmultblock_file(input_files["OnMultBlock"]))

    if not records:
        raise RuntimeError("No benchmark sections were parsed from the input files.")

    df = pd.DataFrame(records)
    for col in BASE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    numeric_columns = [c for c in df.columns if c not in {"algorithm"}]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = compute_derived_metrics(df)
    summary, best_block, all_block_rows = build_summary_tables(df)

    parsed_csv = base_dir / "parsed_perf_data.csv"
    best_block_csv = base_dir / "best_block_per_size.csv"
    summary_csv = base_dir / "summary_comparison.csv"

    df.sort_values(["algorithm", "matrix_size", "block_size"], na_position="last").to_csv(parsed_csv, index=False)
    best_block.to_csv(best_block_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    plot_dir = base_dir / "plots"
    generated_plots: List[str] = []
    generated_plots.append(plot_elapsed_time_vs_size(summary, plot_dir))
    generated_plots.append(plot_program_time_vs_size(summary, plot_dir))
    generated_plots.append(plot_gflops_vs_size(summary, plot_dir))
    generated_plots.append(plot_ipc_vs_size(df, plot_dir))
    generated_plots.append(plot_cache_miss_rate_vs_size(summary, plot_dir))
    generated_plots.append(plot_l1_miss_rate_vs_size(summary, plot_dir))
    generated_plots.append(plot_llc_miss_rate_vs_size(summary, plot_dir))
    generated_plots.append(plot_speedup_vs_onmult(summary, plot_dir))
    generated_plots.append(plot_block_comparison_elapsed_time(all_block_rows, plot_dir))
    generated_plots.append(plot_block_comparison_cache_miss_rate(all_block_rows, plot_dir))
    generated_plots.append(plot_block_comparison_l1_miss_rate(all_block_rows, plot_dir))
    generated_plots.append(plot_best_block_size_by_size(best_block, plot_dir))
    generated_plots.append(plot_instructions_vs_cycles(df, plot_dir))
    generated_plots.append(plot_roofline_like(df, plot_dir))

    print("Performance analysis complete.")
    print("\nLoaded files:")
    for algo, path in input_files.items():
        print(f"  {algo}: {path}")

    print(f"\nParsed benchmark records: {len(df)}")

    print("\nGenerated plots:")
    for name in generated_plots:
        print(f"  {plot_dir / name}")

    _print_insights(summary, best_block)

    if not best_block.empty:
        best_block_freq = (
            best_block["block_size"].dropna().astype(int).value_counts().sort_index()
        )
        print("\nOverall best block size frequency:")
        for blk, count in best_block_freq.items():
            print(f"  block {blk}: {count} matrix sizes")

    print("\nMain trend interpretation:")
    print("  - Naive OnMult tends to degrade much more strongly as matrix size increases.")
    print("  - OnMultLine and OnMultBlock generally improve cache behavior and throughput.")
    print("  - Blocked multiplication can have an optimal block size that depends on matrix size.")

    print("\nCSV outputs:")
    print(f"  {parsed_csv}")
    print(f"  {best_block_csv}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
