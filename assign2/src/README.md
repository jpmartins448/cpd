# Parallel Matrix Multiplication Analysis (Part 2 – Exercises 1 and 2)

## 1. Introduction
Part 2 analyzes OpenMP matrix multiplication with direct comparison of concrete implementations. In Exercise 1, the comparison is between OnMult - omp parallel for, OnMult - omp parallel + omp for, and OnMultLine - parallel with fixed 4 threads. In Exercise 2, the comparison is between OnMultLine variants using omp parallel for, omp parallel for collapse(2), and omp for simd while scaling threads from 4 to 24.

The goal is to explain why each implementation behaves as observed, based on its OpenMP structure: where parallel regions are created, how loop iterations are distributed, how much synchronization is introduced, and how thread-level overhead affects scalability.

## 2. Experimental Setup
Exercise 1 runs with 4 threads and matrix sizes from 1024 to 3072, isolating implementation differences without changing thread count. Exercise 2 runs at fixed size 8192 and scales threads from 4 to 24 to evaluate thread scalability of omp parallel for, omp parallel for collapse(2), and omp for simd in OnMultLine.

All measurements come from perf outputs, combining timing and hardware counters to interpret OpenMP overhead, thread utilization, and scaling saturation.

## 3. Metrics Overview
Execution time captures whether a parallel strategy converts hardware resources into actual wall-clock reduction.

GFLOPS measures sustained floating-point throughput and therefore how much useful arithmetic is delivered per unit time.

Speedup indicates how much faster a parallel configuration runs relative to a baseline and is the primary indicator of practical scaling.

Efficiency normalizes speedup by thread growth, showing whether additional threads still contribute proportionally.

IPC (instructions per cycle) reflects how effectively cores are utilized during execution rather than stalled or underfed.

Cache miss rate provides context for whether memory-side pressure may be reinforcing or limiting observed parallel gains.

## 4. Exercise 1 – Parallel Strategy Comparison

### 4.1 Execution Time
![](plots_ex1/parallel_time_vs_size.png)

![](plots_ex1/gflops_vs_size.png)

The execution-time trend shows that OnMult - omp parallel for and OnMultLine - parallel stay ahead of OnMult - omp parallel + omp for as size increases. The same ordering is reflected in the GFLOPS plot, where the slower implementation also delivers lower throughput.

The main reason is the OpenMP structure used in OnMult. In OnMult - omp parallel for, thread-team creation and iteration distribution are combined in one directive, so loop work is dispatched directly with minimal control overhead. In OnMult - omp parallel + omp for, the split structure introduces extra runtime coordination and synchronization points around the work-sharing region, which increases non-compute cost. OnMultLine - parallel also benefits from a loop organization that keeps thread work steady enough that coordination cost does not dominate.

The implication is that OnMult - omp parallel for and OnMultLine - parallel convert a larger fraction of wall time into useful multiplication work, while OnMult - omp parallel + omp for pays a higher overhead tax per workload increase.

### 4.2 GFLOPS
![](plots_ex1/gflops_vs_size.png)

![](plots_ex1/version1_vs_version2_gflops.png)

The GFLOPS plots show that OnMult - omp parallel for sustains higher throughput than OnMult - omp parallel + omp for, while OnMultLine - parallel remains competitive across sizes. The version-to-version comparison confirms that the gap is consistent rather than incidental.

This happens because GFLOPS penalizes runtime overhead directly: any time spent in synchronization or OpenMP control logic lowers effective floating-point throughput. OnMult - omp parallel for keeps this overhead lower by using a single combined parallel/work-sharing construct. OnMult - omp parallel + omp for introduces additional runtime handling around the separate directives, so threads spend more time outside arithmetic sections. OnMultLine - parallel keeps useful work density relatively high, which helps preserve throughput.

The implication is that the best GFLOPS in Exercise 1 comes from implementations where OpenMP directives map more directly to computation, especially OnMult - omp parallel for.

### 4.3 Speedup
![](plots_ex1/speedup_vs_size.png)

The speedup curve indicates that OnMult - omp parallel for and OnMultLine - parallel realize better parallel gains than OnMult - omp parallel + omp for over the size range tested. The weaker speedup of OnMult - omp parallel + omp for is persistent enough to be treated as a structural effect.

The cause is that speedup is net acceleration after overhead. OnMult - omp parallel + omp for introduces more synchronization and runtime management around the work-sharing sequence, so part of the potential gain from 4 threads is consumed before useful multiply work completes. OnMult - omp parallel for avoids part of this cost by dispatching loop chunks in a single construct, and OnMultLine - parallel maintains competitive thread utilization under the same thread budget.

The implication is that in this project, speedup differences are primarily explained by OpenMP construct overhead, not by thread count or algorithmic intent alone.

### 4.4 IPC
![](plots_ex1/ipc_vs_size.png)

The IPC behavior is consistent with the performance ordering: OnMult - omp parallel for and OnMultLine - parallel tend to maintain stronger effective instruction retirement than OnMult - omp parallel + omp for. Where IPC is lower, execution also tends to show weaker speedup/GFLOPS.

This is expected from the directive structure. OnMult - omp parallel + omp for introduces more runtime control and synchronization exposure, which can leave cores spending more cycles in less productive phases. OnMult - omp parallel for keeps the loop distribution path tighter, so threads spend more cycles in compute-dense sections. OnMultLine - parallel also benefits when thread work remains consistently occupied.

The implication is that IPC supports the same conclusion as time and speedup: OnMult - omp parallel for and OnMultLine - parallel convert thread activity into useful execution more effectively than OnMult - omp parallel + omp for.

### 4.5 Short Cache Note
![](plots_ex1/cache_miss_rate_vs_size.png)

The cache-miss plot changes with size for all implementations, but it does not cleanly explain why OnMult - omp parallel for and OnMultLine - parallel outperform OnMult - omp parallel + omp for. The main ranking is better aligned with OpenMP overhead differences than with cache-miss separation alone.

Because Exercise 1 keeps threads fixed, this graph is informative but not decisive. The implication is that the primary causal driver remains directive-level runtime overhead, while cache effects should be interpreted as secondary context.

### 4.6 Strategy Comparison
![](plots_ex1/version1_strategy_comparison_gflops.png)

The direct comparison confirms that OnMult - omp parallel for is consistently ahead of OnMult - omp parallel + omp for in throughput-oriented behavior. OnMultLine - parallel remains a strong reference in the same exercise context.

The cause is explicit in code structure: omp parallel for creates one parallel region and distributes iterations immediately, reducing directive transition overhead. The omp parallel + omp for variant separates region creation and work-sharing, which increases runtime coordination and can introduce extra synchronization boundaries that are not offset by extra useful work in this kernel.

The implication is that OnMult - omp parallel for is the better OpenMP choice in Exercise 1 because its construct-level overhead is lower for this matrix multiplication pattern.

## 5. Exercise 2 – Thread Scaling Analysis

### 5.1 Execution Time vs Threads
![](plots_ex2/execution_time_vs_threads.png)

The execution-time plot shows that all three OnMultLine variants improve from 4 threads to higher counts, but the gains are strongest early and weaker later. The comparison between omp parallel for, omp parallel for collapse(2), and omp for simd indicates that their scaling trajectories diverge as thread count grows.

This pattern is expected from the OpenMP constructs. At low-to-mid thread counts, each added thread removes a large amount of remaining parallel work. At higher thread counts, omp parallel for and omp parallel for collapse(2) face increasing scheduling and synchronization overhead, while omp for simd can still benefit from stronger per-thread vectorized compute but is not immune to thread coordination cost. As concurrency rises, shared-resource contention limits how much additional wall-clock reduction each new thread can provide.

The implication is that thread scaling quality depends on how each construct balances extra parallel workers against extra runtime overhead, not on thread count alone.

### 5.2 Speedup vs Threads
![](plots_ex2/speedup_vs_threads.png)

The speedup curves are sub-linear for omp parallel for, omp parallel for collapse(2), and omp for simd, with visible saturation in the higher-thread region. None of the three methods follows ideal linear scaling once thread count becomes large.

The cause is a combination of OpenMP overhead and resource contention. omp parallel for pays growing coordination cost as the team grows; omp parallel for collapse(2) expands iteration-space distribution, which can improve balance but also adds scheduling pressure; omp for simd improves compute density inside each thread, yet thread-level synchronization and runtime overhead still prevent linear growth. As a result, each additional thread contributes less incremental speedup than earlier additions.

The implication is that saturation is a structural property of these implementations under this hardware context, and high thread counts should be evaluated by marginal gain rather than ideal scaling expectations.

### 5.3 Efficiency vs Threads
![](plots_ex2/efficiency_vs_threads.png)

Efficiency falls for omp parallel for, omp parallel for collapse(2), and omp for simd as threads increase, even where execution time still decreases. This indicates that parallel gains continue, but each new thread contributes less than proportional ideal scaling.

This decline happens because thread growth amplifies non-compute costs in all three constructs. omp parallel for and omp parallel for collapse(2) both accumulate more synchronization and scheduling overhead at scale, while omp for simd, despite better per-thread arithmetic throughput, still pays thread-management overhead that grows with team size.

The implication is that the most scalable method is the one whose efficiency decays slowest, not necessarily the one with the largest absolute thread count.

### 5.4 GFLOPS vs Threads
![](plots_ex2/gflops_vs_threads.png)

![](plots_ex2/gflops_vs_threads_scatter.png)

GFLOPS increases sharply at first for omp parallel for, omp parallel for collapse(2), and omp for simd, then trends toward a plateau where additional threads produce smaller throughput gains. The scatter confirms that higher threads do not automatically map to proportionally higher GFLOPS.

The reason differs slightly by method. omp parallel for has straightforward work-sharing and scales well initially but eventually loses marginal throughput to coordination and contention. omp parallel for collapse(2) can improve distribution granularity across nested loops, which may help utilization in some regions, but added scheduling complexity can offset that benefit at higher threads. omp for simd improves per-thread compute intensity through vectorization, which can raise GFLOPS early, yet global throughput still plateaus when thread-level overhead and shared-resource contention dominate.

The implication is that omp for simd can improve per-thread compute effectiveness, but long-range scalability is still determined by OpenMP coordination cost across the full thread team.

### 5.5 IPC vs Threads
![](plots_ex2/ipc_vs_threads.png)

IPC does not scale linearly with thread count for omp parallel for, omp parallel for collapse(2), or omp for simd, and high-thread regions show weaker per-cycle effectiveness than low-thread regions. This aligns with the speedup and efficiency saturation behavior.

The cause is that more threads increase synchronization frequency, runtime scheduling activity, and pressure on shared execution resources. omp for simd can retain stronger compute density within each thread, but team-level contention still limits IPC growth. omp parallel for collapse(2) may improve work distribution in nested loops, yet it also introduces overhead that can dilute instruction retirement effectiveness at scale.

The implication is that IPC confirms a central scaling limit in Exercise 2: additional threads increase total activity, but not proportionally productive activity.

### 5.6 Advanced Scaling Insights
![](plots_ex2/user_time_vs_threads.png)

![](plots_ex2/user_cpu_ratio_vs_threads.png)

![](plots_ex2/gflops_per_thread_vs_threads.png)

![](plots_ex2/incremental_speedup_gain_vs_threads.png)

![](plots_ex2/incremental_time_reduction_pct_vs_threads.png)

![](plots_ex2/elapsed_vs_user_time_scatter.png)

Across these graphs, omp parallel for, omp parallel for collapse(2), and omp for simd all show the same high-level pattern: total CPU work rises with thread count while marginal wall-clock benefit shrinks. user_time_vs_threads and user_cpu_ratio_vs_threads indicate that higher-thread runs consume significantly more aggregate CPU time, even when elapsed-time improvements become modest.

The incremental metrics explain why this happens. incremental_speedup_gain_vs_threads and incremental_time_reduction_pct_vs_threads show that each thread increment contributes less acceleration than the previous one, revealing explicit diminishing returns. gflops_per_thread_vs_threads adds the same message at thread granularity: per-thread productivity declines as the team grows because OpenMP coordination overhead and shared-resource contention absorb a larger fraction of execution. elapsed_vs_user_time_scatter visualizes the trade-off directly: improved elapsed time increasingly requires disproportionately higher user CPU effort.

The implication is that the practical scaling limit for omp parallel for, omp parallel for collapse(2), and omp for simd is identified by marginal gain collapse, not by the highest absolute thread count.

### 5.7 Strategy Comparison
![](plots_ex2/speedup_vs_ipc_scatter.png)

The speedup-versus-IPC scatter separates omp parallel for, omp parallel for collapse(2), and omp for simd into different trade-off regions rather than a single dominant line. This indicates that each implementation balances instruction-level effectiveness and thread-level scaling differently.

The cause is rooted in construct behavior. omp parallel for provides a stable baseline with low directive overhead. omp parallel for collapse(2) can distribute nested-loop iterations more broadly, which may help balance but can also increase scheduling overhead. omp for simd can improve arithmetic throughput inside each thread via vectorization, but overall speedup still depends on synchronization and contention across the OpenMP team.

Because point clusters may overlap in parts of the plot, some pairwise conclusions should be interpreted cautiously. The implication remains concrete: the best implementation is the one that keeps both IPC and speedup high simultaneously across the scaled thread range.

## 6. Discussion
Exercise 1 shows that OnMult - omp parallel for and OnMultLine - parallel achieve better performance than OnMult - omp parallel + omp for primarily because of lower OpenMP coordination overhead under the same 4-thread budget. The code-level directive choice changes how much runtime is spent in useful multiplication versus synchronization and runtime control.

Exercise 2 shows that omp parallel for, omp parallel for collapse(2), and omp for simd all experience diminishing returns as threads increase. The key difference is not whether they scale at all, but how quickly marginal gain declines once overhead and contention become comparable to useful work.

The practical conclusion is implementation-specific: choose the OpenMP construct that preserves useful-work density for this kernel, then stop increasing threads when incremental speedup and incremental time reduction indicate saturation.

## 7. Conclusion
In this project, OnMult - omp parallel for and OnMultLine - parallel outperform OnMult - omp parallel + omp for in Exercise 1 because their OpenMP structure spends less time in coordination overhead.  
In Exercise 2, omp parallel for, omp parallel for collapse(2), and omp for simd all improve performance at first but eventually hit diminishing returns as synchronization and contention grow.  
collapse(2) and simd can provide benefits in specific regions, but neither removes the fundamental saturation limit of thread scaling.  
The best practical choice is the implementation that maintains the highest useful-work-to-overhead ratio across the target thread range.
