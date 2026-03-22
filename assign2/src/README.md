# Parallel Matrix Multiplication Analysis (Part 2 – Exercises 1 and 2)

## 1. Introduction
Part 2 studies OpenMP matrix multiplication from two complementary angles: strategy design and scalability. Exercise 1 compares different parallel constructs under the same thread budget, while Exercise 2 analyzes how behavior evolves as thread count increases.

The central question is not only which version is faster, but why: how much time is spent doing useful arithmetic versus coordinating threads, synchronizing work, and paying runtime overhead. This makes parallel efficiency and thread scaling the key interpretation goals.

## 2. Experimental Setup
- Exercise 1: fixed 4 threads, matrix sizes from 1024 to 3072.
- Exercise 2: fixed matrix size 8192, threads varied from 4 to 24.
- Implementations are evaluated with perf counters and timing data from OpenMP runs.
- Compared strategies:
	- Exercise 1: OnMult - omp parallel for, OnMult - omp parallel + omp for, OnMultLine - parallel.
	- Exercise 2: omp parallel for, omp parallel for collapse(2), omp for simd.

## 3. Metrics Overview
- Execution time: shows end-to-end cost and whether added parallelism translates into real wall-clock benefit.
- GFLOPS: measures sustained useful floating-point throughput under each strategy.
- Speedup: reveals how much parallel execution improves runtime relative to a baseline configuration.
- Efficiency: normalizes speedup by thread growth to expose scaling quality.
- IPC: indicates how effectively cores convert cycles into useful instructions.
- Cache miss rate: provides context for memory pressure while assessing whether overhead or data movement dominates behavior.

## 4. Exercise 1 – Parallel Strategy Comparison

### 4.1 Execution Time
![](plots_ex1/parallel_time_vs_size.png)

![](plots_ex1/gflops_vs_size.png)

With threads fixed, the dominant difference comes from parallel region structure rather than raw parallel capacity. Strategies that fuse work-sharing and team management reduce runtime bookkeeping, so a larger fraction of time is spent on arithmetic.

The approach based on omp parallel for tends to perform better because it avoids extra orchestration layers. By contrast, splitting constructs into omp parallel + omp for can introduce additional synchronization and control-path overhead, which is more visible as problem size grows.

### 4.2 GFLOPS
![](plots_ex1/gflops_vs_size.png)

![](plots_ex1/version1_vs_version2_gflops.png)

GFLOPS differs mainly by how efficiently each strategy converts thread activity into useful computation. When overhead remains low, throughput stays higher because threads remain compute-active instead of waiting on runtime coordination.

The mild throughput decline at larger sizes is consistent with practical scaling limits: fixed thread count, increasing orchestration cost, and lower ability to maintain ideal compute intensity across the full workload.

### 4.3 Speedup
![](plots_ex1/speedup_vs_size.png)

Speedup differences reflect how much parallel benefit survives after overhead is paid. A strategy can be parallel in syntax yet still under-deliver if synchronization, scheduling cost, or imbalance consumes too much execution time.

This is why compact OpenMP constructs generally scale better in this comparison: they preserve a higher useful-work fraction as size increases.

### 4.4 IPC
![](plots_ex1/ipc_vs_size.png)

Higher IPC aligns with better parallel efficiency because cores are retiring more instructions per cycle in productive sections. Lower IPC is a sign that threads spend more time stalled or underutilized, which is consistent with excess synchronization or runtime control overhead.

In this exercise, IPC supports the same interpretation as time and GFLOPS: stronger strategies are those that keep execution pipelines busy with useful work.

### 4.5 Short Cache Note
![](plots_ex1/cache_miss_rate_vs_size.png)

Cache behavior still affects performance, but it is not the primary separator in Part 2. The larger explanatory signal here is parallel runtime efficiency: how each OpenMP structure manages thread coordination overhead.

### 4.6 Strategy Comparison
![](plots_ex1/version1_strategy_comparison_gflops.png)

omp parallel for is typically more efficient in this context because it couples team creation and loop partitioning in one directive, reducing structural overhead and avoiding unnecessary synchronization boundaries.

omp parallel + omp for is more flexible in complex regions, but for this workload that extra structure tends to increase management cost without adding proportional useful computation.

## 5. Exercise 2 – Thread Scaling Analysis

### 5.1 Execution Time vs Threads
![](plots_ex2/execution_time_vs_threads.png)

Execution time drops quickly at first because initial thread additions remove obvious serial pressure. After that phase, gains shrink: each extra thread contributes less wall-clock improvement because coordination and shared-resource contention become more relevant.

This transition marks the shift from compute-limited behavior to overhead-limited scaling.

### 5.2 Speedup vs Threads
![](plots_ex2/speedup_vs_threads.png)

Speedup is sub-linear because parallel overhead grows with thread count and hardware resources are finite. The ideal line assumes zero coordination cost and unlimited resource scaling, which is not achievable in practice.

Saturation appears around the mid-to-high thread range, where additional threads yield smaller incremental speedup and may occasionally produce negligible benefit.

### 5.3 Efficiency vs Threads
![](plots_ex2/efficiency_vs_threads.png)

Efficiency declines as threads increase because the marginal benefit per thread drops. This is expected when runtime overhead, synchronization, and contention consume a larger fraction of execution than at low thread counts.

The key interpretation is that adding threads can improve throughput while still reducing per-thread effectiveness.

### 5.4 GFLOPS vs Threads
![](plots_ex2/gflops_vs_threads.png)

![](plots_ex2/gflops_vs_threads_scatter.png)

GFLOPS rises while parallel work remains efficiently exploitable, then tends to plateau as overhead and contention begin to offset additional parallel capacity.

This pattern is typical of shared-memory scaling: beyond a threshold, throughput gains become constrained by coordination costs and hardware bottlenecks rather than available threads alone.

### 5.5 IPC vs Threads
![](plots_ex2/ipc_vs_threads.png)

IPC trends help explain why scaling weakens at higher thread counts. As thread pressure increases, cores can spend more cycles in less productive states, lowering instruction retirement effectiveness.

When IPC stabilizes or drops while thread count keeps rising, it usually indicates that added concurrency is no longer translating into proportional useful progress.

### 5.6 Advanced Scaling Insights
![](plots_ex2/user_time_vs_threads.png)

![](plots_ex2/user_cpu_ratio_vs_threads.png)

![](plots_ex2/gflops_per_thread_vs_threads.png)

![](plots_ex2/incremental_speedup_gain_vs_threads.png)

![](plots_ex2/incremental_time_reduction_pct_vs_threads.png)

![](plots_ex2/elapsed_vs_user_time_scatter.png)

These plots make the scaling trade-off explicit. Total CPU work (user time) grows with thread count, but wall-clock improvement eventually slows. That means the system is spending increasingly more aggregate compute effort for smaller latency gains.

GFLOPS per thread and incremental speedup gain expose diminishing returns directly: early thread increases add clear value, while later increases contribute less useful acceleration. Incremental time reduction confirms where extra threads stop being cost-effective.

### 5.7 Strategy Comparison
![](plots_ex2/speedup_vs_ipc_scatter.png)

omp parallel for provides a strong baseline because of low structural overhead and straightforward work-sharing.

omp parallel for collapse(2) can improve load distribution when loop-space partitioning is uneven, but it may also increase scheduling complexity and coordination overhead depending on runtime behavior.

omp for simd can improve per-thread compute throughput via vectorization, yet overall scaling still depends on thread-level overhead and resource contention. In practice, simd and collapse decisions are beneficial only when their added control costs are smaller than their compute gains.

## 6. Discussion
The combined evidence from Exercises 1 and 2 shows that OpenMP performance is governed by useful-work ratio, not thread count alone. Strategies that minimize synchronization boundaries and runtime orchestration consistently preserve better throughput.

Scaling degrades when the system transitions from compute-limited to overhead-limited behavior. At that point, more threads mainly amplify contention and management cost, so performance improvements become incremental rather than proportional.

The practical lesson is to optimize structure first, then scale threads within the range where incremental gains remain meaningful.

## 7. Conclusion
Part 2 shows that the best OpenMP strategy is the one that maximizes useful computation while minimizing parallel overhead.  
Exercise 1 highlights the impact of construct design on efficiency, even with the same thread budget.  
Exercise 2 demonstrates that scaling is inherently limited: speedup saturates, efficiency drops, and marginal gains shrink as threads increase.  
Sustained performance therefore depends on balanced thread utilization and low coordination cost, not simply on adding more threads.
