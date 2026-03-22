# Parallel Matrix Multiplication Analysis (Part 2 – Exercises 1 and 2)

## 1. Introduction
In this part of the project, the goal is not only to see which line is higher or lower in the plots, but to explain why that happens for the actual implementations we wrote. Exercise 1 compares OnMult - omp parallel for, OnMult - omp parallel + omp for, and OnMultLine - parallel with a fixed thread count. Exercise 2 uses OnMultLine with omp parallel for, omp parallel for collapse(2), and omp for simd while increasing the number of threads.

The main idea across all sections is to connect performance to OpenMP behavior in the code. That means looking at how threads are created, how loop iterations are split, when synchronization happens, and how much time is spent doing useful matrix multiplication versus runtime overhead.

## 2. Experimental Setup
Exercise 1 was executed with 4 threads and matrix sizes from 1024 to 3072. This setup is useful because it keeps thread count constant and makes it easier to isolate differences caused by the OpenMP constructs in OnMult and OnMultLine.

Exercise 2 was executed with fixed matrix size 8192 and threads from 4 to 24. This setup is focused on scaling behavior of OnMultLine with omp parallel for, omp parallel for collapse(2), and omp for simd.

All results come from perf outputs, including execution times and hardware counters. The interpretation below is based on those measurements and on how the OpenMP directives are used in each implementation.

## 3. Metrics Overview
Execution time shows whether a given implementation actually reduces wall-clock runtime.

GFLOPS shows how much useful floating-point work is sustained per second.

Speedup compares a threaded run with the 4-thread baseline used in each Exercise 2 method.

Efficiency shows how much useful speedup each added thread still contributes.

IPC shows how effectively CPU cycles are converted into retired instructions.

Cache miss rate gives secondary context when we need to explain scaling changes.

## 4. Exercise 1 – Parallel Strategy Comparison

### 4.1 Execution Time
![](plots_ex1/parallel_time_vs_size.png)

![](plots_ex1/gflops_vs_size.png)

The execution-time plot shows a clear ordering: OnMult - omp parallel for and OnMultLine - parallel stay ahead of OnMult - omp parallel + omp for as matrix size grows. The GFLOPS plot beside it supports the same result because the slower method is also the one with lower throughput.

The reason is directly related to how the OpenMP directives are used. In OnMult - omp parallel for, the runtime creates the parallel region and distributes iterations in one combined step, so threads move quickly into useful loop work. In OnMult - omp parallel + omp for, region creation and work-sharing are separated, which adds coordination and synchronization overhead around the parallel loop. OnMultLine - parallel keeps threads busy with stable work chunks and avoids spending too much time in runtime control.

The implication is that, with the same 4-thread budget, directive structure matters a lot: OnMult - omp parallel for and OnMultLine - parallel spend more time multiplying matrices, while OnMult - omp parallel + omp for loses more time in OpenMP management.

### 4.2 GFLOPS
![](plots_ex1/gflops_vs_size.png)

![](plots_ex1/version1_vs_version2_gflops.png)

The GFLOPS plots show that OnMult - omp parallel for usually stays above OnMult - omp parallel + omp for, and OnMultLine - parallel remains strong across the tested sizes. The direct comparison between the two OnMult variants confirms that this is a stable behavior, not just a random fluctuation.

The cause is simple: GFLOPS falls when threads spend time doing anything other than floating-point work. OnMult - omp parallel for has less directive overhead because it uses one combined OpenMP construct to launch and distribute work. OnMult - omp parallel + omp for adds extra control steps and synchronization around the loop, so more time is spent outside computation. OnMultLine - parallel keeps useful arithmetic density high enough to remain competitive.

The implication is that higher GFLOPS here means better conversion of thread time into real matrix operations, and that favors OnMult - omp parallel for and OnMultLine - parallel over OnMult - omp parallel + omp for.

### 4.3 Speedup
![](plots_ex1/speedup_vs_size.png)

The speedup plot again places OnMult - omp parallel for and OnMultLine - parallel above OnMult - omp parallel + omp for through most of the tested range. That means the same 4 threads are producing different levels of useful acceleration depending on the construct.

This happens because speedup is measured after all overhead is included. OnMult - omp parallel + omp for spends extra time coordinating between parallel region handling and work-sharing, so part of the theoretical gain is consumed before useful work is done. OnMult - omp parallel for has a more direct path from thread launch to loop execution, and OnMultLine - parallel keeps thread participation more consistently useful.

The implication is that speedup in Exercise 1 reflects quality of parallelization, not just number of threads, and the construct with less synchronization overhead wins.

### 4.4 IPC
![](plots_ex1/ipc_vs_size.png)

The IPC plot follows the same ranking trend seen in time and GFLOPS: OnMult - omp parallel for and OnMultLine - parallel generally keep better effective IPC than OnMult - omp parallel + omp for. Where IPC is weaker, performance is usually weaker too.

The reason is that lower IPC often appears when cores spend more cycles waiting or doing low-value control work. OnMult - omp parallel + omp for includes more runtime handling and synchronization around loop scheduling, so cores can spend a larger share of cycles outside dense arithmetic regions. OnMult - omp parallel for and OnMultLine - parallel keep threads closer to useful computation, which helps maintain instruction retirement.

The implication is that IPC is not just a side metric here; it confirms that OnMult - omp parallel for and OnMultLine - parallel use CPU cycles more productively than OnMult - omp parallel + omp for.

### 4.5 Short Cache Note
![](plots_ex1/cache_miss_rate_vs_size.png)

The cache-miss plot changes with problem size for OnMult - omp parallel for, OnMult - omp parallel + omp for, and OnMultLine - parallel, but the separation is not strong enough to fully explain the performance ranking by itself. The main ordering in runtime and GFLOPS lines up more clearly with OpenMP overhead differences.

Because Exercise 1 keeps thread count fixed, this graph should be read carefully and not over-interpreted. It gives useful context, but it does not provide a stronger explanation than the construct-level overhead we see between OnMult - omp parallel for and OnMult - omp parallel + omp for.

The implication is that cache effects exist, but in this part they are secondary; the primary explanation is still how each implementation manages threads and synchronization.

### 4.6 Strategy Comparison
![](plots_ex1/version1_strategy_comparison_gflops.png)

This direct comparison highlights the central result of Exercise 1: OnMult - omp parallel for is consistently stronger than OnMult - omp parallel + omp for, and the broader set of graphs shows OnMultLine - parallel also performing well in the same environment.

The reason is explicit in the directive design. In OnMult - omp parallel for, a single OpenMP construct handles thread team creation and loop distribution together, which avoids extra transitions. In OnMult - omp parallel + omp for, the split between region creation and work-sharing introduces additional runtime coordination and synchronization costs that do not produce extra useful matrix work.

The implication is practical and clear: for this kernel, OnMult - omp parallel for is the better construct choice than OnMult - omp parallel + omp for when the goal is to maximize useful work under fixed threads.

## 5. Exercise 2 – Thread Scaling Analysis

### 5.1 Execution Time vs Threads
![](plots_ex2/execution_time_vs_threads.png)

The execution-time plot shows that all three OnMultLine variants improve when moving from 4 threads upward, but the biggest gains happen early. As threads continue to increase, omp parallel for, omp parallel for collapse(2), and omp for simd all show slower improvement, which is a classic sign of diminishing returns.

The cause comes from thread scaling mechanics. At lower thread counts, adding threads directly increases useful parallel work. At higher counts, each method pays more overhead in scheduling and synchronization. For omp parallel for collapse(2), the larger flattened iteration space can improve distribution but also adds scheduling pressure. For omp for simd, vectorization can boost per-thread computation, but it does not remove team-level coordination costs. Shared resources also become more contested as thread count rises.

The implication is that runtime scaling should be judged by marginal gains: these methods all improve with threads, but only up to the point where overhead growth starts to dominate benefit growth.

### 5.2 Speedup vs Threads
![](plots_ex2/speedup_vs_threads.png)

The speedup plot shows that omp parallel for, omp parallel for collapse(2), and omp for simd all stay below the ideal linear trend as threads increase. The curves rise, but the slope becomes smaller in the higher-thread region, indicating saturation.

This happens because linear speedup assumes almost zero overhead, which is not realistic in OpenMP thread teams at high concurrency. omp parallel for accumulates more coordination cost as the team grows. omp parallel for collapse(2) can improve load distribution by exposing more loop iterations, but this also increases scheduling work. omp for simd can improve computation inside each thread, yet thread synchronization and resource contention still limit total scaling.

The implication is that adding threads remains useful only while incremental speedup is meaningful; after saturation starts, thread increases give smaller and smaller practical benefits.

### 5.3 Efficiency vs Threads
![](plots_ex2/efficiency_vs_threads.png)

Efficiency decreases for omp parallel for, omp parallel for collapse(2), and omp for simd as thread count increases. Even in ranges where execution time still improves, each additional thread contributes less than the previous ones.

The reason is that efficiency captures how much useful acceleration we get per thread. As teams get larger, all three methods spend a bigger fraction of runtime on synchronization, scheduling, and contention-related waiting. omp for simd can keep per-thread compute stronger than non-simd variants in some ranges, but it still cannot avoid thread-team overhead entirely.

The implication is that efficiency is a practical warning signal: the best operating point is not always the highest thread count, but the point where added threads still return enough useful acceleration.

### 5.4 GFLOPS vs Threads
![](plots_ex2/gflops_vs_threads.png)

![](plots_ex2/gflops_vs_threads_scatter.png)

The GFLOPS plot shows the same overall scaling pattern for omp parallel for, omp parallel for collapse(2), and omp for simd: throughput rises strongly at first, then begins to flatten. The scatter view confirms that moving to higher threads does not always produce equivalent throughput gains.

The cause depends on each construct. omp parallel for gives direct loop distribution and scales well initially, but eventually overhead and contention reduce marginal throughput. omp parallel for collapse(2) increases available iteration chunks by collapsing nested loops, which can improve distribution at some thread counts, but that extra scheduling complexity can also limit gains later. omp for simd raises per-thread arithmetic density through vector instructions, often helping early throughput, yet total GFLOPS still plateaus when team-level overhead dominates.

The implication is that omp for simd can improve computation quality inside each thread, but overall scalability still depends on OpenMP coordination behavior across the entire thread team.

### 5.5 IPC vs Threads
![](plots_ex2/ipc_vs_threads.png)

The IPC plot shows that omp parallel for, omp parallel for collapse(2), and omp for simd do not keep increasing instruction efficiency as threads grow. In higher-thread regions, IPC behavior is flatter or weaker, which matches the slowdown in speedup growth.

This happens because high thread counts increase synchronization and runtime coordination, so cores spend more cycles in less productive states. omp for simd can help each thread execute arithmetic more efficiently, but team-level overhead and shared-resource pressure still reduce IPC improvement at scale. omp parallel for collapse(2) may improve load splitting, yet it can also add enough scheduling cost to limit IPC gains.

The implication is that IPC confirms what the scaling plots already suggest: more threads can increase total work done, but they do not guarantee better per-cycle productivity.

### 5.6 Advanced Scaling Insights
![](plots_ex2/user_time_vs_threads.png)

![](plots_ex2/user_cpu_ratio_vs_threads.png)

![](plots_ex2/gflops_per_thread_vs_threads.png)

![](plots_ex2/incremental_speedup_gain_vs_threads.png)

![](plots_ex2/incremental_time_reduction_pct_vs_threads.png)

![](plots_ex2/elapsed_vs_user_time_scatter.png)

These advanced plots make one trend very clear for omp parallel for, omp parallel for collapse(2), and omp for simd: total CPU effort grows faster than practical speed benefit in the high-thread region. user_time_vs_threads and user_cpu_ratio_vs_threads show that we spend much more aggregate user CPU time as threads increase, even when elapsed-time improvements are already getting small.

The detailed cause appears in the marginal metrics. incremental_speedup_gain_vs_threads and incremental_time_reduction_pct_vs_threads show that each new thread block gives less extra acceleration than the previous one. gflops_per_thread_vs_threads tells the same story from a per-thread angle: as the team gets larger, each thread contributes less throughput because coordination and contention eat a larger part of runtime. elapsed_vs_user_time_scatter summarizes this trade-off visually, showing that better elapsed time requires disproportionately higher total CPU effort in the saturated range.

The implication is that the useful scaling limit should be chosen where marginal gain is still strong. After that point, omp parallel for, omp parallel for collapse(2), and omp for simd may still improve runtime slightly, but with poor cost-effectiveness per added thread.

### 5.7 Strategy Comparison
![](plots_ex2/speedup_vs_ipc_scatter.png)

The speedup-versus-IPC scatter shows that omp parallel for, omp parallel for collapse(2), and omp for simd occupy different trade-off regions instead of collapsing into one common trend. That means each construct combines instruction-level efficiency and thread-level scaling in a different way.

The reason is tied to directive behavior. omp parallel for is a strong baseline because it keeps launch and work-sharing overhead relatively low. omp parallel for collapse(2) can improve work distribution by expanding the iteration space, but this can come with extra scheduling overhead. omp for simd improves per-thread arithmetic efficiency through vectorization, but global speedup still depends on synchronization and contention across the full OpenMP team.

The implication is that the best method is not just the one with high IPC or high speedup alone, but the one that keeps both high at the same time across the useful thread range. Where points are close, conclusions should stay cautious.

## 6. Discussion
Exercise 1 shows that OnMult - omp parallel for and OnMultLine - parallel are more effective than OnMult - omp parallel + omp for under the same 4-thread setup. The key reason is not algorithm name, but directive behavior: OnMult - omp parallel + omp for pays more coordination and synchronization overhead before useful work is completed.

Exercise 2 shows that omp parallel for, omp parallel for collapse(2), and omp for simd all scale at first and then move into diminishing returns. The important difference between them is how quickly marginal speedup drops and how much extra CPU effort is required for small runtime gains.

The practical lesson is straightforward: use the construct that keeps useful work high and overhead low, and choose thread count based on incremental benefit, not only on maximum available threads.

## 7. Conclusion
In this project, OnMult - omp parallel for and OnMultLine - parallel perform better than OnMult - omp parallel + omp for in Exercise 1 because they spend less runtime in OpenMP overhead.  
In Exercise 2, omp parallel for, omp parallel for collapse(2), and omp for simd all improve performance early, but all of them hit diminishing returns as thread coordination and contention increase.  
collapse(2) and simd can help in specific regions, but they do not remove saturation at high thread counts.  
The best practical configuration is the one that keeps the best useful-work-to-overhead balance and still gives strong marginal gains when threads are added.
