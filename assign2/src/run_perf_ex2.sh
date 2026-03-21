#!/bin/bash
BINARY="./assignement_2"
SIZE=8192
THREADS=(4 8 12 16 20 24)
OUTPUT_FILE="perf_results_onmultline_ex2_pragma_omp_parallel_for_collapse(2).txt"
> "$OUTPUT_FILE"

EVENTS="cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,stalled-cycles-backend,mem_load_retired.l1_miss,mem_load_retired.l2_miss"

for T in "${THREADS[@]}"; do
    export OMP_NUM_THREADS=$T
    echo "======================================" | tee -a "$OUTPUT_FILE"
    echo "Method: OnMultLine | Size: ${SIZE}x${SIZE} | Threads: $T" | tee -a "$OUTPUT_FILE"
    echo "======================================" | tee -a "$OUTPUT_FILE"
    perf stat -e "$EVENTS" $BINARY 2 $SIZE 2>&1 | tee -a "$OUTPUT_FILE"
    echo "" | tee -a "$OUTPUT_FILE"
done

echo "Done. Results saved to $OUTPUT_FILE"