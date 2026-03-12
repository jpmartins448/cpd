
#!/bin/bash
 
BINARY="./assignement_1"
 
SIZES=(1024 1536 2048 2560 3072 4096 6144 8192 10240)
BLOCK_SIZES=(128 256 512)
 
OUTPUT_FILE="perf_results_onmultblock.txt"
 
> "$OUTPUT_FILE"  # Clear/create the output file
 
for SIZE in "${SIZES[@]}"; do
    for BKSIZE in "${BLOCK_SIZES[@]}"; do
        echo "======================================" | tee -a "$OUTPUT_FILE"
        echo "Matrix size: ${SIZE}x${SIZE}, Block size: ${BKSIZE}" | tee -a "$OUTPUT_FILE"
        echo "======================================" | tee -a "$OUTPUT_FILE"
 
        perf stat -e \
            cycles,instructions,\
cache-references,cache-misses,\
L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses,\
stalled-cycles-backend,\
mem_load_retired.l1_miss,\
mem_load_retired.l2_miss \
            "$BINARY" "$SIZE" "$BKSIZE" 2>&1 | tee -a "$OUTPUT_FILE"
 
        echo "" | tee -a "$OUTPUT_FILE"
    done
done
 
echo "Done. Results saved to $OUTPUT_FILE"