#!/bin/bash

EXE=../build/tests/gemm

# 50^3 -- 1000^3
# n m k
# V = n*m*k = const in [50^3 .. 1000^3]
# n = m = k/alpha
#
# n = m = (V/alpha)^(1/3)
# k = n*alpha

asymm=(1 8 125 1000)
cbrt=(1 2 5 10)
echo -n "# N"
for((i=0;i<${#asymm[*]};i++)); do
    alpha=${asymm[i]}
    rt=${cbrt[i]}
    echo -n " ${alpha}"
done
echo

for L0 in 160 320 640 1280 2560 5120 10240; do
    echo -n "$L0"
    for((i=0;i<${#asymm[*]};i++)); do
        # Note: $((n*n*k)) remains constant during this loop
        alpha=${asymm[i]}
        rt=${cbrt[i]}
        n=$((L0/rt))
        k=$((n*alpha))
        sz="$($EXE $n $n $k | sed -n -e 's/GFLOPS://p')"
        echo -n " $sz"
    done
    echo
done
