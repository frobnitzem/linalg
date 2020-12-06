jsrun --smpiargs="-gpu" -n4 -g1 -c7 -b packed:7 ./smat $((10*4096)) $((2*4096)) 4096 2048
