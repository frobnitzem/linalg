#!/bin/bash
#BSUB -P <projid>
#BSUB -W 0:15
#BSUB -nnodes 2
#BSUB -q batch
#BSUB -J smat

jsrun --smpiargs="-gpu" -r6 -g1 -c7 -b packed:7 ./smat 4096 2048 12 2

