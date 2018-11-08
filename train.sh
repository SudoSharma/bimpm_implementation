#!/bin/bash
# run train script, store stdout and stderr in output file and capture PID
if [ "$1" == xp ]; then  # Run current experiment
    nohup python train.py -char-hidden-size 100 -s -e 1.1 > ./research/output/xp_1.1_output.txt 2>&1 &
    echo $! > ./research/output/xp_1.1_pid.txt
else  # Default
    nohup python train.py > train.out 2>&1 &
    echo $! > train_pid.txt
fi
