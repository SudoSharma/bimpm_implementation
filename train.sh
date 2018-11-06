#!/bin/bash
# run train script, store stdout and stderr in output file and capture PID

if [ "$1" == 00 ]; then  # xp_0.0
    nohup python train.py -t -e 0.0 > ./research/output/xp_0.0_output.txt 2>&1 &
    echo $! > ./research/output/xp_0.0_pid.txt
else  # Default
    nohup python train.py > train.out 2>&1 &
    echo $! > train_pid.txt
fi
