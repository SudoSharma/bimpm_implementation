#!/bin/bash
# run train script, store stdout and stderr in output file and capture PID
if [ "$1" == xp ]; then  # Run current experiment
    nohup python train.py -s -e 3.0 > ./research/output/xp_3.0_output.txt 2>&1 &
    echo $! > ./research/output/xp_3.0_pid.txt
else  # Default
    nohup python train.py -s > train.out 2>&1 &
    echo $! > train_pid.txt
fi
