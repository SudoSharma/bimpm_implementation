#!/bin/bash
# run train script, store stdout and stderr in output file and capture PID
nohup python train.py > train.out 2>&1 &
echo $! > train_pid.txt
