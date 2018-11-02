#!/bin/bash
# Run train script, store stdout and stderr in output file and capture PID
# This is a baseline run
nohup python train.py -s > xp_0.0_output.txt 2>&1 &
echo $! > xp_0.0_pid.txt
