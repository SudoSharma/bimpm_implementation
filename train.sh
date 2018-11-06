#!/bin/bash
# run train script, store stdout and stderr in output file and capture PID

# default
if [ ! "$1" ]
    nohup python train.py > train.out 2>&1 &
    echo $! > train_pid.txt
elif [ "$1" ==  --00 ] ; then  # xp_0.0
    nohup python train.py -experiment 'xp_0.0' -s > ./research/output/xp_0.0_output.txt 2>&1 &
    echo $! > ./research/output/xp_0.0_pid.txt
elif [ "$1" ==  --01 ] ; then  # xp_0.1
    nohup python train.py -s > ./research/output/xp_0.1_output.txt 2>&1 &
    echo $! > ./research/output/xp_0.1_pid.txt
fi
