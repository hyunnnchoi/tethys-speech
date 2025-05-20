#!/bin/bash

JOB=`python3 /workspace/job_name.py`;

MODEL=`cat /workspace/model.txt`;

/workspace/NVML/NVML > /result/${MODEL}/${JOB}_gpu.txt &

