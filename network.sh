#!/bin/bash

JOB=`python3 /workspace/job_name.py`;
IP=`ifconfig eth0 | grep 'inet ' | awk '{print $2}'`;

MODEL=`cat /workspace/model.txt`;

tcpdump host ${IP} -s 64 -w /result/${MODEL}/${JOB}_${IP}_network.pcap &

