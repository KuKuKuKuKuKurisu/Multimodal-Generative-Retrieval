#!/bin/bash
deepspeed --include localhost:0,1,2,3 stage1_main.py --deepspeed --deepspeed_config deepspeed_config.json