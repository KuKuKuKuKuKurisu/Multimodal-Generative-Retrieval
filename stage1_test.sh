#!/bin/bash
deepspeed --include localhost:1,2,3 stage1_main_test.py --deepspeed --deepspeed_config deepspeed_config.json