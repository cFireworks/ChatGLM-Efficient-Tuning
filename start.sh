#!/bin/bash
cd /mnt/workspace/ChatGLM-Efficient-Tuning
if [ -d "$CHECKPOINT_DIR" ]; then
nohup python ./src/infer.py --checkpoint_dir $CHECKPOINT_DIR > infer.out &
echo "执行命令1"
else
nohup python ./src/infer.py > infer.out &
echo "执行命令2"
fi
python ./api.py
