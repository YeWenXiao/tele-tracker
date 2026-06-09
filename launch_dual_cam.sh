#!/bin/bash
# V2.2 双摄:长焦主追踪 + 广角辅助(Step1 视野框)
cd /home/nvidia/dual_cam_tracker
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority
echo "=================================================="
echo "  双摄 Step1: 广角主画面 + 长焦视野框(auto-calib)"
echo "  Q=退出"
echo "=================================================="
python3 dual_cam_track.py 2>&1 | tee /tmp/dual_cam_last.log
echo "按回车关闭..."
read
