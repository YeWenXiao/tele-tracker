#!/bin/bash
# V2.2 双摄:长焦主追踪 + 广角视野框 + 失锁橡皮筋
# 远距配置(0610 现场验证):现场 ref + 全分辨率 SIFT + unique inliers 12
cd /home/nvidia/dual_cam_tracker
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority
echo "=================================================="
echo "  双摄: 广角(黄框=长焦视野) | 长焦主追踪"
echo "  远距模式: refs_field_0610 + sift-scale 1.0"
echo "  Q=退出"
echo "=================================================="
python3 dual_cam_track.py \
  --tele-ref testdata/refs_field_0610 \
  --tele-sift-scale 1.0 \
  --tele-min-inliers 12 \
  2>&1 | tee /tmp/dual_cam_last.log
echo "按回车关闭..."
read
