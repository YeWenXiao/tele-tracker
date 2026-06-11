#!/bin/bash
# 广角场景归航 现场版(参考图=沿途整张照片,覆盖率切换)
cd /home/nvidia/dual_cam_tracker
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority
echo "=================================================="
echo "  广角归航: 照片序列 1..10 远→近 | 红十字=瞄准点"
echo "  cov 跌破 0.5 连续确认后自动切下一张 | Q 退出"
echo "=================================================="
python3 wide_homing_field.py 2>&1 | tee /tmp/homing_last.log
echo "按回车关闭..."
read
