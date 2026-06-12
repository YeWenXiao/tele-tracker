#!/bin/bash
# DJI 实时图传归航(Mini 4 Pro: DJI Fly → 直播 → RTMP)
cd /home/nvidia/dual_cam_tracker
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority
echo "=================================================="
echo "  DJI 实时归航"
echo "  1. 确保手机和 Jetson 在同一 WiFi"
echo "  2. DJI Fly → 设置 → 图传 → 直播 → RTMP"
echo "     地址填终端显示的 rtmp://...:1935/live"
echo "  3. 开始直播后画面自动出现 | Q 退出"
echo "=================================================="
python3 rtmp_homing_live.py 2>&1 | tee /tmp/rtmp_homing_last.log
echo "按回车关闭..."
read
