#!/bin/bash
# 手机传图到 Jetson(同一 WiFi,手机浏览器打开显示的地址)
cd /home/nvidia/dual_cam_tracker
python3 phone_upload_server.py
echo "按回车关闭..."
read
