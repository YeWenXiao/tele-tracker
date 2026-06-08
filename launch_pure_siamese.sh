#!/bin/bash
# 纯净 Siamese 追踪基线(零补丁)
cd /home/nvidia/dual_cam_tracker
export DISPLAY=:0
export XAUTHORITY=/home/nvidia/.Xauthority
echo "=================================================="
echo "  纯净 Siamese 基线 (零补丁,不依赖 t5)"
echo "  ref SIFT 找一次 → 纯 DaSiamRPN 跟踪 + score 检测消失"
echo "  S=重新搜索 | Q=退出"
echo "=================================================="
python3 pure_siamese_track.py --sensor-id 1 --ref-dir testdata/refs_tele_real --score-min 0.6 --tracker vit-trt 2>&1 | tee /tmp/pure_siamese_last.log
echo "按回车关闭..."
read
