# 纯净 ViT 追踪基线（pure_siamese_track）

Jetson Orin Nano Super + 双 IMX477（长焦追踪 GOOD LUCK 红箱）的单目标追踪基线。
零补丁重写，"找目标 + 跟目标" 两部件清晰分工。

## 架构

| 部件 | 用什么 | 干什么 |
|---|---|---|
| **找目标** | SIFT（多 ref 匹配）| SEARCH 态：找目标在哪 → init tracker |
| **跟目标** | **ViT 纯 TRT**（Transformer）| TRACK 态：逐帧跟 + getTrackingScore 判存在 |

```
SIFT 找到(双门槛) → init ViT → ViT 逐帧跟 → 丢失/漂移 → 回 SEARCH 重找
```

## 追踪器选型（合成考题验证，GT 已知）

| tracker | 速度 | 平均 IoU | 晃动段 IoU | 丢失率 |
|---|---|---|---|---|
| DaSiamRPN TRT | 11ms | 0.81 | 0.59 | 6% |
| **ViT 纯 TRT** | **5.8ms** | **0.92** | **0.92** | **0%** |
| NanoTrack | 17ms | 0.70 | 0.03 | 20% |

**ViT 纯 TRT 全面最优** —— 精度碾压（尤其抗晃动），速度还最快。
`int8_quantize/int_vittrack.py` 干掉了 cv2.dnn 的 30ms 框架开销（纯推理 0.58ms），
后处理精确复现 OpenCV TrackerVit，与 cv2 输出对齐到 0px。

## 防错锁防护链（逐层叠加，宁可漏锁不锁错）

| 环节 | 防什么 | 参数 |
|---|---|---|
| SIFT 双门槛 | 锁错背景/异物 | inliers≥30 + 颜色 sim≥0.25 |
| inlier bbox 框收紧 | 框外推带背景 | 用匹配点范围,非 ref 四角投影 |
| TRACK 颜色检查 | 漂到深色物体 | 每 10 帧 sim<0.15 → 重找 |
| max-box-ratio | tracker 全屏大框 | 框>画面 0.5 → 判丢失重找 |
| score 迟滞 | 锁定状态闪烁 | 进 0.6 / 保持 0.5 |

注：bbox 平滑/震动稳定**不在这层做**，交给飞控。这层只输出准确 bbox。

## 关键文件

- `pure_siamese_track.py` — 主程序（多线程：异步取帧/录像/SIFT）
- `int8_quantize/int_vittrack.py` — ViT 纯 TRT 封装（drop-in cv2.TrackerVit）
- `int8_quantize/int_dasiamrpn.py` — DaSiamRPN TRT 封装
- `synth_bench/eval_trackers.py` — 合成考题三方对比（DaSiamRPN/ViT/Nano）
- `synth_bench/eval_cv_vs_trt.py` — cv FP32 / TRT FP16 / INT8 对比

## 运行

```bash
python3 pure_siamese_track.py --sensor-id 1 \
  --ref-dir testdata/refs_tele_real --tracker vit-trt
# 桌面快捷方式: 8-纯净Siamese基线
```

tracker 可选：`vit-trt`(推荐) / `trt`(DaSiamRPN TRT) / `cv` / `vit`(cv2.dnn)

## 模型文件（不在仓库，体积大需本地 / trtexec 重建）

- `models/vittrack.onnx` → `trtexec --onnx=... --fp16 --saveEngine=models/vittrack_fp16.trt`
- `int8_quantize/dasiamrpn_*.trt`（DaSiamRPN TRT engine）
- 测试数据 `algo_limit_tests/`、备份 `releases/` 均不入库

## 性能（IMX477 1080p）

- 追踪器 5.8ms，主循环目标 ~60fps（摄像头 1080p 支持 60fps）
- 端到端 sensor→bbox ≈ s2r(~15ms) + 主循环
