#!/usr/bin/env python3
"""考题仿真:TRT FP16 vs cv FP32 siamese 同题对比。
红箱(真实物体,siamese 分布内)按已知轨迹运动,GT 已知 → 对比两 tracker 的准确度/速度/一致性。"""
import cv2, numpy as np, time, sys, os
sys.path.insert(0, 'int8_quantize')

box = cv2.imread('testdata/refs_tele_real/box_08.jpg')
bh, bw = box.shape[:2]
H, W = 1080, 1920
rng = np.random.RandomState(7)

# ── 出题:红箱平移+缩放+一段快速运动(模拟晃动),记录 GT ──
N = 150
frames, gts = [], []
for i in range(N):
    t = i / N
    bg = rng.randint(70, 150, (H, W, 3), np.uint8)            # 随机纹理背景
    # 轨迹:正弦平移 + 远近缩放;第 60-90 帧加快速移动(模拟晃动)
    fast = 2.5 if 60 <= i < 90 else 1.0
    cx = int(W/2 + 500 * np.sin(t * 2 * np.pi * fast))
    cy = int(H/2 + 250 * np.cos(t * 2 * np.pi * fast))
    scale = 0.6 + 0.5 * abs(np.sin(t * np.pi))               # 0.6~1.1 远近
    w, h = int(bw * scale), int(bh * scale)
    obj = cv2.resize(box, (w, h))
    if 60 <= i < 90:                                          # 晃动段加运动模糊
        obj = cv2.GaussianBlur(obj, (9, 9), 0)
    x0, y0 = cx - w // 2, cy - h // 2
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xb > xa and yb > ya:
        bg[ya:yb, xa:xb] = obj[ya - y0:yb - y0, xa - x0:xb - x0]
    frames.append(bg)
    gts.append((x0, y0, w, h))


def make_cv():
    p = cv2.TrackerDaSiamRPN_Params()
    p.model = "models/dasiamrpn_model.onnx"
    p.kernel_cls1 = "models/dasiamrpn_kernel_cls1.onnx"
    p.kernel_r1 = "models/dasiamrpn_kernel_r1.onnx"
    p.backend = cv2.dnn.DNN_BACKEND_CUDA; p.target = cv2.dnn.DNN_TARGET_CUDA
    return cv2.TrackerDaSiamRPN.create(p)


def run(tracker):
    tracker.init(frames[0], gts[0])
    bbs, ts = [], []
    for f in frames[1:]:
        t0 = time.perf_counter()
        ok, b = tracker.update(f)
        ts.append((time.perf_counter() - t0) * 1000)
        bbs.append(tuple(int(v) for v in b))
    return bbs, sum(ts) / len(ts)


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw_, bh_ = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw_), min(ay + ah, by + bh_)
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw_ * bh_ - inter)


def cerr(a, b):
    return ((a[0]+a[2]/2-b[0]-b[2]/2)**2 + (a[1]+a[3]/2-b[1]-b[3]/2)**2) ** 0.5


from int_dasiamrpn import IntDaSiamRPN
gN = gts[1:]
INT8_ENG = 'int8_quantize/dasiamrpn_model_271_dynkern_int8.trt'
sh = list(range(58, 88))   # 晃动段


def score(bb):
    return dict(iou=np.mean([iou(bb[i], gN[i]) for i in range(len(bb))]),
                ce=np.mean([cerr(bb[i], gN[i]) for i in range(len(bb))]),
                shake=np.mean([iou(bb[i], gN[i]) for i in sh]))


res = {}
for nm, mk in [('cv FP32', make_cv),
               ('TRT FP16', lambda: IntDaSiamRPN()),
               ('TRT INT8', lambda: IntDaSiamRPN(model_main_path=INT8_ENG))]:
    bb, ms = run(mk())
    s = score(bb); s['bb'] = bb; s['ms'] = ms
    res[nm] = s
names = list(res)
cv_bb = res['cv FP32']['bb']; trt_bb = res['TRT FP16']['bb']; int8_bb = res['TRT INT8']['bb']

print("=" * 64)
print("考题:红箱 150 帧,平移+缩放+晃动段(60-90帧快速+模糊)")
print("=" * 64)
print(f"{'指标':<18}" + "".join(f"{n:>15}" for n in names))
print(f"{'速度 ms/帧':<20}" + "".join(f"{res[n]['ms']:>15.1f}" for n in names))
print(f"{'  fps 上限':<20}" + "".join(f"{1000/res[n]['ms']:>15.0f}" for n in names))
print(f"{'平均 IoU(vsGT)':<17}" + "".join(f"{res[n]['iou']:>15.3f}" for n in names))
print(f"{'中心误差 px':<19}" + "".join(f"{res[n]['ce']:>15.1f}" for n in names))
print(f"{'晃动段 IoU':<20}" + "".join(f"{res[n]['shake']:>15.3f}" for n in names))
print("-" * 64)
for n in names[1:]:
    ci = np.mean([iou(cv_bb[i], res[n]['bb'][i]) for i in range(len(cv_bb))])
    print(f"{n} vs cv 一致性 IoU={ci:.3f}")
print(f"速度: FP16 比 cv 快 {res['cv FP32']['ms']/res['TRT FP16']['ms']:.1f}x | "
      f"INT8 比 FP16 快 {res['TRT FP16']['ms']/res['TRT INT8']['ms']:.2f}x")
print("=" * 64)

# 可视化对比视频:GT(白)/cv(绿)/TRT(蓝)
os.makedirs('synth_bench/cv_vs_trt', exist_ok=True)
vw = cv2.VideoWriter('synth_bench/cv_vs_trt/compare.avi',
                     cv2.VideoWriter_fourcc(*'MJPG'), 20, (W // 2, H // 2))
for i in range(len(cv_bb)):
    f = frames[i + 1].copy()
    for bb, col, lb in [(gN[i], (255, 255, 255), 'GT'), (cv_bb[i], (0, 255, 0), 'cv'),
                        (trt_bb[i], (255, 150, 0), 'TRT')]:
        x, y, w, h = bb
        cv2.rectangle(f, (x, y), (x + w, y + h), col, 3)
    cv2.putText(f, f"WHITE=GT GREEN=cv BLUE=TRT  f{i}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    vw.write(cv2.resize(f, (W // 2, H // 2)))
vw.release()
print("可视化: synth_bench/cv_vs_trt/compare.avi (白=GT 绿=cv 蓝=TRT)")
