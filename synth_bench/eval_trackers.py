#!/usr/bin/env python3
"""考题仿真:DaSiamRPN(2018) vs ViT(Transformer 2021+) vs NanoTrack 三方对比。
同一套红箱考题(平移+缩放+晃动),GT 已知 → 看新模型有没有更强判别力/抗漂移。"""
import cv2, numpy as np, time, os

box = cv2.imread('testdata/refs_tele_real/box_08.jpg')
bh, bw = box.shape[:2]
H, W = 1080, 1920
rng = np.random.RandomState(7)

# ── 出题(与 cv_vs_trt 同 seed,可比)──
N = 150
frames, gts = [], []
for i in range(N):
    t = i / N
    bg = rng.randint(70, 150, (H, W, 3), np.uint8)
    fast = 2.5 if 60 <= i < 90 else 1.0
    cx = int(W/2 + 500 * np.sin(t * 2 * np.pi * fast))
    cy = int(H/2 + 250 * np.cos(t * 2 * np.pi * fast))
    scale = 0.6 + 0.5 * abs(np.sin(t * np.pi))
    w, h = int(bw * scale), int(bh * scale)
    obj = cv2.resize(box, (w, h))
    if 60 <= i < 90:
        obj = cv2.GaussianBlur(obj, (9, 9), 0)
    x0, y0 = cx - w // 2, cy - h // 2
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb > xa and yb > ya:
        bg[ya:yb, xa:xb] = obj[ya - y0:yb - y0, xa - x0:xb - x0]
    frames.append(bg); gts.append((x0, y0, w, h))


def mk_dasiam():
    p = cv2.TrackerDaSiamRPN_Params()
    p.model = "models/dasiamrpn_model.onnx"; p.kernel_cls1 = "models/dasiamrpn_kernel_cls1.onnx"
    p.kernel_r1 = "models/dasiamrpn_kernel_r1.onnx"
    p.backend = cv2.dnn.DNN_BACKEND_CUDA; p.target = cv2.dnn.DNN_TARGET_CUDA
    return cv2.TrackerDaSiamRPN.create(p)


def mk_vit():
    p = cv2.TrackerVit_Params()
    p.net = "models/vittrack.onnx"
    p.backend = cv2.dnn.DNN_BACKEND_CUDA; p.target = cv2.dnn.DNN_TARGET_CUDA
    return cv2.TrackerVit.create(p)


def mk_nano():
    p = cv2.TrackerNano_Params()
    p.backbone = "models/nanotrack_backbone.onnx"; p.neckhead = "models/nanotrack_head.onnx"
    p.backend = cv2.dnn.DNN_BACKEND_CUDA; p.target = cv2.dnn.DNN_TARGET_CUDA
    return cv2.TrackerNano.create(p)


def run(tracker):
    tracker.init(frames[0], gts[0]); bbs, ts = [], []
    for f in frames[1:]:
        t0 = time.perf_counter(); ok, b = tracker.update(f); ts.append((time.perf_counter() - t0) * 1000)
        bbs.append(tuple(int(v) for v in b))
    return bbs, sum(ts) / len(ts)


def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw_, bh_ = b
    x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax + aw, bx + bw_), min(ay + ah, by + bh_)
    if x2 <= x1 or y2 <= y1: return 0.0
    it = (x2 - x1) * (y2 - y1)
    return it / (aw * ah + bw_ * bh_ - it)


def cerr(a, b):
    return ((a[0]+a[2]/2-b[0]-b[2]/2)**2 + (a[1]+a[3]/2-b[1]-b[3]/2)**2) ** 0.5


gN = gts[1:]; sh = list(range(58, 88))
res = {}
for nm, mk in [('DaSiamRPN(2018)', mk_dasiam), ('ViT(Transformer)', mk_vit), ('NanoTrack', mk_nano)]:
    try:
        bb, ms = run(mk())
        res[nm] = dict(
            iou=np.mean([iou(bb[i], gN[i]) for i in range(len(bb))]),
            ce=np.mean([cerr(bb[i], gN[i]) for i in range(len(bb))]),
            shake=np.mean([iou(bb[i], gN[i]) for i in sh]),
            lost=sum(1 for i in range(len(bb)) if iou(bb[i], gN[i]) < 0.1) * 100 // len(bb),
            ms=ms, bb=bb)
    except Exception as e:
        print(f"[{nm}] 失败: {e}")

print("=" * 70)
print("考题:红箱 150 帧,平移+缩放+晃动段(60-90帧快速+模糊)")
print("=" * 70)
names = list(res)
print(f"{'指标':<16}" + "".join(f"{n:>18}" for n in names))
print(f"{'速度 ms/帧':<18}" + "".join(f"{res[n]['ms']:>18.1f}" for n in names))
print(f"{'  fps 上限':<18}" + "".join(f"{1000/res[n]['ms']:>18.0f}" for n in names))
print(f"{'平均 IoU↑':<17}" + "".join(f"{res[n]['iou']:>18.3f}" for n in names))
print(f"{'中心误差px↓':<16}" + "".join(f"{res[n]['ce']:>18.1f}" for n in names))
print(f"{'晃动段 IoU↑':<16}" + "".join(f"{res[n]['shake']:>18.3f}" for n in names))
print(f"{'丢失率%↓':<17}" + "".join(f"{res[n]['lost']:>18d}" for n in names))
print("=" * 70)

# 可视化:GT白 / DaSiam绿 / ViT蓝 / Nano紫
os.makedirs('synth_bench/trackers', exist_ok=True)
vw = cv2.VideoWriter('synth_bench/trackers/compare.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (W//2, H//2))
cols = {'DaSiamRPN(2018)': (0,255,0), 'ViT(Transformer)': (255,150,0), 'NanoTrack': (255,0,255)}
for i in range(len(gN)):
    f = frames[i+1].copy()
    cv2.rectangle(f, gN[i][:2], (gN[i][0]+gN[i][2], gN[i][1]+gN[i][3]), (255,255,255), 2)
    for n in names:
        x,y,w,h = res[n]['bb'][i]
        cv2.rectangle(f, (x,y), (x+w,y+h), cols.get(n,(0,0,255)), 3)
    cv2.putText(f, f"WHITE=GT GREEN=DaSiam BLUE=ViT MAGENTA=Nano f{i}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
    vw.write(cv2.resize(f, (W//2, H//2)))
vw.release()
print("可视化: synth_bench/trackers/compare.avi")
