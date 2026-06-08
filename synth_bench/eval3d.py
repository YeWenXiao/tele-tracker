#!/usr/bin/env python3
"""eval3d.py — 2D ref 在 3D 视差视频里的重定位诊断

用 ref_front.png(2D 正面模板)在 3D 转动视频每帧定位目标,跟 GT bbox 比 IoU。
输出「视角 az vs 检出率」「尺度 dist vs 检出率」曲线 → 回答各算法在视差/尺度下崩在哪。

SIFT:feature match + homography 投影定位
NCC :多尺度 matchTemplate 定位
(siamese 全图重定位需滑窗,留作下一步)
"""
import os, sys, csv, math, json
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

IOU_HIT = 0.30   # IoU > 此值算检出


def classify(box, regions):
    """算法定位框中心落在哪:目标区=detect / 干扰区=mislock(错锁)/ 都不=miss。"""
    if box is None:
        return 'miss'
    cx = (box[0] + box[2]) / 2; cy = (box[1] + box[3]) / 2
    for (x0, y0, x1, y1, is_t) in regions:        # 先判目标(重叠时目标优先)
        if is_t and x0 <= cx <= x1 and y0 <= cy <= y1:
            return 'detect'
    for (x0, y0, x1, y1, is_t) in regions:
        if not is_t and x0 <= cx <= x1 and y0 <= cy <= y1:
            return 'mislock'
    return 'miss'


def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


class SIFTLocator:
    name = 'SIFT'
    def __init__(self, ref):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.bf = cv2.BFMatcher()
        self.gref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        self.kr, self.dr = self.sift.detectAndCompute(self.gref, None)
        h, w = self.gref.shape
        self.corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    def locate(self, frame):
        gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kf, df = self.sift.detectAndCompute(gf, None)
        if df is None or self.dr is None or len(df) < 4:
            return None, 0
        good = []
        for pair in self.bf.knnMatch(self.dr, df, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 4:
            return None, len(good)
        src = np.float32([self.kr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None, len(good)
        inliers = int(mask.sum())
        proj = cv2.perspectiveTransform(self.corners, H).reshape(-1, 2)
        x0, y0 = proj.min(0); x1, y1 = proj.max(0)
        return (float(x0), float(y0), float(x1), float(y1)), inliers


class NCCLocator:
    name = 'NCC'
    def __init__(self, ref):
        self.gref0 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    def locate(self, frame):
        gf = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best = (-1, None)
        for sc in [0.4, 0.55, 0.7, 0.85, 1.0]:
            t = cv2.resize(self.gref0, None, fx=sc, fy=sc)
            th, tw = t.shape
            if th >= gf.shape[0] or tw >= gf.shape[1]:
                continue
            res = cv2.matchTemplate(gf, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, (mloc[0], mloc[1], tw, th))
        if best[1] is None:
            return None, 0.0
        x, y, w, h = best[1]
        return (float(x), float(y), float(x + w), float(y + h)), float(best[0])


class SiameseEmbLocator:
    """siamese backbone embedding 全图滑窗重定位(每帧独立,无连续加成,跟 NCC 公平)"""
    name = 'Siam-emb'
    def __init__(self, ref, stride=50, win=130):
        ip = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'int8_quantize')
        sys.path.insert(0, ip)
        from int_dasiamrpn import (IntDaSiamRPN, EXEMPLAR_SIZE,
                                   CONTEXT_AMOUNT, _CTX_LOCK)
        self.t = IntDaSiamRPN()
        self.E = EXEMPLAR_SIZE; self.CA = CONTEXT_AMOUNT; self.LOCK = _CTX_LOCK
        self.stride = stride; self.win = win
        self.ref_feat = self._embed(ref)
        self.rn = np.linalg.norm(self.ref_feat)
    def _embed(self, img):
        h, w = img.shape[:2]
        pos = np.array([w / 2.0, h / 2.0], np.float32)
        wc = w + self.CA * (w + h); hc = h + self.CA * (w + h)
        sz = float(np.rint(np.sqrt(wc * hc)))
        avg = img.reshape(-1, 3).mean(0).astype(np.float32)
        z = self.t._get_subwindow_square(img, pos, sz, self.E, avg)
        blob = z.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
        with self.LOCK:
            self.t._ctx.push()
            try:
                feat = self.t.eng_template.infer_one_input(blob)[0]
            finally:
                self.t._ctx.pop()
        return feat.flatten().astype(np.float64)
    def locate(self, frame):
        H, W = frame.shape[:2]
        best = (-1.0, None)
        for y in range(0, H - self.win + 1, self.stride):
            for x in range(0, W - self.win + 1, self.stride):
                f = self._embed(frame[y:y + self.win, x:x + self.win])
                cos = float(np.dot(f, self.ref_feat) /
                            (np.linalg.norm(f) * self.rn + 1e-9))
                if cos > best[0]:
                    best = (cos, (x, y, self.win, self.win))
        if best[1] is None:
            return None, 0.0
        x, y, w, h = best[1]
        return (float(x), float(y), float(x + w), float(y + h)), best[0]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'synth_bench/out3d_conf'
    print(f"评测场景: {src}")
    ref = cv2.imread(os.path.join(src, 'ref_front.png'))
    gt = list(csv.DictReader(open(os.path.join(src, 'ground_truth.csv'))))
    out_dir = 'synth_bench/eval3d'; os.makedirs(out_dir, exist_ok=True)

    # pass 1: SIFT/NCC 每帧独立用 2D ref 重定位
    locs = [SIFTLocator(ref), NCCLocator(ref)]
    recs = {l.name: [] for l in locs}   # (az, dist, iou, conf, hit)
    cap = cv2.VideoCapture(os.path.join(src, 'video.avi'))
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi >= len(gt):
            break
        g = gt[fi]
        gt_box = (float(g['bbox_x0']), float(g['bbox_y0']),
                  float(g['bbox_x1']), float(g['bbox_y1']))
        az = abs(float(g['az_deg'])); dist = float(g['dist'])
        regions = json.loads(g.get('boxes_json', '[]') or '[]')
        for l in locs:
            box, conf = l.locate(frame)
            i = iou(box, gt_box) if box else 0.0
            cls = classify(box, regions) if regions else ('detect' if i > IOU_HIT else 'miss')
            recs[l.name].append((az, dist, i, conf, int(i > IOU_HIT), cls))
        fi += 1
    cap.release()
    print(f"评测 {fi} 帧,ref=2D 正面,视频=3D 转动+干扰")

    # pass 2: Siamese 连续跟踪(本职用法:第一帧正面 init,后续逐帧 update)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'int8_quantize'))
        from int_dasiamrpn import IntDaSiamRPN
        sm = IntDaSiamRPN()
        recs['Siamese'] = []
        cap = cv2.VideoCapture(os.path.join(src, 'video.avi'))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok or fi >= len(gt):
                break
            g = gt[fi]
            gt_box = (float(g['bbox_x0']), float(g['bbox_y0']),
                      float(g['bbox_x1']), float(g['bbox_y1']))
            az = abs(float(g['az_deg'])); dist = float(g['dist'])
            if fi == 0:
                x0, y0, x1, y1 = gt_box
                sm.init(frame, (x0, y0, x1 - x0, y1 - y0))
                box, conf = gt_box, 1.0
            else:
                ok2, bb = sm.update(frame)
                box = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])
                conf = sm.getTrackingScore()
            i = iou(box, gt_box)
            regions = json.loads(g.get('boxes_json', '[]') or '[]')
            cls = classify(box, regions) if regions else ('detect' if i > IOU_HIT else 'miss')
            recs['Siamese'].append((az, dist, i, conf, int(i > IOU_HIT), cls))
            fi += 1
        cap.release()
        print(f"Siamese 连续跟踪 {fi} 帧(第一帧正面 init)")
    except Exception as e:
        print(f"Siamese 跳过: {e}")

    # pass 3: Siam-emb 全图滑窗独立重定位(跟 NCC 公平:都每帧从零找)
    try:
        sl = SiameseEmbLocator(ref)
        recs['Siam-emb'] = []
        cap = cv2.VideoCapture(os.path.join(src, 'video.avi'))
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok or fi >= len(gt):
                break
            g = gt[fi]
            gt_box = (float(g['bbox_x0']), float(g['bbox_y0']),
                      float(g['bbox_x1']), float(g['bbox_y1']))
            az = abs(float(g['az_deg'])); dist = float(g['dist'])
            box, conf = sl.locate(frame)
            i = iou(box, gt_box) if box else 0.0
            regions = json.loads(g.get('boxes_json', '[]') or '[]')
            cls = classify(box, regions) if regions else ('detect' if i > IOU_HIT else 'miss')
            recs['Siam-emb'].append((az, dist, i, conf, int(i > IOU_HIT), cls))
            fi += 1
        cap.release()
        print(f"Siam-emb 全图滑窗重定位 {fi} 帧")
    except Exception as e:
        print(f"Siam-emb 跳过: {e}")

    # CSV(含 cls 分类)
    with open(os.path.join(out_dir, 'eval3d.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'az', 'dist', 'iou', 'conf', 'hit', 'cls'])
        for name, rs in recs.items():
            for r in rs:
                w.writerow([name, f'{r[0]:.1f}', f'{r[1]:.2f}', f'{r[2]:.3f}',
                            f'{r[3]:.3f}', r[4], r[5]])

    # 核心指标:检出 / 错锁(锁到干扰)/ 漏检
    print("\n=== 检出 / 错锁 / 漏检 (干扰场景核心) ===")
    print(f'{"算法":16s} 检出   错锁   漏检')
    for name, rs in recs.items():
        n = len(rs)
        det = sum(1 for r in rs if r[5] == 'detect') / n
        mis = sum(1 for r in rs if r[5] == 'mislock') / n
        mss = sum(1 for r in rs if r[5] == 'miss') / n
        print(f'{name:16s} {det*100:4.0f}%  {mis*100:4.0f}%  {mss*100:4.0f}%')

    def binned_cls(rs, target, bins):
        arr = np.array([(r[0], 1.0 if r[5] == target else 0.0) for r in rs])
        out = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
            if m.sum() > 0:
                out.append(((lo + hi) / 2, arr[m, 1].mean()))
        return out

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    az_bins = np.linspace(0, 80, 9)
    colors = {'SIFT': 'tab:red', 'NCC': 'tab:green', 'Siamese': 'tab:blue',
              'Siam-emb': 'tab:purple'}
    for name, rs in recs.items():
        c = colors.get(name, 'gray')
        bd = binned_cls(rs, 'detect', az_bins)
        bm = binned_cls(rs, 'mislock', az_bins)
        a1.plot([x for x, _ in bd], [y for _, y in bd], 'o-', color=c, label=name)
        a2.plot([x for x, _ in bm], [y for _, y in bm], 'o-', color=c, label=name)
    a1.set_xlabel('|azimuth| deg'); a1.set_ylabel('detect rate')
    a1.set_title('视角 vs 检出率'); a1.legend(); a1.grid(alpha=0.3)
    a2.set_xlabel('|azimuth| deg'); a2.set_ylabel('mislock rate (锁到干扰)')
    a2.set_title('视角 vs 错锁率 (越低越好)'); a2.legend(); a2.grid(alpha=0.3)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    png = os.path.join(out_dir, 'eval3d_mislock.png')
    plt.savefig(png, dpi=120, bbox_inches='tight')
    print(f"\n[OUT] 曲线 → {png}")
    print(f"[OUT] 明细 → {out_dir}/eval3d.csv")


if __name__ == '__main__':
    main()
