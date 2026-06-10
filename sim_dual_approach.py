#!/usr/bin/env python3
"""sim_dual_approach.py — 双镜头仿真:真实素材(现场背景+真箱子像素)模拟由远及近 + 横移出视野。
长焦 = 画布中心 1/3 原生裁切(3x);广角 = 整画布降采样。
跑真实管线(TrackPipeline×2 + FovLocator),输出左右并排标注视频 + 统计。
剧本: A 正中接近(锁定+阶梯) B 横移出长焦(LOST+橡皮筋) C 移回(找回) D 逼近(大ref)。
"""
import cv2, numpy as np, time, os
from dual_cam_track import TrackPipeline, FovLocator

W, H = 1920, 1080
CW, CH = W * 3, H * 3            # 世界画布 = 3x(长焦原生裁切不插值)
OUT = 'algo_limit_tests/sim_dual_approach.avi'

# ── 素材:现场照片当背景,真箱子像素当目标 ──
bg = cv2.imread('uploads/20260610/1.jpg')
bg = cv2.resize(bg, (CW, CH))
# 目标 = field_10(photo10 裁的);ref 库留一法排除 field_10 → 目标与 ref 像素异源
#(不同照片独立成像,同物体同光线 = 贴近实战;否则目标=参考图本身,自匹配作弊)
box_src = cv2.imread('testdata/refs_field_0610/field_10.jpg')
import shutil, glob
REF_DIR = 'testdata/refs_sim_loo'
shutil.rmtree(REF_DIR, ignore_errors=True); os.makedirs(REF_DIR)
for f in glob.glob('testdata/refs_field_0610/*.jpg'):
    if os.path.basename(f) != 'field_10.jpg':
        shutil.copy(f, REF_DIR)

rng = np.random.RandomState(11)
def degrade(obj, n):
    """成像退化:轻微模糊 + 亮度抖动 + 噪声(目标不是 ref 的完美副本)"""
    if n % 3 == 0:
        obj = cv2.GaussianBlur(obj, (3, 3), 0)
    gain = 0.9 + 0.2 * np.sin(n * 0.05)           # 亮度 ±10% 缓变
    obj = np.clip(obj.astype(np.float32) * gain +
                  rng.normal(0, 4, obj.shape), 0, 255).astype(np.uint8)
    return obj

N = 600
def scenario(n):
    """返回 (长焦内箱子尺寸px, 箱子横向偏移[画布px])"""
    if n < 200:        # A 正中接近 60→240
        return 60 + (240 - 60) * n / 199, 0
    if n < 320:        # B 横移出长焦(长焦半宽=960,移到1500远超出)
        t = (n - 200) / 119
        return 240, 1500 * t
    if n < 420:        # C 移回
        t = (n - 320) / 99
        return 240, 1500 * (1 - t)
    t = (n - 420) / 179            # D 逼近 240→520
    return 240 + (520 - 240) * t, 0

def render(n):
    s, dx = scenario(n)
    s = int(s)
    canvas = bg.copy()
    obj = degrade(cv2.resize(box_src, (s, s)), n)
    cx, cy = CW // 2 + int(dx), CH // 2 + 300    # 放画面中心偏下(像放地上)
    x0, y0 = cx - s // 2, cy - s // 2
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(CW, x0 + s), min(CH, y0 + s)
    if xb > xa and yb > ya:
        canvas[ya:yb, xa:xb] = obj[ya - y0:yb - y0, xa - x0:xb - x0]
    tele = canvas[CH // 2 - H // 2:CH // 2 + H // 2, CW // 2 - W // 2:CW // 2 + W // 2]
    wide = cv2.resize(canvas, (W, H))
    return wide, tele.copy()

# ── 真实管线 ──
tele = TrackPipeline('TELE', REF_DIR, W, H, sift_scale=1.0, min_inliers=12)
wide = TrackPipeline('WIDE', REF_DIR, W, H, sift_scale=1.0, min_inliers=10)
fovloc = FovLocator(1 / 3, W, H)

vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'MJPG'), 30, (W, H // 2))
stats = dict(first_lock=-1, lost_at=-1, relock=-1, rope=0, t_track=0, w_track=0)
t0 = time.time()
for n in range(N):
    wf, tf = render(n)
    t_state, t_bbox, t_score, t_locked = tele.step(tf)
    w_state, w_bbox, w_score, w_locked = wide.step(wf)
    bx, by, fw, fh, mconf = fovloc.update(wf, tf)
    if t_locked:
        stats['t_track'] += 1
        if stats['first_lock'] < 0:
            stats['first_lock'] = n
            print(f"[A] f{n} 长焦首锁")
        if stats['lost_at'] >= 0 and stats['relock'] < 0:
            stats['relock'] = n
            print(f"[C] f{n} 长焦找回(用 ref: {tele.sift.refs[tele.sift.last_idx][0]})")
    elif stats['first_lock'] >= 0 and stats['lost_at'] < 0 and n >= 200:
        stats['lost_at'] = n
        print(f"[B] f{n} 长焦丢失(目标出视野)")
    if w_locked:
        stats['w_track'] += 1

    # 标注(同 dual_cam 主循环)
    left = wf.copy()
    cx2, cy2 = bx + fw // 2, by + fh // 2
    cv2.rectangle(left, (bx, by), (bx + fw, by + fh), (0, 255, 255), 4)
    if w_locked and w_bbox:
        x, y, ww_, wh_ = w_bbox
        cv2.rectangle(left, (x, y), (x + ww_, y + wh_), (0, 255, 0), 2)
    if not t_locked and w_locked and w_bbox:
        tgt = (w_bbox[0] + w_bbox[2] // 2, w_bbox[1] + w_bbox[3] // 2)
        cv2.line(left, (cx2, cy2), tgt, (0, 0, 255), 5)
        cv2.circle(left, tgt, 12, (0, 0, 255), -1)
        stats['rope'] += 1
    cv2.putText(left, f"WIDE {w_state} m={mconf:.2f}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    right = tf.copy()
    if t_locked and t_bbox:
        x, y, ww_, wh_ = t_bbox
        cv2.rectangle(right, (x, y), (x + ww_, y + wh_), (0, 255, 0), 3)
    cv2.putText(right, f"TELE {t_state} s={t_score:.2f}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
    combo = np.hstack([left, right])
    cv2.putText(combo, f"f{n}", (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    vw.write(cv2.resize(combo, (W, H // 2)))
vw.release()

print()
print(f"长焦锁定 {stats['t_track']}/{N} ({stats['t_track']*100//N}%) | 广角锁定 {stats['w_track']}/{N} ({stats['w_track']*100//N}%)")
print(f"首锁 f{stats['first_lock']} | 出视野丢失 f{stats['lost_at']} | 找回 f{stats['relock']} | 橡皮筋 {stats['rope']} 帧")
print(f"耗时 {time.time()-t0:.0f}s | 输出 {OUT}")
