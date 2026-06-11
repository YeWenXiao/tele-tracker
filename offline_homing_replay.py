#!/usr/bin/env python3
"""offline_homing_replay.py — 用现场录像离线回放归航逻辑(验证切换门槛修复)。
输入: 0611 现场 homing replay.avi(960x540 实拍) + uploads/20260610/1..10.jpg 整图序列。"""
import cv2, numpy as np, os, time

SRC = 'algo_limit_tests/homing_20260611_144415/replay.avi'
sift = cv2.SIFT_create(nfeatures=4000)
bf = cv2.BFMatcher()

refs = []
for i in range(1, 11):
    img = cv2.imread(f'uploads/20260610/{i}.jpg')
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    refs.append(dict(name=f'{i}.jpg', kp=kp, des=des, w=img.shape[1], h=img.shape[0]))
print(f"refs {len(refs)} 张")

W, H = 960, 540

def match_ref(fkp, fdes, ref):
    if fdes is None:
        return None
    good = [p[0] for p in bf.knnMatch(ref['des'], fdes, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([ref['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([fkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if Hm is None:
        return None
    pts = dst.reshape(-1, 2)[mask.ravel() == 1]
    uniq = len(np.unique(np.round(pts), axis=0))
    det = abs(np.linalg.det(Hm[:2, :2]))
    if uniq < 12 or det < 0.01 or det > 20:   # 半尺寸视频 det 比 1080p 小 4 倍,下限放宽
        return None
    return Hm, uniq

def occupancy(Hm, ref):
    corners = np.float32([[0, 0], [ref['w'], 0], [ref['w'], ref['h']], [0, ref['h']]]).reshape(-1, 1, 2)
    foot = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2)
    rect = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    area, _ = cv2.intersectConvexConvex(foot.astype(np.float32), rect)
    return area / (W * H)

cap = cv2.VideoCapture(SRC)
idx = 0; cnt = 0; n = 0
events = []
t0 = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if n % 5 == 0:                                  # 每 5 帧分析一次(加速)
        fkp, fdes = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        m = match_ref(fkp, fdes, refs[idx])
        if m is not None:
            occ = occupancy(m[0], refs[idx])
            cnt = cnt + 1 if occ > 0.5 else 0
            if cnt >= 3 and idx < len(refs) - 1:
                m2 = match_ref(fkp, fdes, refs[idx + 1])
                if m2 is not None:
                    c2 = occupancy(m2[0], refs[idx + 1])
                    if 0.05 <= c2 <= 1.05:          # 修复后的门槛
                        idx += 1; cnt = 0
                        events.append((n, refs[idx]['name'], c2))
                        print(f"[SWITCH] f{n} → {refs[idx]['name']} (occ={c2:.2f})")
    n += 1
print()
print(f"总帧 {n} | 走到 ref {refs[idx]['name']} ({idx+1}/{len(refs)}) | 切换 {len(events)} 次 | 耗时 {time.time()-t0:.0f}s")
