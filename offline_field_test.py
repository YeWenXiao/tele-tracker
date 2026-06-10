#!/usr/bin/env python3
"""offline_field_test.py — 新 ref(refs_field_0610) × 当天实测视频 离线回放。
长焦半边(960x540,比生产 1080p 更难)按生产逻辑跑:SIFT 找(unique+颜色双门槛)→ ViT 跟(防护链)。
输出标注视频 + 统计。"""
import cv2, numpy as np, sys, time
from pure_siamese_track import SiftRef, color_similarity, make_tracker

SRC = 'algo_limit_tests/dual_cam_20260610_144611/replay.avi'
OUT = 'algo_limit_tests/offline_field_newref.avi'
W, H = 960, 540

sift = SiftRef('testdata/refs_field_0610', min_inliers=12, scale=1.0)
tracker = make_tracker('vit-trt')

cap = cv2.VideoCapture(SRC)
vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'MJPG'), 30, (W, H))

state = 'SEARCH'; bbox = None; score = 0.0
locked = False; low_cnt = 0; drift = 0
n = 0; n_track = 0; first_lock = -1; lock_events = []
t0 = time.time()

while True:
    ok, im = cap.read()
    if not ok:
        break
    frame = im[:, 960:].copy()          # 长焦半边
    if state == 'SEARCH':
        if n % 5 == 0:                  # 离线每5帧试一次 SIFT(生产是异步连续)
            b, inl = sift.locate(frame)
            if b is not None and sift.last_idx is not None:
                x, y, w, h = [int(v) for v in b]
                x = max(0, min(x, W - 2)); y = max(0, min(y, H - 2))
                w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
                if w >= 10 and h >= 10:
                    csim = color_similarity(frame[y:y+h, x:x+w], sift.ref_imgs[sift.last_idx])
                    if csim >= 0.25:
                        tracker.init(frame, (x, y, w, h))
                        bbox = (x, y, w, h); state = 'TRACK'
                        locked = True; low_cnt = 0; drift = 0
                        if first_lock < 0:
                            first_lock = n
                        lock_events.append((n, inl, csim))
                        print(f"[LOCK] f{n} bbox={bbox} uniq={inl} sim={csim:.2f}")
                    else:
                        sift.last_idx = None
        cv2.putText(frame, "SEARCHING (SIFT full-res)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        ok2, nb = tracker.update(frame)
        score = float(tracker.getTrackingScore())
        x, y, w, h = [int(v) for v in nb]
        if w > W * 0.5 or h > H * 0.5:          # max-box
            score = 0.0
        locked = score >= (0.5 if locked else 0.6)
        if locked and n % 10 == 0 and sift.last_idx is not None:   # TRACK 颜色检查
            cx0, cy0 = max(0, x), max(0, y)
            crop = frame[cy0:cy0+max(1, h), cx0:cx0+max(1, w)]
            csim = color_similarity(crop, sift.ref_imgs[sift.last_idx])
            drift = drift + 1 if csim < 0.15 else 0
            if drift >= 2:
                state = 'SEARCH'; locked = False; drift = 0
                print(f"[DRIFT] f{n} 颜色漂移 → 回 SEARCH")
        if locked:
            bbox = (x, y, w, h); low_cnt = 0; n_track += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"TRACKING s={score:.2f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        elif state == 'TRACK':
            low_cnt += 1
            cv2.putText(frame, f"LOST s={score:.2f}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            if low_cnt >= 45:
                state = 'SEARCH'
                print(f"[LOST] f{n} score 持续低 → 回 SEARCH")
    cv2.putText(frame, f"f{n}", (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    vw.write(frame)
    n += 1

vw.release()
print()
print(f"总帧 {n} | 锁定(TRACK绿框) {n_track} 帧 ({n_track*100//max(n,1)}%) | 首锁 f{first_lock}")
print(f"SIFT 锁定事件 {len(lock_events)} 次: " + ", ".join(f"f{f}(uniq{i}/sim{s:.2f})" for f, i, s in lock_events[:8]))
print(f"耗时 {time.time()-t0:.0f}s | 输出: {OUT}")
