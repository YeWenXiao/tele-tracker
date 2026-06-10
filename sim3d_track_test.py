#!/usr/bin/env python3
"""sim3d_track_test.py — 3D 场景跑真实算法(长焦视角,由远及近 20m→2.6m)。
每次识别成功(SIFT 锁定 / ref 阶梯切换)输出对比图: 左=参考图, 右=成功画面(带框)。
ref 留一法排除 field_10(场景贴图就是它,防自匹配)。"""
import cv2, numpy as np, os, glob, shutil, time
from sim3d_scene import build_world, render_view
from dual_cam_track import TrackPipeline

W, H = 1920, 1080
FOCAL_TELE = 2700                      # 3x 长焦
OUT = 'algo_limit_tests/sim3d_track'
LOCKS = os.path.join(OUT, 'locks')
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(LOCKS)

REF = 'testdata/refs_sim3d_loo'
shutil.rmtree(REF, ignore_errors=True); os.makedirs(REF)
for f in glob.glob('testdata/refs_field_0610/*.jpg'):
    if os.path.basename(f) != 'field_10.jpg':      # 留一:场景贴图=field_10
        shutil.copy(f, REF)

def save_compare(tag, n, ref_img, ref_name, frame, bbox, extra=''):
    """左=参考图 右=成功画面(带绿框);存 locks/"""
    Hc = 360
    r = cv2.resize(ref_img, (int(ref_img.shape[1] * Hc / ref_img.shape[0]), Hc))
    cv2.putText(r, f"REF {ref_name}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    fr = frame.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 255, 0), 4)
    fr = cv2.resize(fr, (int(W * Hc / H), Hc))
    cv2.putText(fr, f"{tag} f{n} {extra}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    sep = np.full((Hc, 6, 3), 255, np.uint8)
    cv2.imwrite(os.path.join(LOCKS, f"{n:04d}_{tag}.jpg"), np.hstack([r, sep, fr]))

world = build_world(box_size=0.5, box_pos=(0.0, 6.0))
pipe = TrackPipeline('TELE', REF, W, H, sift_scale=1.0, min_inliers=12)
vw = cv2.VideoWriter(os.path.join(OUT, 'replay.avi'),
                     cv2.VideoWriter_fourcc(*'MJPG'), 30, (W // 2, H // 2))

N = 600
events = []
t0 = time.time()
for n in range(N):
    z = -14 + (3.4 - (-14)) * n / (N - 1)          # 距箱 20m → 2.6m
    dist = 6.0 - z
    frame = render_view(world, (0, 0.8, z), 0.0, FOCAL_TELE, W, H)
    prev_state = pipe.state
    prev_idx = pipe.sift.last_idx
    state, bbox, score, locked = pipe.step(frame)
    cur_idx = pipe.sift.last_idx
    if prev_state == 'SEARCH' and state == 'TRACK' and cur_idx is not None:
        ref_name = pipe.sift.refs[cur_idx][0]
        save_compare('LOCK', n, pipe.sift.ref_imgs[cur_idx], ref_name, frame, bbox, f"dist={dist:.1f}m")
        events.append((n, 'LOCK', ref_name, dist))
        print(f"[LOCK] f{n} dist={dist:.1f}m ref={ref_name}")
    elif locked and cur_idx is not None and prev_idx is not None and cur_idx != prev_idx:
        ref_name = pipe.sift.refs[cur_idx][0]
        save_compare('LADDER', n, pipe.sift.ref_imgs[cur_idx], ref_name, frame, bbox, f"dist={dist:.1f}m")
        events.append((n, 'LADDER', ref_name, dist))
        print(f"[LADDER] f{n} dist={dist:.1f}m → {ref_name}")
    disp = frame.copy()
    if locked and bbox:
        x, y, w, h = bbox
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(disp, f"f{n} {state} s={score:.2f} dist={dist:.1f}m", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0) if locked else (0, 0, 255), 2)
    vw.write(cv2.resize(disp, (W // 2, H // 2)))
vw.release()
n_track = sum(1 for _ in range(1))  # placeholder
print()
print(f"识别成功事件 {len(events)} 次(对比图在 {LOCKS}/):")
for n, tag, ref, d in events:
    print(f"  f{n} {tag} {ref} @ {d:.1f}m")
print(f"耗时 {time.time()-t0:.0f}s | 视频 {OUT}/replay.avi")
shutil.rmtree(REF, ignore_errors=True)
