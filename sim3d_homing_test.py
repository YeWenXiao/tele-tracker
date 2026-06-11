#!/usr/bin/env python3
"""sim3d_homing_test.py — 场景级视觉归航:参考图=整张照片(不裁剪),覆盖率驱动切换。
参考图序列 = 沿接近路径在 [20,14,10,7,5,3.5,2.5]m 拍的整帧(广角)。
实时:当前画面 vs 当前参考图整图 SIFT 匹配 → 算"画面覆盖参考图的比例";
参考图占画面 >50% → 切下一张(更近拍的)。每次成功匹配/切换输出 参考图|当前画面 对比图。"""
import cv2, numpy as np, os, shutil, time
from sim3d_scene import build_world, render_view

W, H = 1920, 1080
FOCAL = 900                              # 广角
REF_DISTS = [20, 14, 10, 7, 5, 3.5, 2.5]   # 参考图拍摄距离(米)
OUT = 'algo_limit_tests/sim3d_homing'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, 'switches'))

world = build_world(box_size=0.5, box_pos=(0.0, 6.0))

# ── 预拍参考图序列(整帧,等价于实战中沿途拍的照片)──
print("预拍参考图序列:", REF_DISTS, "米")
sift = cv2.SIFT_create(nfeatures=4000)
bf = cv2.BFMatcher()
refs = []
for d in REF_DISTS:
    img = render_view(world, (0, 0.8, 6 - d), 0.0, FOCAL, W, H)
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    refs.append(dict(dist=d, img=img, kp=kp, des=des))
    cv2.imwrite(os.path.join(OUT, f'ref_{d}m.jpg'), img)


def match_ref(frame_kp, frame_des, ref):
    """当前画面 vs 整张参考图。返回 (H_ref2live, unique_inl) 或 None"""
    if frame_des is None or ref['des'] is None:
        return None
    good = [p[0] for p in bf.knnMatch(ref['des'], frame_des, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([ref['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([frame_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if Hm is None:
        return None
    pts = dst.reshape(-1, 2)[mask.ravel() == 1]
    uniq = len(np.unique(np.round(pts), axis=0))
    det = abs(np.linalg.det(Hm[:2, :2]))
    # 归航场景 det≈(拍摄距离/当前距离)²,合理范围收紧到 0.2~5(2 倍距离内),挡野 H
    if uniq < 12 or det < 0.2 or det > 5.0:
        return None
    return Hm, uniq


def occupancy(Hm):
    """参考图占当前画面的面积比例:参考图四角 → (H) → 画面坐标,与画面矩形求交。
    几何 = (拍摄距离/当前距离)²;>50% = 已逼近到拍摄距离 1.41 倍 → 该换下一张路标"""
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
    foot = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2)
    rect = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    area, _ = cv2.intersectConvexConvex(foot.astype(np.float32), rect)
    return area / (W * H)


def save_switch(tag, n, ref, frame, cov, uniq, aim):
    Hc = 330
    r = cv2.resize(ref['img'], (int(W * Hc / H), Hc))
    cv2.putText(r, f"REF @{ref['dist']}m", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    fr = frame.copy()
    if aim is not None:
        cv2.drawMarker(fr, aim, (0, 0, 255), cv2.MARKER_CROSS, 60, 6)
    fr = cv2.resize(fr, (int(W * Hc / H), Hc))
    cv2.putText(fr, f"{tag} f{n} occ={cov:.2f} uniq={uniq}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    sep = np.full((Hc, 6, 3), 255, np.uint8)
    cv2.imwrite(os.path.join(OUT, 'switches', f"{n:04d}_{tag}_{ref['dist']}m.jpg"), np.hstack([r, sep, fr]))


# ── 实时逼近:21m → 2.2m ──
vw = cv2.VideoWriter(os.path.join(OUT, 'replay.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 30, (W // 2, H // 2))
N = 560
idx = 0; acquired = False
low_cov_cnt = 0          # 防御①: 覆盖<0.5 须连续 K 帧确认(单帧坏匹配不切)
events = []
t0 = time.time()
for n in range(N):
    d = 21 - (21 - 2.2) * n / (N - 1)
    frame = render_view(world, (0, 0.8, 6 - d), 0.0, FOCAL, W, H)
    fkp, fdes = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    ref = refs[idx]
    m = match_ref(fkp, fdes, ref)
    cov = 0.0; aim = None; uniq = 0
    if m is not None:
        Hm, uniq = m
        cov = occupancy(Hm)
        # 瞄准点 = 前方路标(当前参考图)中心在画面的投影
        c = cv2.perspectiveTransform(np.float32([[[W / 2, H / 2]]]), Hm).reshape(2)
        aim = (int(c[0]), int(c[1]))
        if not acquired:
            acquired = True
            save_switch('ACQUIRE', n, ref, frame, cov, uniq, aim)
            events.append((n, 'ACQUIRE', ref['dist'], d, cov))
            print(f"[ACQUIRE] f{n} dist={d:.1f}m ref@{ref['dist']}m occ={cov:.2f} uniq={uniq}")
        else:
            # 你的规则: 参考图占画面 >50%(快走到这张路标了)→ 切下一张更近的
            low_cov_cnt = low_cov_cnt + 1 if cov > 0.5 else 0
            if low_cov_cnt >= 3 and idx < len(refs) - 1:
                # 切换前验证下一张路标可见(占比在合理小区间),否则保持
                ref2 = refs[idx + 1]
                m2 = match_ref(fkp, fdes, ref2)
                cov2 = occupancy(m2[0]) if m2 else -1
                if m2 is not None and 0.10 <= cov2 <= 0.60:
                    idx += 1; low_cov_cnt = 0
                    save_switch('SWITCH', n, ref2, frame, cov2, m2[1], aim)
                    events.append((n, 'SWITCH', ref2['dist'], d, cov2))
                    print(f"[SWITCH] f{n} dist={d:.1f}m → ref@{ref2['dist']}m (旧occ={cov:.2f} 新occ={cov2:.2f})")
    disp = frame.copy()
    if aim:
        cv2.drawMarker(disp, aim, (0, 0, 255), cv2.MARKER_CROSS, 60, 6)
    col = (0, 255, 0) if m else (0, 0, 255)
    cv2.putText(disp, f"f{n} dist={d:.1f}m ref@{refs[idx]['dist']}m occ={cov:.2f}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    vw.write(cv2.resize(disp, (W // 2, H // 2)))
vw.release()
print()
print(f"事件 {len(events)} 个:")
for n, tag, rd, d, c in events:
    print(f"  f{n} {tag} ref@{rd}m (实际距离 {d:.1f}m, cov={c:.2f})")
print(f"耗时 {time.time()-t0:.0f}s | {OUT}/replay.avi + switches/")
