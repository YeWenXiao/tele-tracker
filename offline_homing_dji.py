#!/usr/bin/env python3
"""offline_homing_dji.py — DJI 无人机实拍长距离归航离线测试。
ref = DJI_1..10.JPG 整图路标序列;视频 = wide.MP4(4K60 模拟广角行进)。
输出: 标注视频 + 切换对比图 + 统计。"""
import cv2, numpy as np, os, shutil, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--video', default='uploads/0611_dji/wide.MP4')
ap.add_argument('--tag', default='wide')
ap.add_argument('--step', type=int, default=5, help='每 N 帧分析一次')
args = ap.parse_args()

OUT = f'algo_limit_tests/dji_homing_{args.tag}'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, 'switches'))
W, H = 1920, 1080      # 4K 降到 1080p 分析

sift = cv2.SIFT_create(nfeatures=4000)
bf = cv2.BFMatcher()
refs = []
for i in range(1, 11):
    img = cv2.imread(f'uploads/0611_dji/DJI_{i}.JPG')
    img = cv2.resize(img, (W, int(W * img.shape[0] / img.shape[1])))
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    refs.append(dict(name=f'DJI_{i}', img=img, kp=kp, des=des, w=img.shape[1], h=img.shape[0]))
print(f"refs {len(refs)} 张 (统一 {W}px 宽)")


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
    if uniq < 12 or det < 0.01 or det > 30:
        return None
    return Hm, uniq


def occupancy(Hm, ref):
    corners = np.float32([[0, 0], [ref['w'], 0], [ref['w'], ref['h']], [0, ref['h']]]).reshape(-1, 1, 2)
    foot = cv2.perspectiveTransform(corners, Hm).reshape(-1, 2)
    rect = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    area, _ = cv2.intersectConvexConvex(foot.astype(np.float32), rect)
    return area / (W * H)


cap = cv2.VideoCapture(args.video)
vw = cv2.VideoWriter(os.path.join(OUT, 'replay.avi'),
                     cv2.VideoWriter_fourcc(*'MJPG'), 60, (960, 540))  # 源 60fps,逐帧写必须 60
idx = 0; cnt = 0; n = 0
last = dict(occ=-1.0, uniq=0, aim=None)
events = []; match_ok = 0; analyzed = 0
t0 = time.time()
while True:
    ok, raw = cap.read()
    if not ok:
        break
    frame = cv2.resize(raw, (W, H))
    if n % args.step == 0:
        analyzed += 1
        fkp, fdes = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        m = match_ref(fkp, fdes, refs[idx])
        if m is not None:
            match_ok += 1
            Hm, uniq = m
            occ = occupancy(Hm, refs[idx])
            c = cv2.perspectiveTransform(np.float32([[[refs[idx]['w'] / 2, refs[idx]['h'] / 2]]]), Hm).reshape(2)
            last = dict(occ=occ, uniq=uniq, aim=(int(c[0]), int(c[1])))
            cnt = cnt + 1 if occ > 0.5 else 0
            if cnt >= 3 and idx < len(refs) - 1:
                m2 = match_ref(fkp, fdes, refs[idx + 1])
                if m2 is not None:
                    c2 = occupancy(m2[0], refs[idx + 1])
                    if 0.05 <= c2 <= 1.05:
                        idx += 1; cnt = 0
                        events.append((n, refs[idx]['name'], c2))
                        # 切换对比图
                        Hc = 300
                        r_ = cv2.resize(refs[idx]['img'], (int(W * Hc / H), Hc))
                        cv2.putText(r_, f"REF {refs[idx]['name']}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        f_ = cv2.resize(frame, (int(W * Hc / H), Hc))
                        cv2.putText(f_, f"SWITCH f{n} occ={c2:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(OUT, 'switches', f"{n:05d}_{refs[idx]['name']}.jpg"),
                                    np.hstack([r_, np.full((Hc, 6, 3), 255, np.uint8), f_]))
                        print(f"[SWITCH] f{n} → {refs[idx]['name']} (occ={c2:.2f})")
        else:
            last = dict(occ=-1.0, uniq=0, aim=None)
    disp = frame
    if last['aim'] and 0 <= last['aim'][0] < W and 0 <= last['aim'][1] < H:
        cv2.drawMarker(disp, last['aim'], (0, 0, 255), cv2.MARKER_CROSS, 70, 6)
    col = (0, 255, 0) if last['occ'] >= 0 else (0, 0, 255)
    cv2.putText(disp, f"f{n} ref {refs[idx]['name']} ({idx+1}/10) occ={last['occ']:.2f} uniq={last['uniq']}",
                (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    vw.write(cv2.resize(disp, (960, 540)))
    n += 1
vw.release()
print()
print(f"总帧 {n} | 分析 {analyzed} | 匹配率 {match_ok*100//max(1,analyzed)}% | 走到 {refs[idx]['name']} ({idx+1}/10) | 切换 {len(events)} 次")
print(f"耗时 {time.time()-t0:.0f}s | {OUT}/")
