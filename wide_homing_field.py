#!/usr/bin/env python3
"""wide_homing_field.py — 广角场景归航 现场版。
参考图 = 沿途整张照片序列(默认 uploads/20260610/1..10.jpg,远→近,零裁剪)。
当前画面 vs 整图匹配 → 覆盖率 <50% 连续确认 + 下一张验证 → 切换,持续逼近。
输出: 实时 OSD(瞄准点+覆盖率) + replay.avi + switches/ 对比图 + homing.csv。"""
import cv2, numpy as np, os, time, argparse, queue, threading, csv
from concurrent.futures import ThreadPoolExecutor
from pure_siamese_track import Cam

ap = argparse.ArgumentParser()
ap.add_argument('--sensor-id', type=int, default=0, help='广角=0')
ap.add_argument('--ref-glob', default='uploads/20260610/{}.jpg', help='照片序列模板(1..n,远→近)')
ap.add_argument('--ref-n', type=int, default=10)
ap.add_argument('--width', type=int, default=1920)
ap.add_argument('--height', type=int, default=1080)
ap.add_argument('--fps', type=int, default=30)
ap.add_argument('--cov-switch', type=float, default=0.5)
ap.add_argument('--out-dir', default='')
args = ap.parse_args()

W, H = args.width, args.height
OUT = args.out_dir or f"algo_limit_tests/homing_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(os.path.join(OUT, 'switches'), exist_ok=True)

sift = cv2.SIFT_create(nfeatures=4000)
bf = cv2.BFMatcher()
refs = []
for i in range(1, args.ref_n + 1):
    p = args.ref_glob.format(i)
    img = cv2.imread(p)
    if img is None:
        continue
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    refs.append(dict(name=os.path.basename(p), img=img, kp=kp, des=des,
                     w=img.shape[1], h=img.shape[0]))
print(f"[REF] 整图照片序列 {len(refs)} 张(远→近)")


def match_ref(fkp, fdes, ref):
    if fdes is None or ref['des'] is None:
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
    if uniq < 12 or det < 0.05 or det > 20:    # 现场视点差比仿真大,det 略放宽
        return None
    return Hm, uniq


def coverage(Hm, ref):
    Hinv = np.linalg.inv(Hm)
    corners = np.float32([[0, 0], [W, 0], [W, H], [0, H]]).reshape(-1, 1, 2)
    foot = cv2.perspectiveTransform(corners, Hinv).reshape(-1, 2)
    rect = np.float32([[0, 0], [ref['w'], 0], [ref['w'], ref['h']], [0, ref['h']]])
    area, _ = cv2.intersectConvexConvex(foot.astype(np.float32), rect)
    return area / (ref['w'] * ref['h'])


def analyze(frame, idx):
    """后台线程:匹配当前 ref(必要时验证下一张)。返回 dict。"""
    fkp, fdes = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    r = dict(idx=idx, cov=-1.0, uniq=0, aim=None, next_ok=False, next_cov=-1.0)
    m = match_ref(fkp, fdes, refs[idx])
    if m is not None:
        Hm, uniq = m
        r['cov'] = coverage(Hm, refs[idx]); r['uniq'] = uniq
        c = cv2.perspectiveTransform(
            np.float32([[[refs[idx]['w'] / 2, refs[idx]['h'] / 2]]]), Hm).reshape(2)
        r['aim'] = (int(c[0]), int(c[1]))
        if r['cov'] < args.cov_switch and idx < len(refs) - 1:
            m2 = match_ref(fkp, fdes, refs[idx + 1])
            if m2 is not None:
                c2 = coverage(m2[0], refs[idx + 1])
                r['next_ok'] = 0.45 <= c2 <= 1.1
                r['next_cov'] = c2
    return r


cam = Cam(args.sensor_id, W, H, args.fps)
ex = ThreadPoolExecutor(max_workers=1)
fut = None; fut_frame = None
idx = 0; low_cnt = 0; last = None
rec_q = queue.Queue(maxsize=3); _wr = [None]
def rec_loop():
    while True:
        it = rec_q.get()
        if it is None: break
        if _wr[0] is None:
            hh, ww = it.shape[:2]
            _wr[0] = cv2.VideoWriter(os.path.join(OUT, 'replay.avi'),
                                     cv2.VideoWriter_fourcc(*'MJPG'), 20, (ww, hh))
        _wr[0].write(it)
threading.Thread(target=rec_loop, daemon=True).start()
cf = open(os.path.join(OUT, 'homing.csv'), 'w', newline='')
cw = csv.writer(cf); cw.writerow(['frame', 'ref_idx', 'ref', 'cov', 'uniq', 'aim_x', 'aim_y'])

cv2.namedWindow('Homing', cv2.WINDOW_NORMAL); cv2.resizeWindow('Homing', 1280, 720)
print(f"[OUT] {OUT} | Q 退出")
n = 0; t0 = time.perf_counter(); fps_now = 0.0
while True:
    frame = cam.read()
    if frame is None:
        continue
    if fut is None:
        fut_frame = frame
        fut = ex.submit(analyze, frame, idx)
    elif fut.done():
        try:
            last = fut.result()
        except Exception:
            last = None
        fut = None
        if last and last['idx'] == idx:
            if 0 <= last['cov'] < args.cov_switch:
                low_cnt += 1
            elif last['cov'] >= args.cov_switch:
                low_cnt = 0
            if low_cnt >= 3 and last['next_ok'] and idx < len(refs) - 1:
                idx += 1; low_cnt = 0
                r = refs[idx]
                th = cv2.resize(r['img'], (480, int(480 * r['h'] / r['w'])))
                fr = cv2.resize(fut_frame, (480, 270))
                pad = max(th.shape[0], 270)
                cmb = np.zeros((pad, 966, 3), np.uint8)
                cmb[:th.shape[0], :480] = th; cmb[:270, 486:966] = fr
                cv2.imwrite(os.path.join(OUT, 'switches', f'{n:05d}_to_{r["name"]}'), cmb) if False else \
                    cv2.imwrite(os.path.join(OUT, 'switches', f'{n:05d}_to_{os.path.splitext(r["name"])[0]}.jpg'), cmb)
                print(f"[SWITCH] f{n} → {r['name']} (next_cov={last['next_cov']:.2f})")
    disp = frame
    if last:
        cov = last['cov']
        col = (0, 255, 0) if cov >= 0 else (0, 0, 255)
        if last['aim'] and 0 <= last['aim'][0] < W and 0 <= last['aim'][1] < H:
            cv2.drawMarker(disp, last['aim'], (0, 0, 255), cv2.MARKER_CROSS, 70, 6)
        cv2.putText(disp, f"ref {refs[idx]['name']} ({idx+1}/{len(refs)})  cov={cov:.2f} uniq={last['uniq']}",
                    (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
        cw.writerow([n, idx, refs[idx]['name'], f"{last['cov']:.3f}", last['uniq'],
                     last['aim'][0] if last['aim'] else -1, last['aim'][1] if last['aim'] else -1])
    cv2.putText(disp, f"f{n} {fps_now:.0f}fps", (15, H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    out = cv2.resize(disp, (W // 2, H // 2))
    try:
        rec_q.put_nowait(out)
    except queue.Full:
        pass
    cv2.imshow('Homing', out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    n += 1
    if n % 20 == 0:
        now = time.perf_counter(); fps_now = 20 / (now - t0); t0 = now
rec_q.put(None); time.sleep(0.4)
if _wr[0] is not None:
    _wr[0].release()
cf.close(); cam.close(); cv2.destroyAllWindows()
print(f"[OUT] 数据: {OUT}/")
