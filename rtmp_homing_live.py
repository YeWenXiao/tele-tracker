#!/usr/bin/env python3
"""rtmp_homing_live.py — DJI 实时图传归航。
DJI Fly 直播(RTMP)推到本机 → ffmpeg -listen 收流 → 实时路标切换 + 瞄准点。
手机端设置: DJI Fly → 直播 → RTMP → rtmp://<Jetson IP>:1935/live
归航规则: 参考图占画面 >50%(连续确认+下一张验证)→ 切下一张更近路标。"""
import cv2, numpy as np, os, time, argparse, subprocess, threading, queue, csv, socket
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument('--port', type=int, default=1935)
ap.add_argument('--ref-glob', default='uploads/0611_dji/DJI_{}.JPG')
ap.add_argument('--ref-n', type=int, default=10)
ap.add_argument('--no-display', action='store_true')
ap.add_argument('--once', action='store_true', help='流断开后不重新监听(测试用)')
ap.add_argument('--max-sec', type=float, default=0, help='收流 N 秒后退出(测试用)')
ap.add_argument('--out-dir', default='')
args = ap.parse_args()

W, H = 1280, 720          # 分析分辨率(262ms/次≈3.8Hz,uniq 实测过门槛)
OUT = args.out_dir or f"algo_limit_tests/rtmp_homing_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(os.path.join(OUT, 'switches'), exist_ok=True)

sift = cv2.SIFT_create(nfeatures=2000)
bf = cv2.BFMatcher()
refs = []
for i in range(1, args.ref_n + 1):
    img = cv2.imread(args.ref_glob.format(i))
    if img is None:
        continue
    img = cv2.resize(img, (W, int(W * img.shape[0] / img.shape[1])))
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    refs.append(dict(name=f'DJI_{i}', img=img, kp=kp, des=des, w=img.shape[1], h=img.shape[0]))
print(f"[REF] {len(refs)} 张整图路标(@{W}px)")


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


def analyze(frame, idx):
    fkp, fdes = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
    r = dict(idx=idx, occ=-1.0, uniq=0, aim=None, next_ok=False, next_occ=-1.0)
    m = match_ref(fkp, fdes, refs[idx])
    if m is not None:
        Hm, uniq = m
        r['occ'] = occupancy(Hm, refs[idx]); r['uniq'] = uniq
        c = cv2.perspectiveTransform(np.float32([[[refs[idx]['w'] / 2, refs[idx]['h'] / 2]]]), Hm).reshape(2)
        r['aim'] = (int(c[0]), int(c[1]))
        if r['occ'] > 0.5 and idx < len(refs) - 1:
            m2 = match_ref(fkp, fdes, refs[idx + 1])
            if m2 is not None:
                c2 = occupancy(m2[0], refs[idx + 1])
                r['next_ok'] = 0.05 <= c2 <= 1.05
                r['next_occ'] = c2
    return r


def my_ips():
    try:
        return [ip for ip in subprocess.check_output(['hostname', '-I'], text=True).split()
                if not ip.startswith('127.')]
    except Exception:
        return ['<本机IP>']


# 录像(MJPG 流式) + CSV
rec_q = queue.Queue(maxsize=3); _wr = [None]
def rec_loop():
    while True:
        it = rec_q.get()
        if it is None:
            break
        if _wr[0] is None:
            hh, ww = it.shape[:2]
            _wr[0] = cv2.VideoWriter(os.path.join(OUT, 'replay.avi'),
                                     cv2.VideoWriter_fourcc(*'MJPG'), 30, (ww, hh))
        _wr[0].write(it)
threading.Thread(target=rec_loop, daemon=True).start()
csv_f = open(os.path.join(OUT, 'homing.csv'), 'w', newline='')
cw = csv.writer(csv_f); cw.writerow(['frame', 'ref_idx', 'ref', 'occ', 'uniq', 'aim_x', 'aim_y'])

ex = ThreadPoolExecutor(max_workers=1)
fut = None; fut_frame = None
idx = 0; high_cnt = 0; last = None
n = 0; t_start = None
fps_now = 0.0; t0 = time.perf_counter()
if not args.no_display:
    cv2.namedWindow('LiveHoming', cv2.WINDOW_NORMAL); cv2.resizeWindow('LiveHoming', 1280, 720)

FRAME_BYTES = W * H * 3
stop = False
while not stop:
    for ip in my_ips():
        print(f"[RTMP] 等待推流: DJI Fly → 直播 → rtmp://{ip}:{args.port}/live")
    proc = subprocess.Popen(
        ['ffmpeg', '-loglevel', 'error', '-listen', '1',
         '-i', f'rtmp://0.0.0.0:{args.port}/live',
         '-an', '-vf', f'scale={W}:{H}', '-pix_fmt', 'bgr24', '-f', 'rawvideo', '-'],
        stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 4)
    buf = b''
    while True:
        chunk = proc.stdout.read(FRAME_BYTES - len(buf))
        if not chunk:
            print("[RTMP] 流断开")
            break
        buf += chunk
        if len(buf) < FRAME_BYTES:
            continue
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
        buf = b''
        if t_start is None:
            t_start = time.time()
            print("[RTMP] ✓ 收到画面,归航开始")
        # 异步分析
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
                high_cnt = high_cnt + 1 if last['occ'] > 0.5 else (0 if last['occ'] >= 0 else high_cnt)
                if high_cnt >= 3 and last['next_ok'] and idx < len(refs) - 1:
                    idx += 1; high_cnt = 0
                    r = refs[idx]
                    th = cv2.resize(r['img'], (480, int(480 * r['h'] / r['w'])))
                    fr = cv2.resize(fut_frame, (480, 270))
                    pad = max(th.shape[0], 270)
                    cmb = np.zeros((pad, 966, 3), np.uint8)
                    cmb[:th.shape[0], :480] = th; cmb[:270, 486:966] = fr
                    cv2.imwrite(os.path.join(OUT, 'switches', f'{n:06d}_to_{r["name"]}.jpg'), cmb)
                    print(f"[SWITCH] f{n} → {r['name']} (occ={last['next_occ']:.2f})")
        disp = frame
        if last:
            if last['aim'] and 0 <= last['aim'][0] < W and 0 <= last['aim'][1] < H:
                cv2.drawMarker(disp, last['aim'], (0, 0, 255), cv2.MARKER_CROSS, 60, 5)
            col = (0, 255, 0) if last['occ'] >= 0 else (0, 0, 255)
            txt = f"ref {refs[idx]['name']} ({idx+1}/{len(refs)}) occ={last['occ']:.2f} uniq={last['uniq']}" \
                if last['occ'] >= 0 else f"ref {refs[idx]['name']} ({idx+1}/{len(refs)}) NO MATCH"
            cv2.putText(disp, txt, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            cw.writerow([n, idx, refs[idx]['name'], f"{last['occ']:.3f}", last['uniq'],
                         last['aim'][0] if last['aim'] else -1, last['aim'][1] if last['aim'] else -1])
        cv2.putText(disp, f"f{n} {fps_now:.0f}fps LIVE", (12, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        try:
            rec_q.put_nowait(disp.copy())
        except queue.Full:
            pass
        if not args.no_display:
            cv2.imshow('LiveHoming', disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop = True
                break
        n += 1
        if n % 30 == 0:
            now = time.perf_counter(); fps_now = 30 / (now - t0); t0 = now
        if args.max_sec and t_start and time.time() - t_start > args.max_sec:
            stop = True
            break
    proc.kill()
    if args.once:
        break

rec_q.put(None); time.sleep(0.4)
if _wr[0] is not None:
    _wr[0].release()
csv_f.close()
if not args.no_display:
    cv2.destroyAllWindows()
print(f"[OUT] {OUT}/ (replay.avi + homing.csv + switches/) | 共 {n} 帧 | 走到 {refs[idx]['name']}")
