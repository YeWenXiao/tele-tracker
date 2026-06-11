#!/usr/bin/env python3
"""offline_tele_dji.py — 长焦提前捕获最终目标测试。
tele.MP4(4K 长焦行进)逐段匹配最终路标 DJI_10/DJI_9(白色建筑),
看长焦从多早(多远)就能锁定最终目标并稳定给出瞄准点。"""
import cv2, numpy as np, os, shutil, time

OUT = 'algo_limit_tests/dji_tele'
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)
W, H = 1920, 1080

sift = cv2.SIFT_create(nfeatures=4000)
bf = cv2.BFMatcher()
targets = []
for name in ('DJI_10', 'DJI_9'):
    img = cv2.imread(f'uploads/0611_dji/{name}.JPG')
    img = cv2.resize(img, (W, int(W * img.shape[0] / img.shape[1])))
    kp, des = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    targets.append(dict(name=name, img=img, kp=kp, des=des, w=img.shape[1], h=img.shape[0]))

def match(fkp, fdes, t):
    if fdes is None:
        return None
    good = [p[0] for p in bf.knnMatch(t['des'], fdes, k=2)
            if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return None
    src = np.float32([t['kp'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([fkp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if Hm is None:
        return None
    pts = dst.reshape(-1, 2)[mask.ravel() == 1]
    uniq = len(np.unique(np.round(pts), axis=0))
    det = abs(np.linalg.det(Hm[:2, :2]))
    if uniq < 12 or det < 0.005 or det > 50:
        return None
    return Hm, uniq

cap = cv2.VideoCapture('uploads/0611_dji/tele.MP4')
vw = cv2.VideoWriter(os.path.join(OUT, 'replay.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 60, (960, 540))  # 源 60fps,逐帧写必须 60
n = 0; analyzed = 0
hits = {t['name']: 0 for t in targets}
first = {t['name']: -1 for t in targets}
last = dict(aim=None, name='', uniq=0)
t0 = time.time()
while True:
    ok, raw = cap.read()
    if not ok:
        break
    frame = cv2.resize(raw, (W, H))
    if n % 5 == 0:
        analyzed += 1
        fkp, fdes = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        last = dict(aim=None, name='', uniq=0)
        for t in targets:                       # 优先最终路标 DJI_10
            m = match(fkp, fdes, t)
            if m is not None:
                Hm, uniq = m
                c = cv2.perspectiveTransform(np.float32([[[t['w'] / 2, t['h'] / 2]]]), Hm).reshape(2)
                last = dict(aim=(int(c[0]), int(c[1])), name=t['name'], uniq=uniq)
                hits[t['name']] += 1
                if first[t['name']] < 0:
                    first[t['name']] = n
                    print(f"[FIRST] {t['name']} 首次捕获 @ f{n} ({n/60:.0f}s) uniq={uniq}")
                break
    disp = frame
    if last['aim'] and -2000 < last['aim'][0] < W + 2000:
        a = (int(np.clip(last['aim'][0], 10, W - 10)), int(np.clip(last['aim'][1], 10, H - 10)))
        cv2.drawMarker(disp, a, (0, 0, 255), cv2.MARKER_CROSS, 70, 6)
    col = (0, 255, 0) if last['name'] else (0, 0, 255)
    cv2.putText(disp, f"f{n} target={last['name'] or 'none'} uniq={last['uniq']}", (15, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    vw.write(cv2.resize(disp, (960, 540)))
    n += 1
vw.release()
print()
for t in targets:
    nm = t['name']
    print(f"{nm}: 首捕获 f{first[nm]} ({first[nm]/60:.0f}s) | 命中 {hits[nm]}/{analyzed} ({hits[nm]*100//max(1,analyzed)}%)")
print(f"耗时 {time.time()-t0:.0f}s | {OUT}/replay.avi")
