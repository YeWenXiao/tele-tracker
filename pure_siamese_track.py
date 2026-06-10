#!/usr/bin/env python3
"""pure_siamese_track.py — 纯净 Siamese 追踪基线(零补丁,不依赖 t5)

逻辑极简:
  SEARCH 态: SIFT 匹配 ref 找目标 → init DaSiamRPN
  TRACK  态: 每帧 tracker.update + getTrackingScore
             score >= min → 跟踪(绿框)
             score <  min → 目标消失,不锁干扰(显示 SEARCHING,tracker 仍在 search region)
             目标重入原处 → score 回升 → 自然恢复
             score 持续低 N 帧 → 回 SEARCH(SIFT 全画面重找,目标从别处回来)
键: S 强制重新搜索 | Q 退出
"""
import os, sys, time, csv, glob, argparse, threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


class Cam:
    def __init__(self, sensor_id, w=1920, h=1080, fps=30, lock_awb=False):
        self.w, self.h = w, h
        # 锁白平衡+曝光:同物体在不同帧颜色稳定(自动 AWB/AE 会让红箱忽明忽暗,干扰颜色判据)
        lock = "wbmode=1 awblock=true aelock=true " if lock_awb else ""
        s = (f"nvarguscamerasrc sensor-id={sensor_id} tnr-mode=1 ee-mode=1 {lock}! "
             f"video/x-raw(memory:NVMM), width={w}, height={h}, format=NV12, framerate={fps}/1 ! "
             f"nvvidconv ! video/x-raw, format=BGRx ! "
             f"appsink name=sink emit-signals=1 drop=1 max-buffers=1")
        self.pipe = Gst.parse_launch(s)
        self.sink = self.pipe.get_by_name("sink")
        self.pipe.set_state(Gst.State.PLAYING)
        if self.pipe.get_state(10 * Gst.SECOND)[0] != Gst.StateChangeReturn.SUCCESS:
            raise RuntimeError("camera pipeline 启动失败")
        print(f"[CAM] sensor-id={sensor_id} {w}x{h}@{fps} OK")
        self.latest = None
        self._lock = threading.Lock()
        self._run = True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        # 后台线程一直拉最新帧(取帧与主线程 tracker 处理并行,主线程不阻塞等帧)
        while self._run:
            smp = self.sink.emit("pull-sample")
            if smp is None:
                continue
            buf = smp.get_buffer()
            ok, mi = buf.map(Gst.MapFlags.READ)
            if ok:
                f = np.frombuffer(mi.data, np.uint8).reshape(self.h, self.w, 4)[:, :, :3].copy()
                buf.unmap(mi)
                with self._lock:
                    self.latest = f

    def read(self):
        with self._lock:
            return None if self.latest is None else self.latest.copy()

    def close(self):
        self._run = False
        time.sleep(0.1)
        self.pipe.set_state(Gst.State.NULL)


class VideoFile:
    """读视频文件(合成测试用,同 Cam 接口)。"""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开视频 {path}")
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.eof = False
        print(f"[VIDEO] {path} {self.w}x{self.h}")

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            self.eof = True
            return None
        return f

    def close(self):
        self.cap.release()


def make_tracker(backend='trt'):
    # cv=DaSiamRPN cv2.dnn FP32 / trt=DaSiamRPN TRT FP16 / vit,vit-fp16=ViT Transformer(考题碾压 DaSiamRPN)
    if backend == 'cv':
        p = cv2.TrackerDaSiamRPN_Params()
        p.model = os.path.join(MODEL_DIR, "dasiamrpn_model.onnx")
        p.kernel_cls1 = os.path.join(MODEL_DIR, "dasiamrpn_kernel_cls1.onnx")
        p.kernel_r1 = os.path.join(MODEL_DIR, "dasiamrpn_kernel_r1.onnx")
        p.backend = cv2.dnn.DNN_BACKEND_CUDA
        p.target = cv2.dnn.DNN_TARGET_CUDA
        return cv2.TrackerDaSiamRPN.create(p)
    if backend == 'vit-trt':   # 纯 TRT ViT(干掉 cv2.dnn,5.8ms vs cv2.dnn 30ms,精度对齐)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'int8_quantize'))
        from int_vittrack import IntViTTrack
        return IntViTTrack()
    if backend in ('vit', 'vit-fp16'):
        p = cv2.TrackerVit_Params()
        p.net = os.path.join(MODEL_DIR, "vittrack.onnx")
        p.backend = cv2.dnn.DNN_BACKEND_CUDA
        p.target = cv2.dnn.DNN_TARGET_CUDA_FP16 if backend == 'vit-fp16' else cv2.dnn.DNN_TARGET_CUDA
        return cv2.TrackerVit.create(p)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'int8_quantize'))
    from int_dasiamrpn import IntDaSiamRPN
    return IntDaSiamRPN()   # dasiamrpn_model_271_dynkern_fp16.trt


def color_similarity(crop, ref_img):
    """crop 和 ref 的 HSV 颜色直方图相关性(挡'锁到颜色完全不同的东西',如背景)。
    实测:锁对红箱 0.31~0.84,锁错背景 0.02 → 门槛 0.25 能分开。"""
    if crop is None or crop.size == 0 or ref_img is None:
        return 0.0
    def hist(im):
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
        return h
    return float(cv2.compareHist(hist(crop), hist(ref_img), cv2.HISTCMP_CORREL))


def save_sift_lock(lock_dir, fi, ref_img, crop, inl, ref_name, sim=1.0, accepted=True):
    """每次 SIFT 锁定/拒绝后:把锁到的物体 crop 和命中的 ref 并排存图,标 inliers + 颜色 sim。"""
    H = 240
    status = "OK" if accepted else "REJECT"
    def fit(im, label, color):
        if im is None or im.size == 0:
            im = np.zeros((H, H, 3), np.uint8)
        w = max(1, int(im.shape[1] * H / im.shape[0]))
        r = cv2.resize(im, (w, H))
        cv2.putText(r, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return r
    left = fit(ref_img, f"REF {ref_name}", (0, 255, 0))
    rc = (0, 200, 255) if accepted else (0, 0, 255)
    right = fit(crop, f"{status} inl={inl} sim={sim:.2f}", rc)
    sep = np.full((H, 6, 3), 255, np.uint8)
    cv2.imwrite(os.path.join(lock_dir, f"lock_f{fi:04d}_{status}_inl{inl}_sim{int(sim*100):02d}.jpg"),
                np.hstack([left, sep, right]))


class SiftRef:
    def __init__(self, ref_dir, min_inliers=15, scale=0.5):
        self.sift = cv2.SIFT_create(nfeatures=2000)
        self.bf = cv2.BFMatcher()
        self.min_inliers = min_inliers
        self.scale = scale   # SEARCH 态 SIFT 在缩小帧上跑(detectAndCompute 加速 ~4x)
        self.last_idx = None   # 上次成功命中的 ref 索引(优先复用)
        self.refs = []
        self.ref_imgs = []   # 原图,锁定后对比验证用
        for f in sorted(glob.glob(os.path.join(ref_dir, '*.jpg')) +
                        glob.glob(os.path.join(ref_dir, '*.png'))):
            img = cv2.imread(f)
            kp, des = self.sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            if des is not None:
                self.refs.append((os.path.basename(f), img.shape, kp, des))
                self.ref_imgs.append(img)
        print(f"[REF] 加载 {len(self.refs)} 张 from {ref_dir}")

    def locate(self, frame):
        # 帧 SIFT 只算一次(不是每张 ref 重算),ref early-exit(找到第一张过门槛就返回)
        s = self.scale
        small = cv2.resize(frame, None, fx=s, fy=s) if s < 1.0 else frame
        kf, df = self.sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
        if df is None:
            return None, 0
        # 【回退 v1】early-exit + 优先上次成功 ref(找到第一张过门槛就返回,锁对真目标)
        order = list(range(len(self.refs)))
        if self.last_idx is not None:
            order = [self.last_idx] + [i for i in order if i != self.last_idx]
        for i in order:
            name, shape, kp, des = self.refs[i]
            good = [p[0] for p in self.bf.knnMatch(des, df, k=2)
                    if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) < 4:
                continue
            src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None:
                continue
            inl_pts_all = dst.reshape(-1, 2)[mask.ravel() == 1]
            # inliers 按 unique 帧点计数:远距下"多 ref 点匹配同一帧点"会伪造高 inliers
            #(实测 22 inliers 挤在 4 个点上),unique 才是真匹配数
            inl = len(np.unique(np.round(inl_pts_all), axis=0))
            det = abs(np.linalg.det(H[:2, :2]))
            if det < 0.02 or det > 30:     # 退化 H(实测假匹配 det~1e-3)
                continue
            if inl >= self.min_inliers:
                self.last_idx = i
                # 框收紧:用 inlier 匹配点的实际范围(不用 ref 四角投影,避免半物体外推把背景框进来)
                pts = dst.reshape(-1, 2)[mask.ravel() == 1]
                # 远目标修复:中位数+MAD 聚类踢离群点 —— 真匹配密集在目标上,
                # 零散误匹配(远处干扰物/平面背景巧合过 RANSAC)会把 min/max 框撑到半个场景
                med = np.median(pts, axis=0)
                dev = np.abs(pts - med).max(axis=1)            # 每点到中位中心的切比雪夫距离
                mad = max(np.median(dev), 1.0)
                core = pts[dev <= 5.0 * mad]                   # 5×MAD 内 = 目标上的密集簇
                if len(core) >= max(4, len(pts) // 2):         # 簇要占一半以上才可信
                    pts = core
                x0, y0 = pts.min(0); x1, y1 = pts.max(0)
                pw, ph = x1 - x0, y1 - y0
                pad = 0.08   # 补偿边缘特征缺失(GOOD LUCK 的红边几乎无特征点)
                x0 -= pw * pad; y0 -= ph * pad; pw *= 1.16; ph *= 1.16
                return (int(x0 / s), int(y0 / s), int(pw / s), int(ph / s)), inl
        return None, 0


def main():
    ap = argparse.ArgumentParser(description="纯净 Siamese 追踪基线")
    ap.add_argument('--sensor-id', type=int, default=1, help='长焦=1')
    ap.add_argument('--video', default='', help='读视频文件代替摄像头(合成测试用)')
    ap.add_argument('--ref-dir', default='testdata/refs_tele_real')
    ap.add_argument('--score-min', type=float, default=0.6, help='目标消失阈值')
    ap.add_argument('--sift-scale', type=float, default=0.3, help='SIFT 找回缩放(0.3=54ms;有双门槛保底,误匹配会被挡)')
    ap.add_argument('--sift-min-inliers', type=int, default=30, help='SIFT 找回 inliers 门槛(<30 一律拒;宁可漏锁不锁错)')
    ap.add_argument('--relost-frames', type=int, default=45, help='score 持续低 N 帧 → 回 SEARCH(先给 siamese 机会从原处恢复)')
    ap.add_argument('--search-interval', type=int, default=3, help='SEARCH 态每 N 帧跑一次 SIFT(降频防卡)')
    ap.add_argument('--display-scale', type=float, default=0.6, help='显示/录像缩放(tracker 仍用原帧),小=快')
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=60, help='摄像头帧率(IMX477 1080p 支持 60)')
    ap.add_argument('--no-display', action='store_true', help='不显示窗口(只录像,合成测试/headless 用)')
    ap.add_argument('--tracker', choices=['cv', 'trt', 'vit', 'vit-fp16', 'vit-trt'], default='trt',
                    help='cv/trt=DaSiamRPN / vit=ViT cv2.dnn / vit-trt=ViT 纯TRT 5.8ms(碾压精度+最快,推荐)')
    ap.add_argument('--color-min', type=float, default=0.25, help='锁定后颜色相似度门槛(crop vs ref);<门槛=锁错背景/异物,拒绝。宁可漏不锁错')
    ap.add_argument('--lock-awb', action='store_true', help='锁白平衡+曝光(红箱颜色稳定,曝光变化不再忽明忽暗)')
    ap.add_argument('--score-low', type=float, default=0.5, help='score 迟滞低门槛(已锁定时降到此值才丢,防锁定状态反复闪)')
    ap.add_argument('--color-check-interval', type=int, default=10, help='TRACK 态每 N 帧颜色检查一次当前框')
    ap.add_argument('--color-track-min', type=float, default=0.15, help='TRACK 态颜色门槛(松,防白平衡误杀;漂到深色物体 sim≈0 必挡)')
    ap.add_argument('--max-box-ratio', type=float, default=0.5, help='框宽/高超画面此比例=tracker解码失败(全屏大框)→判丢失重找')
    args = ap.parse_args()

    if args.video:
        cam = VideoFile(args.video)
        args.width, args.height = cam.w, cam.h   # 用视频实际尺寸
    else:
        cam = Cam(args.sensor_id, args.width, args.height, args.fps, lock_awb=args.lock_awb)
    sift = SiftRef(args.ref_dir, min_inliers=args.sift_min_inliers, scale=args.sift_scale)
    tracker = make_tracker(args.tracker)   # 创建一次复用(后续只 re-init 换 template)
    print(f"[TRACKER] {args.tracker.upper()}")
    state = 'SEARCH'      # SEARCH / TRACK
    bbox = None
    low_cnt = 0
    score = 0.0

    out_dir = args.out_dir or f"algo_limit_tests/pure_siamese_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    lock_dir = os.path.join(out_dir, 'sift_locks')   # 每次锁定的 ref vs 实锁物体对比图
    os.makedirs(lock_dir, exist_ok=True)
    # 异步录像:写盘线程编码,主循环不等(MJPG 编码 61ms 太慢,不能阻塞主循环)
    rec_q = queue.Queue(maxsize=3)
    _wr = [None]

    def rec_loop():
        while True:
            item = rec_q.get()
            if item is None:
                break
            if _wr[0] is None:
                hh, ww = item.shape[:2]
                _wr[0] = cv2.VideoWriter(os.path.join(out_dir, 'replay.avi'),
                                         cv2.VideoWriter_fourcc(*'MJPG'), 30, (ww, hh))
            _wr[0].write(item)
    rec_t = threading.Thread(target=rec_loop, daemon=True)
    rec_t.start()
    csv_f = open(os.path.join(out_dir, 'track.csv'), 'w', newline='')
    cw = csv.writer(csv_f); cw.writerow(['frame', 'ms', 'state', 'score', 'x', 'y', 'w', 'h'])

    if not args.no_display:
        cv2.namedWindow("Pure Siamese", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pure Siamese", 1280, 720)
    print("S=重新搜索 | Q=退出")
    fi = 0
    t0 = time.perf_counter()
    fps_now = 0.0
    sift_exec = ThreadPoolExecutor(max_workers=1)   # SIFT 异步,不阻塞主循环
    sift_future = None
    sift_frame = None   # 记住提交给 SIFT 的那一帧(异步返回时主循环已前进,必须用同一帧 init)
    locked = False      # score 迟滞锁定状态
    drift_cnt = 0       # TRACK 态颜色漂移计数
    while True:
        frame_start = time.perf_counter()
        frame = cam.read()
        if frame is None:
            if getattr(cam, 'eof', False):   # 视频播完退出
                break
            continue
        disp = frame
        if state == 'SEARCH':
            # SIFT 异步:后台线程跑(594ms 全帧/151ms 缩小,不能阻塞主循环),主循环保持 30fps
            if sift_future is None:
                sift_frame = frame   # 记住提交给 SIFT 的帧
                sift_future = sift_exec.submit(sift.locate, frame)
            elif sift_future.done():
                try:
                    b, inl = sift_future.result()
                except Exception as e:
                    print(f"[SIFT 异步异常] {e}")
                    b, inl = None, 0
                sift_future = None
                if b is not None and sift_frame is not None:
                    # 验证 + clamp bbox(SIFT homography 可能投影出越界/退化框 → tracker.init resize 崩)
                    H_img, W_img = sift_frame.shape[:2]
                    x, y, w, h = [int(v) for v in b]
                    x = max(0, min(x, W_img - 2)); y = max(0, min(y, H_img - 2))
                    w = max(1, min(w, W_img - x)); h = max(1, min(h, H_img - y))
                    if w >= 20 and h >= 20:
                        crop = sift_frame[y:y + h, x:x + w].copy()
                        ref_img = sift.ref_imgs[sift.last_idx] if sift.last_idx is not None else None
                        ref_name = sift.refs[sift.last_idx][0] if sift.last_idx is not None else '?'
                        csim = color_similarity(crop, ref_img)
                        if csim < args.color_min:
                            # 颜色不像 ref(锁到背景/异物)→ 拒绝(宁可漏锁,不可锁错)
                            save_sift_lock(lock_dir, fi, ref_img, crop, inl, ref_name, csim, accepted=False)
                            sift.last_idx = None   # 清除,下次重新找
                            print(f"[REJECT] 颜色 sim={csim:.2f} < {args.color_min} → 拒锁(疑似锁错背景/异物)")
                        else:
                            b = (x, y, w, h)
                            tracker.init(sift_frame, b)   # 帧-bbox 同步
                            save_sift_lock(lock_dir, fi, ref_img, crop, inl, ref_name, csim, accepted=True)
                            bbox = b; state = 'TRACK'; low_cnt = 0
                            locked = True; drift_cnt = 0   # 重置跟踪后处理状态
                            print(f"[LOCK] bbox={b} inl={inl} 颜色sim={csim:.2f} → init siamese")
                    else:
                        print(f"[SKIP] SIFT bbox 无效(w={w} h={h}),继续搜索")
            cv2.putText(disp, "SEARCHING for target (SIFT async)...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:  # TRACK
            ok, nb = tracker.update(frame)
            score = float(tracker.getTrackingScore())
            x, y, w, h = [int(v) for v in nb]   # 框直接用 siamese 原始输出(震动平滑交给飞控,不在这层做)
            # tracker 解码异常:框占满屏(目标不可能突然全屏)= 解码失败,当 score 0 触发重找
            if w > frame.shape[1] * args.max_box_ratio or h > frame.shape[0] * args.max_box_ratio:
                score = 0.0
            # score 迟滞(进锁高门槛 score_min,保持低门槛 score_low,防锁定状态一帧锁一帧不锁的闪)
            locked = score >= (args.score_low if locked else args.score_min)
            # TRACK 态颜色检查(每 N 帧;漂到颜色不同的东西如深色物体 → 回 SEARCH)
            if locked and fi % args.color_check_interval == 0 and sift.last_idx is not None:
                cx0, cy0 = max(0, x), max(0, y)
                crop = frame[cy0:cy0 + max(1, h), cx0:cx0 + max(1, w)]
                csim = color_similarity(crop, sift.ref_imgs[sift.last_idx])
                drift_cnt = drift_cnt + 1 if csim < args.color_track_min else 0
                if drift_cnt >= 2:
                    state = 'SEARCH'; locked = False; drift_cnt = 0
                    print(f"[DRIFT] TRACK 颜色 sim={csim:.2f} → 漂到非目标,回 SEARCH")
            # ④ 显示
            if locked:
                bbox = (x, y, w, h); low_cnt = 0
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(disp, f"TRACKING  score={score:.2f}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            elif state == 'TRACK':
                low_cnt += 1
                cv2.putText(disp, f"TARGET LOST  score={score:.2f}  (not locking distractor)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(disp, "waiting target re-enter / S=research", (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                if low_cnt >= args.relost_frames:
                    state = 'SEARCH'
                    print(f"[LOST] score 持续低 {low_cnt} 帧 → 回 SEARCH(SIFT 全画面重找)")
        ms = (time.perf_counter() - t0) * 1000
        bx = bbox if (state == 'TRACK' and locked) else (-1, -1, -1, -1)
        cw.writerow([fi, f'{ms:.0f}', state, f'{score:.3f}',
                     bx[0], bx[1], bx[2], bx[3]])
        # 显示/录像用缩小帧(tracker 已用原帧跟踪完,这里只为显示)
        out = cv2.resize(disp, None, fx=args.display_scale, fy=args.display_scale)
        cv2.putText(out, f"f{fi} {fps_now:.0f}fps", (15, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        try:
            rec_q.put_nowait(out)          # 异步录像,队列满则丢帧(不阻塞主循环)
        except queue.Full:
            pass
        if not args.no_display:
            cv2.imshow("Pure Siamese", out)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                state = "SEARCH"; low_cnt = 0
                print("[KEY] 手动回 SEARCH")
        fi += 1
        # 实时 fps + 每帧耗时诊断
        dt = time.perf_counter() - frame_start
        fps_now = 1.0 / dt if dt > 0 else 0
        if fi % 30 == 0:
            csv_f.flush()
            print(f"[FPS] {fps_now:.1f}fps  帧耗时 {dt*1000:.0f}ms  state={state} score={score:.2f}")

    sift_exec.shutdown(wait=False)
    rec_q.put(None); rec_t.join(timeout=3)
    if _wr[0]:
        _wr[0].release()
    csv_f.close(); cam.close(); cv2.destroyAllWindows()
    print(f"[OUT] {out_dir}/replay.avi + track.csv")


if __name__ == '__main__':
    main()
