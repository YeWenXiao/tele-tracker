#!/usr/bin/env python3
"""dual_cam_track.py — V2.2 双摄:长焦主追踪 + 广角辅助。
Step 1: 双摄取帧 + 长焦主追踪(SIFT异步找+ViT跟+防护链) + 广角显示固定长焦视野框(黄框)。
        视野框=固定几何(两摄物理固定),--fov-ratio/center-x/y 标定;不依赖识别目标。
坐标系: 广角画面 左→右 x:-1→1, 上→下 y:1→-1, 中心(0,0)。Step 2 再加失锁指引。
"""
import os, sys, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
# 复用 pure_siamese 的组件(有 __main__ guard,import 安全)
from pure_siamese_track import Cam, make_tracker, SiftRef, color_similarity


class TrackPipeline:
    """单摄追踪管线:SIFT 找(双门槛+颜色) → ViT 跟(score迟滞+颜色检查+max-box)。长焦/广角各一个。"""
    def __init__(self, name, ref_dir, W, H, sift_scale=0.3, min_inliers=30,
                 color_min=0.25, score_min=0.6, score_low=0.5, color_track_min=0.15,
                 max_box_ratio=0.5, relost=45):
        self.name = name
        self.sift = SiftRef(ref_dir, min_inliers=min_inliers, scale=sift_scale)
        self.tracker = make_tracker('vit-trt')
        self.W, self.H = W, H
        self.color_min = color_min; self.score_min = score_min; self.score_low = score_low
        self.color_track_min = color_track_min; self.max_box_ratio = max_box_ratio; self.relost = relost
        self.state = 'SEARCH'; self.bbox = None; self.score = 0.0
        self.locked = False; self.low_cnt = 0; self.drift = 0; self.fi = 0
        self.sift_exec = ThreadPoolExecutor(max_workers=1)   # SIFT 异步,不阻塞主循环
        self.sift_future = None; self.sift_frame = None

    def step(self, frame):
        self.fi += 1
        if self.state == 'SEARCH':
            # 异步 SIFT(SEARCH 态 54ms 不能阻塞主循环,双摄尤其)
            if self.sift_future is None:
                self.sift_frame = frame
                self.sift_future = self.sift_exec.submit(self.sift.locate, frame)
            elif self.sift_future.done():
                try:
                    b, inl = self.sift_future.result()
                except Exception:
                    b, inl = None, 0
                self.sift_future = None
                sf = self.sift_frame
                if b is not None and sf is not None:
                    x, y, w, h = [int(v) for v in b]
                    x = max(0, min(x, self.W - 2)); y = max(0, min(y, self.H - 2))
                    w = max(1, min(w, self.W - x)); h = max(1, min(h, self.H - y))
                    if w >= 20 and h >= 20 and self.sift.last_idx is not None:
                        crop = sf[y:y + h, x:x + w]
                        csim = color_similarity(crop, self.sift.ref_imgs[self.sift.last_idx])
                        if csim >= self.color_min:
                            self.tracker.init(sf, (x, y, w, h))   # 用 SIFT 处理的那帧 init
                            self.bbox = (x, y, w, h); self.state = 'TRACK'
                            self.locked = True; self.low_cnt = 0; self.drift = 0
                        else:
                            self.sift.last_idx = None
        else:  # TRACK
            ok, nb = self.tracker.update(frame)
            self.score = float(self.tracker.getTrackingScore())
            x, y, w, h = [int(v) for v in nb]
            # max-box: 全屏大框=解码失败
            if w > self.W * self.max_box_ratio or h > self.H * self.max_box_ratio:
                self.score = 0.0
            # score 迟滞
            self.locked = self.score >= (self.score_low if self.locked else self.score_min)
            # TRACK 颜色检查(每10帧)
            if self.locked and self.fi % 10 == 0 and self.sift.last_idx is not None:
                cx0, cy0 = max(0, x), max(0, y)
                crop = frame[cy0:cy0 + max(1, h), cx0:cx0 + max(1, w)]
                csim = color_similarity(crop, self.sift.ref_imgs[self.sift.last_idx])
                self.drift = self.drift + 1 if csim < self.color_track_min else 0
                if self.drift >= 2:
                    self.state = 'SEARCH'; self.locked = False; self.drift = 0
            if self.locked:
                self.bbox = (x, y, w, h); self.low_cnt = 0
            else:
                self.low_cnt += 1
                if self.low_cnt >= self.relost:
                    self.state = 'SEARCH'
        return self.state, self.bbox, self.score, self.locked


class FovLocator:
    """实时配准长焦视野在广角的位置:长焦缩到 fov 尺度,在广角 matchTemplate 找。
    异步(15ms 不阻塞主循环)+ EMA 平滑 + conf 门槛(双摄色差时不乱跳)。"""
    def __init__(self, fov_ratio, W, H, match_scale=0.3, conf_min=0.25, alpha=0.3):
        self.fov_w = int(W * fov_ratio); self.fov_h = int(H * fov_ratio)
        self.s = match_scale; self.conf_min = conf_min; self.alpha = alpha
        self.W, self.H = W, H
        self.bx = (W - self.fov_w) // 2; self.by = (H - self.fov_h) // 2   # 初始中心
        self.conf = 0.0
        self.exec = ThreadPoolExecutor(max_workers=1)
        self.future = None

    def _match(self, wf, tf):
        tmpl = cv2.resize(tf, (self.fov_w, self.fov_h))
        s = self.s
        wg = cv2.cvtColor(cv2.resize(wf, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        tg = cv2.cvtColor(cv2.resize(tmpl, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(wg, tg, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        return int(ml[0] / s), int(ml[1] / s), float(mv)

    def update(self, wf, tf):
        if self.future is None:
            self.future = self.exec.submit(self._match, wf.copy(), tf.copy())
        elif self.future.done():
            try:
                bx, by, conf = self.future.result()
            except Exception:
                bx, by, conf = self.bx, self.by, 0.0
            self.future = None
            self.conf = conf
            if conf >= self.conf_min:          # 匹配可信才更新(EMA 平滑防抖)
                a = self.alpha
                self.bx = int((1 - a) * self.bx + a * bx)
                self.by = int((1 - a) * self.by + a * by)
        return self.bx, self.by, self.fov_w, self.fov_h, self.conf


def main():
    ap = argparse.ArgumentParser(description="双摄:长焦主追踪 + 广角辅助(Step1 视野框)")
    ap.add_argument('--tele-sensor', type=int, default=1)
    ap.add_argument('--wide-sensor', type=int, default=0)
    ap.add_argument('--tele-ref', default='testdata/refs_tele_real')
    ap.add_argument('--wide-ref', default='testdata/refs_0519_imx477_wide_boxcrop')
    ap.add_argument('--width', type=int, default=1920)
    ap.add_argument('--height', type=int, default=1080)
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--fov-ratio', type=float, default=0.33, help='长焦视野占广角的比例(焦距比;调到黄框=长焦实际内容)')
    ap.add_argument('--center-x', type=float, default=0.0, help='长焦光轴在广角水平偏移 [-1,1]')
    ap.add_argument('--center-y', type=float, default=0.0, help='长焦光轴在广角垂直偏移 [-1,1] 上正')
    args = ap.parse_args()

    # 双摄并行启动(nvargus 每个 3-5s,串行=6-10s,并行省一半)
    _cams = {}
    def _mk(key, sid):
        _cams[key] = Cam(sid, args.width, args.height, args.fps)
    ths = [threading.Thread(target=_mk, args=(k, s)) for k, s in
           [('tele', args.tele_sensor), ('wide', args.wide_sensor)]]
    for th in ths: th.start()
    for th in ths: th.join()
    tele_cam, wide_cam = _cams['tele'], _cams['wide']
    tele = TrackPipeline('TELE', args.tele_ref, args.width, args.height)
    # 广角找目标(长焦丢时画橡皮筋用);广角箱子小 → scale 大保留特征 + inliers 低
    wide = TrackPipeline('WIDE', args.wide_ref, args.width, args.height, sift_scale=0.5, min_inliers=18)
    # 长焦视野框 = 实时配准(matchTemplate 找长焦画面在广角的位置),fov-ratio 给模板缩放(焦距比)
    fovloc = FovLocator(args.fov_ratio, args.width, args.height)

    cv2.namedWindow("DualCam", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DualCam", 1280, 720)
    print(f"双摄启动: 广角(实时配准长焦视野框 fov={args.fov_ratio}) | 长焦主追踪 | Q退出")
    t0 = time.perf_counter(); fps_now = 0.0; cnt = 0

    while True:
        tf = tele_cam.read(); wf = wide_cam.read()
        if tf is None or wf is None:
            continue
        t_state, t_bbox, t_score, t_locked = tele.step(tf)
        w_state, w_bbox, w_score, w_locked = wide.step(wf)    # 广角找目标(橡皮筋用)
        bx, by, fov_w, fov_h, mconf = fovloc.update(wf, tf)   # 实时配准长焦视野位置

        # ── 左:广角 + 实时长焦视野框(黄,matchTemplate 配准)──
        left = wf.copy()
        cx, cy = bx + fov_w // 2, by + fov_h // 2
        cv2.rectangle(left, (bx, by), (bx + fov_w, by + fov_h), (0, 255, 255), 4)
        cv2.circle(left, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(left, f"TELE FOV  m={mconf:.2f}", (bx + 5, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(left, "WIDE", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        # 广角找到的目标(绿框)
        if w_locked and w_bbox:
            wx, wy, ww_, wh_ = w_bbox
            cv2.rectangle(left, (wx, wy), (wx + ww_, wy + wh_), (0, 255, 0), 2)
        # ── 长焦丢目标 + 广角有目标 → 橡皮筋(长焦视觉中心 ↔ 广角目标)──
        if not t_locked and w_locked and w_bbox:
            wx, wy, ww_, wh_ = w_bbox
            tgt = (wx + ww_ // 2, wy + wh_ // 2)
            cv2.line(left, (cx, cy), tgt, (0, 0, 255), 5)       # 橡皮筋
            cv2.circle(left, tgt, 12, (0, 0, 255), -1)
            cv2.putText(left, "TARGET", (tgt[0] + 14, tgt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(left, "TELE LOST -> guide", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # ── 右:长焦 + 追踪 ──
        right = tf.copy()
        if t_locked and t_bbox:
            x, y, ww, wh = t_bbox
            cv2.rectangle(right, (x, y), (x + ww, y + wh), (0, 255, 0), 3)
        cv2.putText(right, f"TELE  {t_state}  s={t_score:.2f}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        # ── 左右并排(广角 | 长焦,同样大)──
        combo = np.hstack([left, right])
        cv2.putText(combo, f"{fps_now:.0f}fps  fov={args.fov_ratio} cx={args.center_x} cy={args.center_y}",
                    (20, args.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imshow("DualCam", cv2.resize(combo, (args.width, args.height // 2)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cnt += 1
        if cnt % 20 == 0:
            now = time.perf_counter(); fps_now = 20 / (now - t0); t0 = now

    tele_cam.close(); wide_cam.close(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
