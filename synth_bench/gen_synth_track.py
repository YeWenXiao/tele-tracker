#!/usr/bin/env python3
"""gen_synth_track.py — 合成追踪 benchmark 生成器

受控数字沙盘:矢量随机点阵场景 + 多尺度 ref + 视场移动视频流 + 运动耦合畸变 + GT。
目的:隔离变量量化 SIFT/siamese/NCC 的检出率、错锁率、尺度-错锁曲线、运动速度-错锁曲线。

设计要点(对齐讨论):
- 目标=随机色点阵:大尺度独特(好认)/ 小尺度信息塌缩(自然 distractor → 错锁)
- 背景=随机色块(非纯白):小尺度目标淹没其中
- 多尺度 ref:同一目标从小渲染到大(遞進式)
- 视场 30/100 在场景里按轨迹移动 → 目标进出视场
- 运动耦合畸变:模糊核方向/长度 ∝ 视场速度;滚动快门 shear ∝ 速度
- 独立噪声:jitter(帧间随机平移)、drift(缓慢漂移)
- 输出:refs/  video.avi  ground_truth.csv(每帧 GT,固定 seed 可复现)

矢量表示:场景所有元素用归一化场景坐标存储,按需渲染任意分辨率 → 多尺度无损。
"""
import os, csv, argparse, math
import numpy as np
import cv2


# ---------- 矢量场景:归一化坐标(场景单位),按需渲染 ----------
class VectorScene:
    """场景所有元素用 [0, scene_units] 坐标存,render() 投影到像素。"""
    def __init__(self, scene_units, rng):
        self.U = float(scene_units)
        self.rng = rng
        self.bg_blocks = []   # (x, y, w, h, (b,g,r))  背景色块
        self.dots = []        # (x, y, r_units, (b,g,r))  点
        self.target_box = None  # (cx, cy, half) 目标区域(场景单位)

    def build_background(self, mode, block_n):
        if mode == 'white':
            self.bg_blocks = [(0, 0, self.U, self.U, (255, 255, 255))]
            return
        # random_blocks: 铺满随机色块作干扰背景
        self.bg_blocks = [(0, 0, self.U, self.U, (240, 240, 240))]
        for _ in range(block_n):
            w = self.rng.uniform(self.U * 0.05, self.U * 0.2)
            h = self.rng.uniform(self.U * 0.05, self.U * 0.2)
            x = self.rng.uniform(0, self.U - w)
            y = self.rng.uniform(0, self.U - h)
            c = tuple(int(v) for v in self.rng.integers(120, 230, 3))
            self.bg_blocks.append((x, y, w, h, c))

    def build_dots(self, grid_n, dot_frac=0.35):
        """整齐网格,每格一个随机颜色点(高饱和,区别于背景)。"""
        step = self.U / grid_n
        r = step * dot_frac
        for i in range(grid_n):
            for j in range(grid_n):
                cx = (i + 0.5) * step
                cy = (j + 0.5) * step
                # 高饱和随机色(背景是灰,点是彩 → 大尺度可分)
                c = tuple(int(v) for v in self.rng.integers(0, 256, 3))
                self.dots.append((cx, cy, r, c))

    def set_target(self, cx, cy, half):
        self.target_box = (cx, cy, half)

    def render(self, view_x, view_y, view_units, out_px, stretch=None):
        """渲染场景中 [view_x, view_x+view_units]^2 区域到 out_px×out_px BGR。
        stretch=(angle_deg, factor):运动拉扯,圆点画成椭圆(长轴沿运动方向×factor)。"""
        scale = out_px / view_units
        img = np.full((out_px, out_px, 3), 255, np.uint8)
        # 背景块
        for (x, y, w, h, c) in self.bg_blocks:
            x0 = int((x - view_x) * scale); y0 = int((y - view_y) * scale)
            x1 = int((x + w - view_x) * scale); y1 = int((y + h - view_y) * scale)
            if x1 < 0 or y1 < 0 or x0 > out_px or y0 > out_px:
                continue
            cv2.rectangle(img, (max(0, x0), max(0, y0)),
                          (min(out_px, x1), min(out_px, y1)), c, -1)
        # 点
        for (cx, cy, r, c) in self.dots:
            px = (cx - view_x) * scale; py = (cy - view_y) * scale
            pr = max(1, int(r * scale))
            if px < -pr or py < -pr or px > out_px + pr or py > out_px + pr:
                continue
            if stretch is not None:
                ang, factor = stretch
                a = max(1, int(pr * factor))   # 长轴(沿运动方向拉长)
                cv2.ellipse(img, (int(px), int(py)), (a, pr), ang, 0, 360, c, -1)
            else:
                cv2.circle(img, (int(px), int(py)), pr, c, -1)
        return img

    def target_bbox_in_view(self, view_x, view_y, view_units, out_px):
        """目标在当前视场渲染像素里的 bbox,以及是否在视场内。"""
        cx, cy, half = self.target_box
        scale = out_px / view_units
        px = (cx - view_x) * scale; py = (cy - view_y) * scale
        ph = half * scale
        x0, y0, x1, y1 = px - ph, py - ph, px + ph, py + ph
        # 中心在视场内才算 in_fov
        in_fov = (0 <= px < out_px) and (0 <= py < out_px)
        return (x0, y0, x1, y1), in_fov, ph * 2  # bbox, in_fov, 目标像素边长


# ---------- 运动耦合畸变 ----------
def apply_motion_blur(img, vx, vy, k):
    """方向性运动模糊:核方向=速度方向,核长 ∝ 速度大小。"""
    speed = math.hypot(vx, vy)
    L = int(k * speed)
    if L < 2:
        return img
    L = min(L, 41)
    kernel = np.zeros((L, L), np.float32)
    ang = math.atan2(vy, vx)
    cx = (L - 1) / 2.0
    for t in np.linspace(-cx, cx, L * 2):
        x = int(round(cx + t * math.cos(ang)))
        y = int(round(cx + t * math.sin(ang)))
        if 0 <= x < L and 0 <= y < L:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s == 0:
        return img
    kernel /= s
    return cv2.filter2D(img, -1, kernel)


def apply_rolling_shutter(img, vx, vy, k):
    """滚动快门斜切:垂直速度造成逐行水平错切(果冻效应)。"""
    shear = k * vy
    if abs(shear) < 1e-3:
        return img
    h, w = img.shape[:2]
    M = np.float32([[1, shear, -shear * h / 2.0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


# ---------- 闭环双控制器伺服 ----------
class ServoLoop:
    """视场位置 = 识别收敛控制器 + 扰动控制器 的闭环结果。

    - 识别控制器:用 detected 目标像素位置 → 偏离画面中心误差 → 比例拉回(Kp)
      目标出视场或识别失败时,失去输入 → 伺服 hold(只剩扰动 → 视场漂走 = 丢失动态)
    - 扰动控制器:低通滤波随机方向(OU 过程)→ 平滑慢速扰动,看得清
    """
    def __init__(self, scene_units, fov_units, out_px, kp, disturb_amp, disturb_smooth, rng):
        self.U = scene_units; self.fov = fov_units; self.out_px = out_px
        self.kp = kp; self.amp = disturb_amp; self.smooth = disturb_smooth
        self.rng = rng
        self.dist_v = np.zeros(2)          # 扰动速度(低通状态)
        self.px_per_unit = out_px / fov_units

    def step(self, view, detected_px, in_fov):
        """view: 当前视场左上角(场景单位)。detected_px: 识别到的目标画面像素中心或 None。
        返回 (new_view, servo_v, dist_v)。"""
        # 扰动控制器:OU 平滑随机
        self.dist_v = self.smooth * self.dist_v + (1 - self.smooth) * \
            self.rng.normal(0, self.amp, 2)
        # 识别收敛控制器
        servo_v = np.zeros(2)
        if detected_px is not None and in_fov:
            err_px = np.array([detected_px[0] - self.out_px / 2.0,
                               detected_px[1] - self.out_px / 2.0])
            err_scene = err_px / self.px_per_unit
            servo_v = self.kp * err_scene   # 目标在右下 → 视场右下移,把目标拉回中心
        new_view = np.array(view) + servo_v + self.dist_v
        lo, hi = -self.fov * 0.5, self.U - self.fov * 0.5
        new_view = np.clip(new_view, lo, hi)
        return new_view, servo_v, self.dist_v.copy()


def main():
    ap = argparse.ArgumentParser(description="合成追踪 benchmark 生成器")
    ap.add_argument('--out-dir', default='synth_bench/out')
    ap.add_argument('--seed', type=int, default=42)
    # 场景
    ap.add_argument('--scene-units', type=float, default=500.0)
    ap.add_argument('--grid-n', type=int, default=120, help='点阵网格密度 NxN')
    ap.add_argument('--bg-mode', choices=['white', 'random_blocks'], default='random_blocks')
    ap.add_argument('--bg-block-n', type=int, default=200)
    # 目标
    ap.add_argument('--target-units', type=float, default=50.0, help='目标逻辑边长(场景单位)')
    # 视场
    ap.add_argument('--fov-units', type=float, default=150.0)
    ap.add_argument('--out-px', type=int, default=480, help='输出帧像素')
    # 闭环双控制器
    ap.add_argument('--servo-kp', type=float, default=0.30, help='识别收敛增益(0=纯扰动开环)')
    ap.add_argument('--disturb-amp', type=float, default=1.5, help='扰动幅度(场景单位/帧)')
    ap.add_argument('--disturb-smooth', type=float, default=0.92, help='扰动平滑度(越大越平滑)')
    ap.add_argument('--detect-noise-px', type=float, default=0.0, help='识别位置噪声(模拟不完美识别)')
    # 多尺度 ref
    ap.add_argument('--ref-px-min', type=int, default=24, help='ref 最小渲染像素(远/小)')
    ap.add_argument('--ref-px-max', type=int, default=240, help='ref 最大渲染像素(近/大)')
    ap.add_argument('--ref-count', type=int, default=10, help='遞進 ref 档数')
    # 视频
    ap.add_argument('--n-frames', type=int, default=300)
    ap.add_argument('--fps', type=int, default=30)
    # 运动耦合畸变
    ap.add_argument('--stretch-k', type=float, default=0.0, help='运动拉扯:圆点沿运动方向拉成椭圆,长轴∝速度,0=关')
    ap.add_argument('--motion-blur-k', type=float, default=0.0, help='模糊核长/速度,0=关')
    ap.add_argument('--rolling-shutter-k', type=float, default=0.0, help='shear/速度,0=关')
    # 独立噪声
    ap.add_argument('--jitter-px', type=float, default=0.0, help='帧间随机平移像素')
    ap.add_argument('--drift-units', type=float, default=0.0, help='缓慢漂移幅度(场景单位)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    ref_dir = os.path.join(args.out_dir, 'refs')
    os.makedirs(ref_dir, exist_ok=True)

    # 1. 建场景
    scene = VectorScene(args.scene_units, rng)
    scene.build_background(args.bg_mode, args.bg_block_n)
    scene.build_dots(args.grid_n)
    half = args.target_units / 2.0
    tcx, tcy = args.scene_units / 2.0, args.scene_units / 2.0  # 目标放场景中心
    scene.set_target(tcx, tcy, half)

    # 2. 多尺度 ref:同一目标区域渲染到不同像素(遞進)
    ref_pxs = np.linspace(args.ref_px_min, args.ref_px_max, args.ref_count).astype(int)
    ref_meta = []
    for idx, rpx in enumerate(ref_pxs):
        img = scene.render(tcx - half, tcy - half, args.target_units, int(rpx))
        name = f'ref_{idx:02d}_px{rpx}.png'
        cv2.imwrite(os.path.join(ref_dir, name), img)
        ref_meta.append((idx, int(rpx)))
    print(f"[REF] 生成 {len(ref_pxs)} 档遞進 ref: {args.ref_px_min}px → {args.ref_px_max}px → {ref_dir}")

    # 3. 闭环伺服 + 渲染视频 + GT
    servo = ServoLoop(args.scene_units, args.fov_units, args.out_px,
                      args.servo_kp, args.disturb_amp, args.disturb_smooth, rng)
    # 初始:视场中心对准目标
    view = np.array([tcx - args.fov_units / 2.0, tcy - args.fov_units / 2.0])
    video_path = os.path.join(args.out_dir, 'video.avi')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'),
                             args.fps, (args.out_px, args.out_px))
    gt_path = os.path.join(args.out_dir, 'ground_truth.csv')
    gt_f = open(gt_path, 'w', newline='')
    gt = csv.writer(gt_f)
    gt.writerow(['frame', 'view_x', 'view_y', 'vx', 'vy', 'speed',
                 'servo_x', 'servo_y', 'disturb_x', 'disturb_y',
                 'target_in_fov', 'tgt_px_cx', 'tgt_px_cy', 'tgt_px_size',
                 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1'])

    prev = None
    for t in range(args.n_frames):
        # 当前视场下目标的 GT 像素位置(识别控制器的输入,演示阶段=完美识别)
        bbox0, in_fov0, _ = scene.target_bbox_in_view(
            view[0], view[1], args.fov_units, args.out_px)
        det_px = None
        if in_fov0:
            dcx = (bbox0[0] + bbox0[2]) / 2; dcy = (bbox0[1] + bbox0[3]) / 2
            if args.detect_noise_px > 0:
                dcx += rng.normal(0, args.detect_noise_px)
                dcy += rng.normal(0, args.detect_noise_px)
            det_px = (dcx, dcy)

        # 闭环:识别收敛 + 扰动 → 新视场
        new_view, servo_v, dist_v = servo.step(view, det_px, in_fov0)

        # 速度(场景单位 → 像素/帧),用于运动耦合畸变
        if prev is None:
            vx = vy = 0.0
        else:
            scale = args.out_px / args.fov_units
            vx = (new_view[0] - prev[0]) * scale
            vy = (new_view[1] - prev[1]) * scale
        prev = new_view.copy()
        view = new_view
        view_x, view_y = view[0], view[1]
        speed = math.hypot(vx, vy)

        # 运动拉扯:圆点沿运动方向拉成椭圆,拉伸量 ∝ 速度
        stretch = None
        if args.stretch_k > 0 and speed > 0.1:
            ang = math.degrees(math.atan2(vy, vx))
            stretch = (ang, 1.0 + args.stretch_k * speed)
        img = scene.render(view_x, view_y, args.fov_units, args.out_px, stretch=stretch)
        # 运动耦合畸变
        if args.motion_blur_k > 0:
            img = apply_motion_blur(img, vx, vy, args.motion_blur_k)
        if args.rolling_shutter_k > 0:
            img = apply_rolling_shutter(img, vx, vy, args.rolling_shutter_k)

        bbox, in_fov, tsize = scene.target_bbox_in_view(
            view_x, view_y, args.fov_units, args.out_px)
        cen = args.out_px // 2
        # 画面中心十字(伺服参考点)
        cv2.drawMarker(img, (cen, cen), (255, 0, 0), cv2.MARKER_CROSS, 28, 2)
        # GT 框 + 中心点(绿=在视场)
        if in_fov:
            pcx = (bbox[0] + bbox[2]) / 2; pcy = (bbox[1] + bbox[3]) / 2
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.circle(img, (int(pcx), int(pcy)), 4, (0, 255, 0), -1)
            # error 线:画面中心 → 目标(伺服要消除的偏差,橡皮筋)
            cv2.line(img, (cen, cen), (int(pcx), int(pcy)), (0, 255, 255), 2)
        else:
            pcx = pcy = -1
        dmag = math.hypot(dist_v[0], dist_v[1]); smag = math.hypot(servo_v[0], servo_v[1])
        cv2.putText(img, f"f{t} {'IN' if in_fov else 'OUT'} sz={tsize:.0f} spd={speed:.0f}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, f"disturb={dmag:.1f}  servo={smag:.1f}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 80, 0), 2)
        writer.write(img)

        gt.writerow([t, f'{view_x:.3f}', f'{view_y:.3f}', f'{vx:.2f}', f'{vy:.2f}',
                     f'{speed:.2f}', f'{servo_v[0]:.3f}', f'{servo_v[1]:.3f}',
                     f'{dist_v[0]:.3f}', f'{dist_v[1]:.3f}',
                     int(in_fov), f'{pcx:.1f}', f'{pcy:.1f}', f'{tsize:.1f}',
                     f'{bbox[0]:.1f}', f'{bbox[1]:.1f}', f'{bbox[2]:.1f}', f'{bbox[3]:.1f}'])

    writer.release(); gt_f.close()
    print(f"[VIDEO] {video_path}  ({args.n_frames} 帧 @ {args.fps}fps, {args.out_px}px)")
    print(f"[GT]    {gt_path}")
    print(f"[DONE]  seed={args.seed} 可复现。下一步:评测器读 GT + 跑算法出指标")


if __name__ == '__main__':
    main()
