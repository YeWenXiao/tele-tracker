#!/usr/bin/env python3
"""scene3d.py — 3D box 渲染器(模拟视差/自遮挡/透视)

核心 sim-to-real:ref 是 2D 正面投影,但目标是 3D 立方体。
相机绕转 → 露出侧面、透视形变、自遮挡 → 2D ref 匹配不上 = 真实视差杀伤力。

Box3D.render(azimuth, elevation, distance, out_px, focal):
  立方体 8 顶点 → 旋转 → 透视投影 → 背面剔除 → 画家算法深度排序 → warpPerspective 贴 6 面 pattern。
"""
import os, argparse, csv, math, json
import numpy as np
import cv2


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


class Box3D:
    # 立方体 8 顶点(±1),6 面(顶点索引,逆时针朝外),面法线
    V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64)
    FACES = [(0, 1, 2, 3),   # -z 正面(ref 面)
             (5, 4, 7, 6),   # +z 背面
             (4, 0, 3, 7),   # -x 左
             (1, 5, 6, 2),   # +x 右
             (4, 5, 1, 0),   # -y 下
             (3, 2, 6, 7)]   # +y 上
    NORMALS = np.array([[0, 0, -1], [0, 0, 1], [-1, 0, 0],
                        [1, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64)

    def __init__(self, size, face_imgs):
        self.h = size / 2.0
        self.faces_img = face_imgs  # 6 张 BGR pattern

    def render(self, az, el, dist, out_px, focal, bg=(40, 40, 40),
               shade=0.0, light=(0.3, -0.5, -1.0)):
        R = _rot_y(az) @ _rot_x(el)
        verts = (R @ (self.V * self.h).T).T + np.array([0, 0, dist])
        cx = cy = out_px / 2.0
        # 透视投影(z>0 前方)
        proj = np.zeros((8, 2))
        for i, v in enumerate(verts):
            z = max(1e-3, v[2])
            proj[i] = [focal * v[0] / z + cx, focal * v[1] / z + cy]
        normals_cam = (R @ self.NORMALS.T).T
        ld = np.array(light, np.float64); ld /= (np.linalg.norm(ld) + 1e-9)
        canvas = np.full((out_px, out_px, 3), bg, np.uint8)
        order = []
        for fi, face in enumerate(self.FACES):
            cen = verts[list(face)].mean(0)
            if np.dot(normals_cam[fi], cen) >= 0:   # 背面剔除
                continue
            order.append((cen[2], fi))
        order.sort(reverse=True)   # 画家算法
        for _, fi in order:
            face = self.FACES[fi]
            dst = proj[list(face)].astype(np.float32)
            img = self.faces_img[fi]
            # Lambert 光照:亮度随面法线与光源夹角(shade 强度可调)
            if shade > 0:
                lam = max(0.0, -float(np.dot(normals_cam[fi], ld)))
                bright = (1 - shade) + shade * lam
                img = np.clip(img.astype(np.float32) * bright, 0, 255).astype(np.uint8)
            H, W = img.shape[:2]
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            M = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(img, M, (out_px, out_px))
            mask = np.zeros((out_px, out_px), np.uint8)
            cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
            canvas[mask > 0] = warp[mask > 0]
        x0, y0 = proj.min(0); x1, y1 = proj.max(0)
        return canvas, (x0, y0, x1, y1), proj


def make_box_faces(theme, px, rng):
    """造一个 box 的 6 面 pattern。theme 决定正面外观(目标/各类干扰)。"""
    def base(color, structured=False, alt=False):
        img = np.full((px, px, 3), color, np.uint8)
        if structured:
            # 正面:色块 + 条纹(像箱面);alt=换位置(同色异构干扰)
            offs = [0.15, 0.43, 0.71] if not alt else [0.25, 0.58]
            for x_frac in offs:
                x = int(px * x_frac)
                cv2.rectangle(img, (x, int(px * 0.3)), (x + int(px * 0.15), int(px * 0.6)),
                              (20, 20, 20), -1)
            cv2.rectangle(img, (0, int(px * 0.75)), (px, int(px * 0.85)), (200, 200, 200), -1)
        else:
            for _ in range(8):
                p = rng.integers(0, px, 4)
                cv2.line(img, (p[0], p[1]), (p[2], p[3]), (30, 30, 30), 2)
        cv2.rectangle(img, (2, 2), (px - 2, px - 2), (10, 10, 10), 3)
        return img
    side = [base((140, 110, 70)), base((70, 150, 70)), base((160, 160, 70)), base((90, 90, 90))]
    if theme == 'target':       # 目标:红底标准结构
        front = base((40, 40, 200), structured=True)
    elif theme == 'redalt':     # 同色异构:同红底,结构不同(最迷惑)
        front = base((40, 40, 200), structured=True, alt=True)
    elif theme == 'blue':       # 异色:蓝底同结构
        front = base((200, 80, 40), structured=True)
    elif theme == 'green':
        front = base((40, 160, 40), structured=True)
    else:
        front = base((120, 120, 120))
    return [front, side[0], side[1], side[2], side[3], side[3]]


class Scene3D:
    """多 box 场景:1 目标 + N 干扰,统一相机变换 + 全局深度排序(视差)。"""
    def __init__(self):
        self.boxes = []   # (Box3D, world_pos(3,), is_target)
    def add(self, box, pos, is_target=False):
        self.boxes.append((box, np.array(pos, np.float64), is_target))
    def render(self, az, el, dist, out_px, focal, shade=0.0, light=(0.3, -0.5, -1.0)):
        R = _rot_y(az) @ _rot_x(el)
        cx = cy = out_px / 2.0
        ld = np.array(light, np.float64); ld /= (np.linalg.norm(ld) + 1e-9)
        canvas = np.full((out_px, out_px, 3), (40, 40, 40), np.uint8)
        draw = []   # (depth_z, dst_pts, img)
        tgt_front = None; tgt_vis = 0
        regions = []   # (x0,y0,x1,y1,is_target) 每个 box 整体投影区(判错锁用)
        for box, wpos, is_t in self.boxes:
            local = box.V * box.h + wpos
            vcam = (R @ local.T).T + np.array([0, 0, dist])
            proj = np.zeros((8, 2))
            for i, v in enumerate(vcam):
                z = max(1e-3, v[2]); proj[i] = [focal * v[0] / z + cx, focal * v[1] / z + cy]
            bx0, by0 = proj.min(0); bx1, by1 = proj.max(0)
            regions.append((float(bx0), float(by0), float(bx1), float(by1), int(is_t)))
            ncam = (R @ box.NORMALS.T).T
            for fi, face in enumerate(box.FACES):
                cen = vcam[list(face)].mean(0)
                if np.dot(ncam[fi], cen) >= 0:
                    continue
                img = box.faces_img[fi]
                if shade > 0:
                    lam = max(0.0, -float(np.dot(ncam[fi], ld)))
                    img = np.clip(img.astype(np.float32) * ((1 - shade) + shade * lam),
                                  0, 255).astype(np.uint8)
                draw.append((cen[2], proj[list(face)].astype(np.float32), img))
                if is_t and fi == 0:
                    tgt_front = proj[list(face)]; tgt_vis = 1
        draw.sort(key=lambda d: -d[0])   # 远先画
        for _, dst, img in draw:
            H, W = img.shape[:2]
            src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
            M = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(img, M, (out_px, out_px))
            mask = np.zeros((out_px, out_px), np.uint8)
            cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
            canvas[mask > 0] = warp[mask > 0]
        if tgt_front is not None:
            x0, y0 = tgt_front.min(0); x1, y1 = tgt_front.max(0)
            return canvas, (float(x0), float(y0), float(x1), float(y1)), tgt_vis, regions
        return canvas, (-1, -1, -1, -1), 0, regions


def make_face(kind, px, rng):
    """造 6 个面 pattern。正面=可识别目标面(红底+字符),其余面不同色/纹理。"""
    img = np.full((px, px, 3), 245, np.uint8)
    if kind == 'front':   # 红底 + 黑字符块 + 条纹(像 GOOD LUCK 箱面)
        img[:] = (40, 40, 200)  # 红 (BGR)
        for i in range(3):
            x = int(px * (0.15 + i * 0.28))
            cv2.rectangle(img, (x, int(px * 0.3)), (x + int(px * 0.15), int(px * 0.6)),
                          (20, 20, 20), -1)
        cv2.rectangle(img, (0, int(px * 0.75)), (px, int(px * 0.85)), (200, 200, 200), -1)
        cv2.rectangle(img, (2, 2), (px - 2, px - 2), (10, 10, 10), 3)
    else:
        base = {'back': (180, 120, 60), 'left': (60, 160, 60), 'right': (160, 160, 60),
                'top': (200, 200, 200), 'bot': (90, 90, 90)}.get(kind, (120, 120, 120))
        img[:] = base
        for _ in range(8):
            p = rng.integers(0, px, 4)
            cv2.line(img, (p[0], p[1]), (p[2], p[3]), (30, 30, 30), 2)
        cv2.rectangle(img, (2, 2), (px - 2, px - 2), (10, 10, 10), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="3D box 视差 demo")
    ap.add_argument('--out-dir', default='synth_bench/out3d')
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--out-px', type=int, default=480)
    ap.add_argument('--focal', type=float, default=500.0)
    ap.add_argument('--size', type=float, default=2.0)
    ap.add_argument('--n-frames', type=int, default=240)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--sweep-mode', choices=['sine', 'az_mono'], default='sine',
                    help='sine=综合摆动; az_mono=纯视角单调扫描(隔离视角变量)')
    ap.add_argument('--az-deg', type=float, default=75.0, help='方位摆幅/最大角(度)')
    ap.add_argument('--el-deg', type=float, default=25.0, help='俯仰摆幅(度)')
    ap.add_argument('--dist-min', type=float, default=5.0)
    ap.add_argument('--dist-max', type=float, default=11.0)
    # 真实感强度(可调)
    ap.add_argument('--n-confusers', type=int, default=0, help='干扰 box 数量(0=只目标)')
    ap.add_argument('--shading', type=float, default=0.5, help='光照明暗强度 0=平光 1=强')
    ap.add_argument('--noise-sigma', type=float, default=0.0, help='高斯噪声 sigma(像素)')
    ap.add_argument('--blur-k', type=float, default=0.0, help='运动模糊系数(耦合角速度)')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dist_mid = (args.dist_min + args.dist_max) / 2

    # 目标 box + ref(目标正面 crop,无背景无干扰)
    target_box = Box3D(args.size, make_box_faces('target', 256, rng))
    ref_full, _, ref_proj = target_box.render(0, 0, dist_mid, args.out_px,
                                              args.focal, shade=args.shading)
    fp = ref_proj[list(Box3D.FACES[0])]
    rx0, ry0 = fp.min(0); rx1, ry1 = fp.max(0)
    cv2.imwrite(os.path.join(args.out_dir, 'ref_front.png'),
                ref_full[int(ry0):int(ry1), int(rx0):int(rx1)])

    # 场景 = 目标(原点) + N 干扰(周围,类似但不同)
    scene = Scene3D()
    scene.add(target_box, (0, 0, 0), True)
    conf_specs = [('redalt', (-3.5, 0.5, 1.5)), ('blue', (3.5, -0.3, -1.0)),
                  ('green', (0.6, -3.2, 2.0)), ('graytex', (-2.6, 2.9, -0.5))]
    for i in range(min(args.n_confusers, len(conf_specs))):
        th, pos = conf_specs[i]
        scene.add(Box3D(args.size, make_box_faces(th, 256, rng)), pos, False)

    video = os.path.join(args.out_dir, 'video.avi')
    wr = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'MJPG'),
                         args.fps, (args.out_px, args.out_px))
    gt_f = open(os.path.join(args.out_dir, 'ground_truth.csv'), 'w', newline='')
    gt = csv.writer(gt_f)
    gt.writerow(['frame', 'az_deg', 'el_deg', 'dist', 'front_visible',
                 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'boxes_json'])

    for t in range(args.n_frames):
        if args.sweep_mode == 'az_mono':
            az = math.radians(args.az_deg) * t / max(1, args.n_frames - 1)
            el = 0.0; dist = dist_mid
        else:
            ph = 2 * math.pi * t / args.n_frames
            az = math.radians(args.az_deg) * math.sin(ph)
            el = math.radians(args.el_deg) * math.sin(2 * ph)
            dist = dist_mid + (args.dist_max - args.dist_min) / 2 * math.sin(ph * 0.7)
        img, bbox, vis, regions = scene.render(az, el, dist, args.out_px, args.focal, shade=args.shading)
        if args.noise_sigma > 0:
            img = np.clip(img.astype(np.float32) +
                          rng.normal(0, args.noise_sigma, img.shape), 0, 255).astype(np.uint8)
        if vis:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                          (0, 255, 0), 2)
            cv2.putText(img, 'TARGET', (int(bbox[0]), max(12, int(bbox[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"f{t} az={math.degrees(az):.0f} d={dist:.1f} "
                    f"conf={args.n_confusers} tgt={'Y' if vis else 'N'}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        wr.write(img)
        gt.writerow([t, f'{math.degrees(az):.1f}', f'{math.degrees(el):.1f}',
                     f'{dist:.2f}', vis,
                     f'{bbox[0]:.1f}', f'{bbox[1]:.1f}', f'{bbox[2]:.1f}', f'{bbox[3]:.1f}',
                     json.dumps([[round(v, 1) for v in r[:4]] + [r[4]] for r in regions])])

    wr.release(); gt_f.close()
    print(f"[REF]   目标正面 2D 模板 → {args.out_dir}/ref_front.png")
    print(f"[VIDEO] 3D 转动 + {args.n_confusers} 干扰 → {video}  ({args.n_frames} 帧)")
    print(f"[GT]    {args.out_dir}/ground_truth.csv")
    print("  看点:画面里目标(绿框) + 干扰 box,转动产生视差,看算法能否在干扰中认出目标")


if __name__ == '__main__':
    main()
