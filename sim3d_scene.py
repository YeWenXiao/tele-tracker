#!/usr/bin/env python3
"""sim3d_scene.py — 数字 3D 场景 + 自由移动相机(前后左右)。
世界: 地面 + 后墙 + 柱子 + GOOD LUCK 箱子(真实纹理贴面) + 红色干扰柜。
渲染: 平面贴片透视投影(每个面=一次 warpPerspective),画家算法排序。
坐标系: X 右, Y 上, Z 前(米)。相机 pos=(x,y,z) + yaw(弧度,左正)。
main: 生成相机前/后/左/右移动的效果视频。
"""
import cv2, numpy as np


# ── 纹理 ─────────────────────────────────────────────
def tex_ground(px_per_m=60, size_m=40):
    """地面:浅灰反光地砖 + 缝隙网格(贴近现场大厅)"""
    n = px_per_m * size_m
    t = np.full((n, n, 3), 195, np.uint8)
    rng = np.random.RandomState(5)
    t = (t.astype(np.int16) + rng.randint(-6, 6, t.shape)).clip(0, 255).astype(np.uint8)
    step = px_per_m * 2          # 2m 一块砖
    for i in range(0, n, step):
        t[i:i+2, :] = 150
        t[:, i:i+2] = 150
    return t


def tex_wall(w_m=40, h_m=4, px=50):
    """白墙 + 蓝色横幅色块(类似现场)"""
    w, h = w_m * px, h_m * px
    t = np.full((h, w, 3), 235, np.uint8)
    cv2.rectangle(t, (w//6, h//5), (w//6 + w//5, h//5 + h//4), (200, 150, 60), -1)
    cv2.rectangle(t, (w//2, h//6), (w//2 + w//4, h//6 + h//5), (180, 120, 40), -1)
    cv2.rectangle(t, (0, h - px//2), (w, h), (120, 120, 120), -1)   # 踢脚线
    return t


def tex_pillar(px=120):
    t = np.full((px*4, px, 3), 215, np.uint8)
    t[:, :6] = 170; t[:, -6:] = 170
    return t


def tex_cabinet(px=140):
    """红色消防柜干扰物:红底白字条"""
    t = np.full((px*2, px, 3), (40, 40, 200), np.uint8)
    for i in range(4):
        cv2.rectangle(t, (px//6, px//3 + i*px//3), (px - px//6, px//3 + i*px//3 + px//8), (230, 230, 230), -1)
    cv2.rectangle(t, (0, 0), (px-1, px*2-1), (30, 30, 150), 6)
    return t


# ── 世界(四边形面片集合)──────────────────────────────
class Quad:
    def __init__(self, corners, tex, double=False):
        self.c = np.float32(corners)       # 4 个世界坐标角点(左上,右上,右下,左下 顺时针看向正面)
        self.tex = tex
        self.double = double               # 双面(地面/墙)
        e1, e2 = self.c[1] - self.c[0], self.c[3] - self.c[0]
        self.normal = np.cross(e1, e2)
        self.normal /= (np.linalg.norm(self.normal) + 1e-9)
        self.center = self.c.mean(axis=0)


def build_world(box_size=0.5, box_pos=(0.0, 6.0)):
    """场景:地面 y=0(特殊无限平面),后墙 z=12,箱子立在 (box_pos.x, z=box_pos.y),柱子,干扰柜"""
    quads = []
    w = tex_wall()
    quads.append(Quad([(-20, 4, 12), (20, 4, 12), (20, 0, 12), (-20, 0, 12)], w, double=True))
    # 柱子(两侧)
    p = tex_pillar()
    for px_, pz in [(-5, 10), (5, 10), (-7, 4), (7, 4)]:
        quads.append(Quad([(px_-0.4, 3, pz), (px_+0.4, 3, pz), (px_+0.4, 0, pz), (px_-0.4, 0, pz)], p, double=True))
    # 红色干扰柜(贴墙)
    quads.append(Quad([(3.4, 1.9, 11.95), (4.6, 1.9, 11.95), (4.6, 0, 11.95), (3.4, 0, 11.95)], tex_cabinet(), double=True))
    # GOOD LUCK 箱子(立方体,正面=真实纹理,其余暗红)
    bx, bz = box_pos
    s = box_size
    face = cv2.imread('testdata/refs_field_0610/field_10.jpg')
    face = cv2.resize(face, (400, 400))
    side = np.full((400, 400, 3), (35, 35, 160), np.uint8)
    top = np.full((400, 400, 3), (30, 30, 130), np.uint8)
    x0, x1, y0, y1, z0, z1 = bx - s/2, bx + s/2, 0, s, bz - s/2, bz + s/2
    quads.append(Quad([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], face))            # 前(朝-Z)
    quads.append(Quad([(x1, y1, z1), (x0, y1, z1), (x0, y0, z1), (x1, y0, z1)], side))            # 后
    quads.append(Quad([(x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (x0, y0, z1)], side))            # 左
    quads.append(Quad([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], side))            # 右
    quads.append(Quad([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], top))             # 顶
    return quads


# ── 渲染 ─────────────────────────────────────────────
_GROUND_TEX = None
_G_PXM = 60        # 地面纹理 px/米
_G_X0, _G_Z0 = -40.0, -40.0   # 纹理左上对应的世界坐标(覆盖 x:-40~40, z:-40~40)


def _draw_ground(frame, C, R, yaw, focal, W, H):
    """地面特殊渲染:可见区段跟随相机朝向构造(角点永在前方,任意 yaw 不消失);
    纹理坐标由世界坐标直接映射,大纹理一次 warp 上屏。"""
    global _GROUND_TEX
    if _GROUND_TEX is None:
        _GROUND_TEX = tex_ground(_G_PXM, 80)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    ax = np.float32([cy_, 0, -sy_])      # 相机 x 轴(世界系)
    az = np.float32([sy_, 0, cy_])       # 相机 z 轴(世界系)
    base = np.float32([C[0], 0, C[2]])
    corners = []
    for dx, dz in [(-34, 36), (34, 36), (34, 0.3), (-34, 0.3)]:   # 相机系矩形
        p = base + ax * dx + az * dz
        corners.append(p)
    corners = np.float32(corners)
    pc = (corners - C) @ R.T
    u = focal * pc[:, 0] / pc[:, 2] + W / 2
    v = -focal * pc[:, 1] / pc[:, 2] + H / 2
    dst = np.float32(list(zip(u, v)))
    n = _GROUND_TEX.shape[0]
    tex_pts = np.float32([[np.clip((p[0] - _G_X0) * _G_PXM, 0, n - 1),
                           np.clip((p[2] - _G_Z0) * _G_PXM, 0, n - 1)] for p in corners])
    M = cv2.getPerspectiveTransform(tex_pts, dst)
    warped = cv2.warpPerspective(_GROUND_TEX, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((n, n), 255, np.uint8), M, (W, H))
    mask[:H // 2 + 1] = 0     # 地面只存在于地平线以下(裁掉平面单应的镜像翻折)
    frame[mask > 0] = warped[mask > 0]


def render_view(quads, cam_pos, yaw, focal, W, H, sky=(60, 55, 50)):
    """相机 pos=(x,y,z) yaw(左正,绕Y) → 1 帧。focal 大=长焦。"""
    C = np.float32(cam_pos)
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    R = np.float32([[cy_, 0, -sy_], [0, 1, 0], [sy_, 0, cy_]])     # world→cam
    frame = np.full((H, W, 3), sky, np.uint8)
    _draw_ground(frame, C, R, yaw, focal, W, H)      # 地面最先画(最远层)

    def project(pts):
        pc = (pts - C) @ R.T
        return pc

    order = sorted(quads, key=lambda q: -np.linalg.norm(q.center - C))
    for q in order:
        if not q.double:                       # 背面剔除
            if np.dot(q.normal, q.center - C) > 0:
                continue
        pc = project(q.c)
        if (pc[:, 2] < 0.15).any():            # 任一角点过近/在身后 → 简化跳过(场景设计保证主体不触发)
            continue
        u = focal * pc[:, 0] / pc[:, 2] + W / 2
        v = -focal * pc[:, 1] / pc[:, 2] + H / 2
        dst = np.float32(list(zip(u, v)))
        if (dst[:, 0].max() < 0 or dst[:, 0].min() > W or
                dst[:, 1].max() < 0 or dst[:, 1].min() > H):
            continue                            # 完全出画
        th, tw = q.tex.shape[:2]
        src = np.float32([(0, 0), (tw, 0), (tw, th), (0, th)])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(q.tex, M, (W, H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full((th, tw), 255, np.uint8), M, (W, H))
        frame[mask > 0] = warped[mask > 0]
    return frame


# ── 相机移动效果视频 ──────────────────────────────────
def main():
    quads = build_world(box_size=0.5, box_pos=(0.0, 6.0))
    W, H = 1280, 720
    focal = 900.0
    OUT = 'algo_limit_tests/sim3d_camera_demo.avi'
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'MJPG'), 30, (W, H))

    def path(n):
        """前 → 后 → 左 → 右 → 环视。返回 (pos, yaw, 段名)。起步 -14m(距箱 20m 远视距)"""
        if n < 180:                                  # 前进 z -14→4(距箱 20m→2m)
            return (0, 0.8, -14 + 18 * n / 179), 0.0, 'FORWARD'
        if n < 300:                                  # 后退 4→-14
            return (0, 0.8, 4 - 18 * (n - 180) / 119), 0.0, 'BACKWARD'
        if n < 420:                                  # 左移 x 0→-3(在 -2m 处,距箱 8m)
            return (-3 * (n - 300) / 119, 0.8, -2), 0.0, 'STRAFE LEFT'
        if n < 540:                                  # 右移 -3→+3
            return (-3 + 6 * (n - 420) / 119, 0.8, -2), 0.0, 'STRAFE RIGHT'
        if n < 630:                                  # 回中 +3→0 顺带轻微转头
            t = (n - 540) / 89
            return (3 * (1 - t), 0.8, -2), 0.25 * np.sin(t * np.pi), 'PAN'
        return (0, 0.8, -2), 0.0, "END"

    import time
    t0 = time.time()
    for n in range(630):
        pos, yaw, label = path(n)
        f = render_view(quads, pos, yaw, focal, W, H)
        cv2.putText(f, f"{label}  cam=({pos[0]:+.1f},{pos[2]:+.1f})m yaw={yaw:+.2f}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(f, f"f{n}", (15, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        vw.write(f)
    vw.release()
    print(f"630 帧 {time.time()-t0:.0f}s → {OUT}")


if __name__ == '__main__':
    main()
