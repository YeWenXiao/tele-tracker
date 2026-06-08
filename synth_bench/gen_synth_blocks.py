#!/usr/bin/env python3
"""gen_synth_blocks.py — 2D 富纹理点阵场景 + 干扰块(对照 3D 大色块 box)

目标 = 随机彩色点阵块(SIFT 友好,富特征);干扰 = 目标的扰动变体(同色翻转/打乱)。
背景随机点,视场 sine 平移扫过目标+干扰。输出格式兼容 viz3d / eval3d。
对照实验:富纹理目标下各模型表现 vs 3D 大色块 box 场景。
"""
import os, sys, csv, json, math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_confusers import make_target_pattern, perturb, render_cell


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'synth_bench/out2d_conf'
    n_conf = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(5)
    SCENE, FOV, OUT, BLOCK, K, NF = 1400, 640, 480, 170, 8, 240

    # 背景:铺满随机彩色点
    canvas = np.full((SCENE, SCENE, 3), 240, np.uint8)
    for y in range(10, SCENE, 22):
        for x in range(10, SCENE, 22):
            c = tuple(int(v) for v in rng.integers(0, 256, 3))
            cv2.circle(canvas, (x, y), 5, c, -1)

    # 目标 + 干扰块(干扰=目标扰动变体)
    target = make_target_pattern(K, rng)
    cx0, cy0 = SCENE // 2, SCENE // 2
    blocks = [(cx0, cy0, 1, render_cell(target, BLOCK))]
    specs = [('color_flip', 0.2), ('shuffle', 0.3), ('color_flip', 0.5), ('copy', 1.0)]
    poss = [(cx0 - 380, cy0 - 230), (cx0 + 390, cy0 + 170),
            (cx0 - 240, cy0 + 360), (cx0 + 280, cy0 - 330)]
    for i in range(min(n_conf, len(specs))):
        pat, _ = perturb(target, specs[i][0], specs[i][1], rng)
        blocks.append((poss[i][0], poss[i][1], 0, render_cell(pat, BLOCK)))
    # 贴块到场景
    for bx, by, _, bimg in blocks:
        h, w = bimg.shape[:2]
        canvas[by - h // 2:by + h // 2, bx - w // 2:bx + w // 2] = bimg

    # ref = 目标块(纯目标,无背景)
    cv2.imwrite(os.path.join(out, 'ref_front.png'), render_cell(target, BLOCK))

    video = os.path.join(out, 'video.avi')
    wr = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (OUT, OUT))
    gt_f = open(os.path.join(out, 'ground_truth.csv'), 'w', newline='')
    gt = csv.writer(gt_f)
    gt.writerow(['frame', 'az_deg', 'el_deg', 'dist', 'front_visible',
                 'bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1', 'boxes_json'])

    scale = OUT / FOV
    half_block_px = BLOCK / 2 * scale
    lo, hi = 0, SCENE - FOV
    mid = (lo + hi) / 2; amp = (hi - lo) / 2
    for t in range(NF):
        ph = 2 * math.pi * t / NF
        vx = mid + amp * math.sin(ph)
        vy = mid + amp * 0.6 * math.sin(2 * ph)
        crop = canvas[int(vy):int(vy) + FOV, int(vx):int(vx) + FOV]
        frame = cv2.resize(crop, (OUT, OUT))
        # 各块在视场里的 bbox
        regions = []; tgt_bbox = (-1, -1, -1, -1); tgt_vis = 0
        for bx, by, ist, _ in blocks:
            px = (bx - vx) * scale; py = (by - vy) * scale
            r = (px - half_block_px, py - half_block_px,
                 px + half_block_px, py + half_block_px)
            # 块中心在视场内才算
            if 0 <= px < OUT and 0 <= py < OUT:
                regions.append([round(r[0], 1), round(r[1], 1),
                                round(r[2], 1), round(r[3], 1), ist])
                if ist:
                    tgt_bbox = r; tgt_vis = 1
        gt.writerow([t, '0', '0', '8', tgt_vis,
                     f'{tgt_bbox[0]:.1f}', f'{tgt_bbox[1]:.1f}',
                     f'{tgt_bbox[2]:.1f}', f'{tgt_bbox[3]:.1f}', json.dumps(regions)])
        wr.write(frame)
    wr.release(); gt_f.close()
    print(f"[REF]   目标点阵块 → {out}/ref_front.png")
    print(f"[VIDEO] 2D 点阵场景 + {n_conf} 干扰块 → {video} ({NF} 帧)")
    print(f"[GT]    {out}/ground_truth.csv (含 boxes_json,兼容 viz3d/eval3d)")


if __name__ == '__main__':
    main()
