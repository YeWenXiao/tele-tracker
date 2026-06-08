#!/usr/bin/env python3
"""gen_confusers.py — 难负样本(混淆项)出题器

先定答案(目标 pattern),再出题(从目标派生分级混淆项)。
每个混淆项标相似度 s + 算子类型 → 用于量化模型判别阈值 s*。

混淆算子:
  color_flip(f)  翻转 f 比例点颜色      s = 1-f   测颜色判别
  shuffle(f)     打乱 f 比例点位置      s = 1-f   测结构判别
  subset(f)      只留 f 比例点          s = f     测部分/截断
  mirror_h/v     镜像                   s ~ 高    测旋转不变性
  rot180         旋转 180               s ~ 高
  copy           精确副本               s = 1.0   理论上限(谁都分不开)

输出:scene.png(全景题目可视化)+ confusers.csv(每项 pos/s/kind)+ target_pattern.npy
"""
import os, csv, argparse
import numpy as np
import cv2


def make_target_pattern(k, rng):
    """k×k 网格,每格一个随机高饱和色。返回 (k, k, 3) BGR uint8 网格色 + 点结构。"""
    return rng.integers(0, 256, (k, k, 3), dtype=np.uint8)


def perturb(pattern, kind, amount, rng):
    """对目标 pattern 施加扰动,返回 (新pattern, similarity_s)。"""
    p = pattern.copy()
    k = p.shape[0]
    if kind == 'copy':
        return p, 1.0
    if kind == 'color_flip':
        n = int(k * k * amount)
        idx = rng.choice(k * k, n, replace=False)
        flat = p.reshape(-1, 3)
        flat[idx] = rng.integers(0, 256, (n, 3), dtype=np.uint8)
        return flat.reshape(k, k, 3), 1.0 - amount
    if kind == 'shuffle':
        n = int(k * k * amount)
        flat = p.reshape(-1, 3)
        idx = rng.choice(k * k, n, replace=False)
        perm = rng.permutation(idx)
        flat[idx] = flat[perm]
        return flat.reshape(k, k, 3), 1.0 - amount
    if kind == 'subset':
        flat = p.reshape(-1, 3).copy()
        n_drop = int(k * k * (1 - amount))
        idx = rng.choice(k * k, n_drop, replace=False)
        flat[idx] = (245, 245, 245)  # 抹成背景色
        return flat.reshape(k, k, 3), amount
    if kind == 'mirror_h':
        return p[:, ::-1], 0.85
    if kind == 'mirror_v':
        return p[::-1, :], 0.85
    if kind == 'rot180':
        return p[::-1, ::-1], 0.80
    return p, 0.0


def render_cell(pattern, px):
    """把 k×k pattern 渲染成 px×px 图(每格一个圆点,灰底)。"""
    k = pattern.shape[0]
    img = np.full((px, px, 3), 245, np.uint8)
    step = px / k
    r = max(1, int(step * 0.38))
    for i in range(k):
        for j in range(k):
            cx = int((i + 0.5) * step); cy = int((j + 0.5) * step)
            c = tuple(int(v) for v in pattern[i, j])
            cv2.circle(img, (cx, cy), r, c, -1)
    return img


def main():
    ap = argparse.ArgumentParser(description="难负样本混淆项出题器")
    ap.add_argument('--out-dir', default='synth_bench/confusers')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--k', type=int, default=8, help='目标 pattern 网格 k×k')
    ap.add_argument('--cell-px', type=int, default=120, help='每个 pattern 渲染像素')
    ap.add_argument('--cols', type=int, default=5, help='题目网格列数')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    target = make_target_pattern(args.k, rng)
    np.save(os.path.join(args.out_dir, 'target_pattern.npy'), target)

    # 出题:分级混淆项谱系
    specs = [('copy', 1.0)]
    for f in [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]:
        specs.append(('color_flip', f))
    for f in [0.2, 0.4, 0.6]:
        specs.append(('shuffle', f))
    for f in [0.8, 0.5, 0.3]:
        specs.append(('subset', f))
    specs += [('mirror_h', 0), ('mirror_v', 0), ('rot180', 0)]

    items = [('TARGET', target, 1.0)]  # 答案放第一格
    for kind, amt in specs:
        pat, s = perturb(target, kind, amt, rng)
        label = f'{kind}' + (f'{amt:.2f}' if amt else '')
        items.append((label, pat, s))

    # 渲染成题目网格图
    cp = args.cell_px; pad = 30; cols = args.cols
    rows = (len(items) + cols - 1) // cols
    W = cols * (cp + pad) + pad
    H = rows * (cp + pad + 24) + pad
    canvas = np.full((H, W, 3), 60, np.uint8)
    csv_f = open(os.path.join(args.out_dir, 'confusers.csv'), 'w', newline='')
    wr = csv.writer(csv_f); wr.writerow(['idx', 'kind', 'similarity_s', 'grid_x', 'grid_y'])

    for n, (label, pat, s) in enumerate(items):
        gx = n % cols; gy = n // cols
        x0 = pad + gx * (cp + pad); y0 = pad + gy * (cp + pad + 24)
        cell = render_cell(pat, cp)
        # 答案绿框,其余按 s 红→黄渐变(越像越红=越难)
        if n == 0:
            cv2.rectangle(cell, (0, 0), (cp - 1, cp - 1), (0, 255, 0), 3)
        canvas[y0:y0 + cp, x0:x0 + cp] = cell
        col = (0, int(255 * (1 - s)), 255) if n > 0 else (0, 255, 0)
        cv2.putText(canvas, f'{label}', (x0, y0 + cp + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(canvas, f's={s:.2f}', (x0 + cp - 56, y0 + cp + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        wr.writerow([n, label, f'{s:.3f}', gx, gy])

    csv_f.close()
    out_png = os.path.join(args.out_dir, 'scene.png')
    cv2.imwrite(out_png, canvas)
    print(f"[OUT] 题目网格 {len(items)} 项(1 答案 + {len(items)-1} 混淆)→ {out_png}")
    print(f"[OUT] confusers.csv + target_pattern.npy → {args.out_dir}")
    print(f"  绿框=目标答案; 其余标 s(相似度,越高越难,红色=最像)")


if __name__ == '__main__':
    main()
