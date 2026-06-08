#!/usr/bin/env python3
"""viz3d.py — 干扰场景各模型效果对比视频(2x2 并排)

每帧 4 个算法各自定位,画框:绿=锁对目标 / 红=锁错干扰 / 灰=没找到。
黄细框=GT 目标(参考)。顶部标算法名 + 实时累计错锁数。
直观看:在一堆干扰里,谁认对目标、谁锁错。
"""
import os, sys, csv, json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval3d import (SIFTLocator, NCCLocator, SiameseEmbLocator,
                    classify, iou, IOU_HIT)


def panel(frame, box, gt_box, regions, title, mis, tot, scale=400):
    p = cv2.resize(frame, (scale, scale))
    sx = scale / frame.shape[1]; sy = scale / frame.shape[0]
    # GT 目标黄细框
    gb = [int(gt_box[0] * sx), int(gt_box[1] * sy), int(gt_box[2] * sx), int(gt_box[3] * sy)]
    cv2.rectangle(p, (gb[0], gb[1]), (gb[2], gb[3]), (0, 220, 220), 1)
    cls = classify(box, regions) if regions else 'miss'
    if box is not None:
        col = {'detect': (0, 255, 0), 'mislock': (0, 0, 255)}.get(cls, (150, 150, 150))
        bx = [int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)]
        cv2.rectangle(p, (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
    cv2.rectangle(p, (0, 0), (scale, 26), (35, 35, 35), -1)
    tag = {'detect': 'LOCK-OK', 'mislock': 'MISLOCK!', 'miss': '--'}.get(cls, '--')
    tcol = {'detect': (0, 255, 0), 'mislock': (0, 80, 255)}.get(cls, (180, 180, 180))
    cv2.putText(p, f"{title}  {tag}  mislock {mis}/{tot}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, tcol, 1)
    return p


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else 'synth_bench/out3d_conf'
    ref = cv2.imread(os.path.join(src, 'ref_front.png'))
    gt = list(csv.DictReader(open(os.path.join(src, 'ground_truth.csv'))))
    out = os.path.join(src, 'viz_compare.avi')

    sift = SIFTLocator(ref); ncc = NCCLocator(ref); semb = SiameseEmbLocator(ref)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'int8_quantize'))
    from int_dasiamrpn import IntDaSiamRPN
    strack = IntDaSiamRPN()

    cap = cv2.VideoCapture(os.path.join(src, 'video.avi'))
    wr = None
    mis = {'SIFT': 0, 'NCC': 0, 'Siam-track': 0, 'Siam-emb': 0}
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok or fi >= len(gt):
            break
        g = gt[fi]
        gt_box = (float(g['bbox_x0']), float(g['bbox_y0']),
                  float(g['bbox_x1']), float(g['bbox_y1']))
        regions = json.loads(g.get('boxes_json', '[]') or '[]')

        b_sift, _ = sift.locate(frame)
        b_ncc, _ = ncc.locate(frame)
        b_semb, _ = semb.locate(frame)
        if fi == 0:
            x0, y0, x1, y1 = gt_box
            strack.init(frame, (x0, y0, x1 - x0, y1 - y0)); b_tr = gt_box
        else:
            _, bb = strack.update(frame); b_tr = (bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3])

        for nm, bx in [('SIFT', b_sift), ('NCC', b_ncc),
                       ('Siam-track', b_tr), ('Siam-emb', b_semb)]:
            if regions and classify(bx, regions) == 'mislock':
                mis[nm] += 1

        p1 = panel(frame, b_sift, gt_box, regions, 'SIFT(reloc)', mis['SIFT'], fi + 1)
        p2 = panel(frame, b_ncc, gt_box, regions, 'NCC(reloc)', mis['NCC'], fi + 1)
        p3 = panel(frame, b_tr, gt_box, regions, 'Siamese(track)', mis['Siam-track'], fi + 1)
        p4 = panel(frame, b_semb, gt_box, regions, 'Siam-emb(reloc)', mis['Siam-emb'], fi + 1)
        grid = np.vstack([np.hstack([p1, p2]), np.hstack([p3, p4])])
        if wr is None:
            h, w = grid.shape[:2]
            wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'MJPG'), 20, (w, h))
        wr.write(grid)
        fi += 1
        if fi % 30 == 0:
            print(f"  {fi}/{len(gt)} 帧...")
    cap.release(); wr.release()
    print(f"[OUT] 对比视频 → {out}")
    print(f"  累计错锁: " + " / ".join(f"{k}={v}" for k, v in mis.items()))


if __name__ == '__main__':
    main()
