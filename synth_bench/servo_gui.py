#!/usr/bin/env python3
"""servo_gui.py — 闭环双控制器实时沙盘(滑块调参)

复用 gen_synth_track 的 VectorScene / ServoLoop / 畸变。
实时滑块:扰动幅度/平滑/伺服增益/识别噪声/运动模糊/视场(变焦)。
看「平滑扰动推开 vs 识别控制器拉回」的动态博弈。

键:q 退出 | r 重置视场对准目标 | 空格 暂停 | d 重新随机扰动种子
"""
import os, sys, math
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_synth_track import VectorScene, ServoLoop, apply_motion_blur, apply_rolling_shutter


def _nop(_):
    pass


def main():
    SCENE = 500.0
    OUT = 560
    rng = np.random.default_rng(42)

    scene = VectorScene(SCENE, rng)
    scene.build_background('random_blocks', 200)
    scene.build_dots(80)
    tcx = tcy = SCENE / 2.0
    half = 25.0  # 目标 50 单位
    scene.set_target(tcx, tcy, half)

    win = 'Servo Sandbox'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    # 滑块(整数 → 实值映射见下)
    cv2.createTrackbar('disturb_amp x0.1', win, 50, 300, _nop)   # 5.0,最大 30(大扰动)
    cv2.createTrackbar('disturb_smooth %', win, 92, 99, _nop)    # 0.92
    cv2.createTrackbar('servo_kp x0.01', win, 30, 100, _nop)     # 0.30
    cv2.createTrackbar('detect_noise px', win, 0, 120, _nop)     # 识别噪声
    cv2.createTrackbar('motion_blur x.01', win, 0, 50, _nop)     # 运动模糊耦合
    cv2.createTrackbar('fov_units', win, 150, 350, _nop)         # 视场=变焦

    servo = ServoLoop(SCENE, 150.0, OUT, 0.30, 5.0, 0.92, rng)
    view = np.array([tcx - 75.0, tcy - 75.0])
    prev = None
    vx = vy = 0.0
    sv = np.zeros(2); dv = np.zeros(2)
    paused = False
    t = 0
    lost_frames = 0

    while True:
        amp = cv2.getTrackbarPos('disturb_amp x0.1', win) * 0.1
        smooth = cv2.getTrackbarPos('disturb_smooth %', win) * 0.01
        kp = cv2.getTrackbarPos('servo_kp x0.01', win) * 0.01
        noise = float(cv2.getTrackbarPos('detect_noise px', win))
        blur_k = cv2.getTrackbarPos('motion_blur x.01', win) * 0.01
        fov = float(max(50, cv2.getTrackbarPos('fov_units', win)))
        servo.amp = amp; servo.smooth = smooth; servo.kp = kp
        servo.fov = fov; servo.px_per_unit = OUT / fov

        if not paused:
            bbox0, in0, _ = scene.target_bbox_in_view(view[0], view[1], fov, OUT)
            det = None
            if in0:
                dcx = (bbox0[0] + bbox0[2]) / 2; dcy = (bbox0[1] + bbox0[3]) / 2
                if noise > 0:
                    dcx += rng.normal(0, noise); dcy += rng.normal(0, noise)
                det = (dcx, dcy)
            new_view, sv, dv = servo.step(view, det, in0)
            if prev is None:
                vx = vy = 0.0
            else:
                sc = OUT / fov
                vx = (new_view[0] - prev[0]) * sc
                vy = (new_view[1] - prev[1]) * sc
            prev = new_view.copy(); view = new_view; t += 1

        img = scene.render(view[0], view[1], fov, OUT)
        if blur_k > 0 and not paused:
            img = apply_motion_blur(img, vx, vy, blur_k)

        bbox, in_fov, tsize = scene.target_bbox_in_view(view[0], view[1], fov, OUT)
        cen = OUT // 2
        cv2.drawMarker(img, (cen, cen), (255, 0, 0), cv2.MARKER_CROSS, 30, 2)
        err = -1.0
        if in_fov:
            pcx = (bbox[0] + bbox[2]) / 2; pcy = (bbox[1] + bbox[3]) / 2
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.line(img, (cen, cen), (int(pcx), int(pcy)), (0, 255, 255), 2)
            cv2.circle(img, (int(pcx), int(pcy)), 4, (0, 255, 0), -1)
            err = math.hypot(pcx - cen, pcy - cen)
            lost_frames = 0
        else:
            lost_frames += 1

        dmag = math.hypot(dv[0], dv[1]); smag = math.hypot(sv[0], sv[1])
        # 顶部状态栏
        cv2.rectangle(img, (0, 0), (OUT, 70), (30, 30, 30), -1)
        st = 'IN' if in_fov else f'LOST x{lost_frames}'
        col = (0, 255, 0) if in_fov else (0, 0, 255)
        cv2.putText(img, f"{st}  err={err:.0f}px  sz={tsize:.0f}  fov={fov:.0f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(img, f"disturb={dmag:.1f}  servo={smag:.1f}  kp={kp:.2f} amp={amp:.1f}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if paused:
            cv2.putText(img, "PAUSED", (OUT - 130, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            view = np.array([tcx - fov / 2.0, tcy - fov / 2.0])
            servo.dist_v = np.zeros(2); prev = None; lost_frames = 0
        elif k == ord(' '):
            paused = not paused
        elif k == ord('d'):
            servo.dist_v = rng.normal(0, amp, 2)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
