"""IntViTTrack — 纯 TRT 的 ViT tracker,干掉 cv2.dnn(框架开销 30ms → engine 0.58ms)。
drop-in replacement for cv2.TrackerVit:init(img,bbox) / update(img)->(ok,bbox) / getTrackingScore()。
逻辑精确复现 opencv/modules/video/src/tracking/tracker_vit.cpp。"""
import os
import numpy as np
import cv2
from int_dasiamrpn import _Engine, _get_or_create_context, _CTX_LOCK, cuda

_BASE = "/home/nvidia/dual_cam_tracker/models"
_DEFAULT_ENGINE = os.path.join(_BASE, "vittrack_fp16.trt")


def _hann1d(sz):
    i = np.arange(sz)
    return 0.5 * (1.0 - np.cos(2 * np.pi / (sz + 1) * (i + 1)))


def _hann2d(sz):
    h = _hann1d(sz)
    return np.outer(h, h).astype(np.float32)


class IntViTTrack:
    def __init__(self, engine_path=None):
        engine_path = engine_path or _DEFAULT_ENGINE
        self.ctx = _get_or_create_context()
        with _CTX_LOCK:
            self.ctx.push()
            try:
                self.eng = _Engine(engine_path)
            finally:
                self.ctx.pop()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
        self.hann = _hann2d(16)
        self.rect = None
        self.score = 0.0

    def _crop(self, img, box, factor):
        # 精确复现 crop_image:中心 crop sqrt(w*h)*factor,越界 BORDER_CONSTANT pad
        x, y, w, h = [int(v) for v in box]
        crop_sz = int(np.ceil(np.sqrt(max(1, w * h)) * factor))
        x1 = x + (w - crop_sz) // 2
        x2 = x1 + crop_sz
        y1 = y + (h - crop_sz) // 2
        y2 = y1 + crop_sz
        H, W = img.shape[:2]
        x1p = max(0, -x1); y1p = max(0, -y1)
        x2p = max(x2 - W + 1, 0); y2p = max(y2 - H + 1, 0)
        roi = img[y1 + y1p:y2 - y2p, x1 + x1p:x2 - x2p]
        return cv2.copyMakeBorder(roi, y1p, y2p, x1p, x2p, cv2.BORDER_CONSTANT)

    def _prep(self, crop, size):
        # resize → /255 → (x-mean)/std,BGR(不 swapRB,与 cv2 一致)
        img = cv2.resize(crop, (size, size)).astype(np.float32)
        blob = img.transpose(2, 0, 1) / 255.0
        blob = (blob - self.mean) / self.std
        return blob[None].astype(np.float32).copy()

    def init(self, image, bbox):
        crop = self._crop(image, bbox, 2)
        tmpl = self._prep(crop, 128)
        with _CTX_LOCK:
            self.ctx.push()
            try:
                self.eng.set_input_persistent("template", tmpl)
            finally:
                self.ctx.pop()
        self.rect = tuple(int(v) for v in bbox)
        return True

    def update(self, image):
        crop = self._crop(image, self.rect, 4)
        srch = self._prep(crop, 256)
        with _CTX_LOCK:
            self.ctx.push()
            try:
                outs = self.eng.infer({"search": srch})
            finally:
                self.ctx.pop()
        omap = {n: outs[i] for i, n in enumerate(self.eng.out_names)}
        conf = omap["output1"].reshape(16, 16)
        size_m = omap["output2"].reshape(2, 16, 16)
        offset = omap["output3"].reshape(2, 16, 16)
        conf = conf * self.hann
        idx = int(np.argmax(conf)); my, mx = idx // 16, idx % 16
        self.score = float(conf[my, mx])
        cx = (mx + offset[0, my, mx]) / 16.0
        cy = (my + offset[1, my, mx]) / 16.0
        w = size_m[0, my, mx]; h = size_m[1, my, mx]
        # returnfromcrop:归一化坐标 → 原图(search crop factor=4)
        rl = self.rect
        cw = 4 * int(np.floor(np.sqrt(max(1, rl[2] * rl[3]))))
        x0 = rl[0] + (rl[2] - cw) // 2
        y0 = rl[1] + (rl[3] - cw) // 2
        bx = int(np.floor((cx - w / 2) * cw + x0))
        by = int(np.floor((cy - h / 2) * cw + y0))
        bw = int(np.floor(w * cw))
        bh = int(np.floor(h * cw))
        self.rect = (bx, by, bw, bh)
        return True, self.rect

    def getTrackingScore(self):
        return self.score
