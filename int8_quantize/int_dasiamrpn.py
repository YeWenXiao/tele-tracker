"""int_dasiamrpn.py — TensorRT FP16/INT8 DaSiamRPN runtime, drop-in for cv2.TrackerDaSiamRPN.

接口语义跟 cv2.TrackerDaSiamRPN 完全一致:
    tracker = IntDaSiamRPN(model_main_path, kernel_r1_path, kernel_cls1_path,
                           template_path)
    tracker.init(image, (x, y, w, h))           # image=BGR np.ndarray
    ok, bbox = tracker.update(image)            # bbox=(x, y, w, h)

实现遵循 OpenCV cpp 原版 (modules/video/src/tracking/tracker_dasiamrpn.cpp):
- generateAnchors / generateHanningWindow
- search region cropping (context-aware) 跟 getSubwindow 同语义 (square)
- anchor decode: dx/dy * anchor_w + anchor_cx, exp(dw) * anchor_w
- penalty: scale change × ratio change × penaltyK
- score = (1-window_influence) * penalty * cls + window_influence * hanning
- argmax → lr smooth (lr = penalty * cls * base_lr)

资产 (4 个 TRT engine):
  - dasiamrpn_template_127_fp16.trt
      input (1,3,127,127), output 63 (1,256,6,6)
      用于 init() 时抽 template feature
  - dasiamrpn_kernel_r1_6x6_fp16.trt
      input (1,256,6,6), output (1,5120,4,4)
      把 template feature 卷成 reg head 的 new_layer_1.weight (reshape 20×256×4×4)
  - dasiamrpn_kernel_cls1_6x6_fp16.trt
      input (1,256,6,6), output (1,2560,4,4)
      cls head 的 new_layer_2.weight (reshape 10×256×4×4)
  - dasiamrpn_model_271_dynkern_fp16.trt
      inputs: (1,3,271,271), (20,256,4,4), (10,256,4,4)
      outputs: 66 (1,20,19,19) reg, 68 (1,10,19,19) cls
      主推理: 喂 search image + cached kernel weights → reg/cls map

为什么 4 个 engine:
  原版 ONNX (dasiamrpn_model.onnx) 把 new_layer_1/2 当作 initializer (固定权重),
  cpp 通过 setParam 在 init 时动态替换. TRT engine 不允许 setParam,所以我们把
  这两个 weight tensor 提到 graph input,build 时声明为 dynamic input.
  template engine 跟 main engine 共享 backbone 但输入尺寸不同 (127 vs 271),
  TRT 不支持单 engine 多 fixed shape,所以分两个 engine.

Day 1 build 的 dasiamrpn_model_int8.trt (360×480 input) 几何跟 cpp 不一致,
本文件不使用. INT8 量化要重新 calibrate 271×271 input — Day 2.5 task.
"""

import os
import time
from typing import Optional, Tuple

import numpy as np
import cv2

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    # NOTE: 不再用 pycuda.autoinit (它把 context 绑死在 import 它的线程).
    # 当主程序在 worker thread 调 init/update 时,autoinit 的 context 在那个线程不 current,
    # 报 "explicit_context_dependent failed: invalid device context".
    # 改为显式 cuda.init() + 在 IntDaSiamRPN 里管自己的 context (push/pop per call).
    cuda.init()
    _HAS_TRT = True
except Exception as _e:
    _HAS_TRT = False
    _IMPORT_ERR = _e


# 全局共享 context (一个 device 一个 primary context, 多 IntDaSiamRPN 实例共用,
# 多线程通过 push/pop 安全切换). lazy 创建.
_GLOBAL_CTX = None
_GLOBAL_CTX_LOCK = threading.Lock() if False else None  # 占位; 下面用 import threading


def _get_or_create_context():
    """惰性创建 device 0 的 cuda context, 多线程共享 (push/pop 串行化)."""
    global _GLOBAL_CTX
    if _GLOBAL_CTX is None:
        dev = cuda.Device(0)
        _GLOBAL_CTX = dev.make_context()
        # make_context() 把 ctx 推到当前线程栈顶, 立刻 pop 让 caller 自己 push
        _GLOBAL_CTX.pop()
    return _GLOBAL_CTX


import threading as _threading_for_ctx
_CTX_LOCK = _threading_for_ctx.Lock()


# ---- 跟 cpp 对齐的常量 (cpp Params() 默认值, 不要改) -----------------------

WINDOW_INFLUENCE = 0.43
LR = 0.4
PENALTY_K = 0.055
CONTEXT_AMOUNT = 0.5
TOTAL_STRIDE = 8
EXEMPLAR_SIZE = 127
INSTANCE_SIZE = 271
RATIOS = [0.33, 0.5, 1.0, 2.0, 3.0]
SCALES = 8
SCORE_SIZE = (INSTANCE_SIZE - EXEMPLAR_SIZE) // TOTAL_STRIDE + 1   # 19
ANCHOR_NUM = len(RATIOS)   # 5

# Default engine 路径 (用于 IntDaSiamRPN.from_default_paths 便利构造)
# 主 engine 可由 env DASIAMRPN_MAIN_ENGINE 覆盖,用于 INT8 vs FP16 切换
_BASE = "/home/nvidia/dual_cam_tracker/int8_quantize"
_DEFAULT_MAIN = os.environ.get(
    "DASIAMRPN_MAIN_ENGINE",
    os.path.join(_BASE, "dasiamrpn_model_271_dynkern_fp16.trt"),
)
_DEFAULT_KERN_R1 = os.path.join(_BASE, "dasiamrpn_kernel_r1_6x6_fp16.trt")
_DEFAULT_KERN_CLS1 = os.path.join(_BASE, "dasiamrpn_kernel_cls1_6x6_fp16.trt")
_DEFAULT_TEMPLATE = os.path.join(_BASE, "dasiamrpn_template_127_fp16.trt")


# ---- TRT engine wrapper -----------------------------------------------------

class _Engine:
    """Minimal pycuda + TRT engine wrapper. 支持多输入 (按 name bind).

    - 一次 alloc, 多次 infer
    - 可复用 host output buffer
    """

    def __init__(self, engine_path: str):
        with open(engine_path, 'rb') as f:
            data = f.read()
        self.engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"deserialize failed: {engine_path}")
        self.context = self.engine.create_execution_context()

        n_io = self.engine.num_io_tensors
        self.in_names = []
        self.in_shapes = {}
        self.out_names = []
        self.out_shapes = []
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(self.engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                self.in_names.append(name)
                self.in_shapes[name] = shape
            else:
                self.out_names.append(name)
                self.out_shapes.append(shape)

        # alloc input buffers (one per input)
        self.d_ins = {}
        for n in self.in_names:
            sz = int(np.prod(self.in_shapes[n])) * 4   # float32
            self.d_ins[n] = cuda.mem_alloc(sz)
            self.context.set_tensor_address(n, int(self.d_ins[n]))

        # alloc output buffers + host
        self.d_outs = []
        self.h_outs = []
        for n, s in zip(self.out_names, self.out_shapes):
            d = cuda.mem_alloc(int(np.prod(s)) * 4)
            self.context.set_tensor_address(n, int(d))
            self.d_outs.append(d)
            self.h_outs.append(np.empty(s, dtype=np.float32))

        self.stream = cuda.Stream()

    def set_input_persistent(self, name: str, x: np.ndarray) -> None:
        """把权重等 init 期一次 H2D, 后续 infer 不重传."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        cuda.memcpy_htod(self.d_ins[name], x)

    def infer_one_input(self, x: np.ndarray):
        """单输入 engine 的便利接口."""
        assert len(self.in_names) == 1
        return self.infer({self.in_names[0]: x})

    def infer(self, inputs: dict):
        """inputs: {name: np.ndarray}. 只对 inputs 中给出的 name 做 H2D
        (其他输入沿用之前 set_input_persistent 设的)."""
        for n, x in inputs.items():
            x = np.ascontiguousarray(x, dtype=np.float32)
            cuda.memcpy_htod_async(self.d_ins[n], x, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        for h, d in zip(self.h_outs, self.d_outs):
            cuda.memcpy_dtoh_async(h, d, self.stream)
        self.stream.synchronize()
        return self.h_outs


# ---- 主 tracker class -------------------------------------------------------

class IntDaSiamRPN:
    """drop-in replacement for cv2.TrackerDaSiamRPN, TRT FP16 backed (INT8-ready)."""

    def __init__(
        self,
        model_main_path: Optional[str] = None,
        kernel_r1_path: Optional[str] = None,
        kernel_cls1_path: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        if not _HAS_TRT:
            raise RuntimeError(f"[IntDaSiamRPN] TensorRT/pycuda 不可用: {_IMPORT_ERR}")

        # default paths (用 brief 给的 3 个 + 我们 rebuild 的 template engine)
        model_main_path = model_main_path or _DEFAULT_MAIN
        kernel_r1_path = kernel_r1_path or _DEFAULT_KERN_R1
        kernel_cls1_path = kernel_cls1_path or _DEFAULT_KERN_CLS1
        template_path = template_path or _DEFAULT_TEMPLATE

        for label, p in [
            ("model_main", model_main_path),
            ("kernel_r1", kernel_r1_path),
            ("kernel_cls1", kernel_cls1_path),
            ("template", template_path),
        ]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"[IntDaSiamRPN] missing {label}: {p}")

        # CUDA context: 共享 device-0 ctx, 多线程串行 push/pop.
        # 调用方所在的线程必须 push 后再调 TRT, 用完 pop. 保证 thread-safe.
        self._ctx = _get_or_create_context()

        try:
            with _CTX_LOCK:
                self._ctx.push()
                try:
                    self.eng_main = _Engine(model_main_path)
                    self.eng_template = _Engine(template_path)
                    self.eng_kernel_r1 = _Engine(kernel_r1_path)
                    self.eng_kernel_cls1 = _Engine(kernel_cls1_path)
                finally:
                    self._ctx.pop()
        except Exception as e:
            raise RuntimeError(f"[IntDaSiamRPN] engine load failed: {e}")

        # main engine output 区分 reg (20ch) vs cls (10ch)
        self._reg_idx = None
        self._cls_idx = None
        for i, s in enumerate(self.eng_main.out_shapes):
            if len(s) == 4 and s[1] == 20:
                self._reg_idx = i
            elif len(s) == 4 and s[1] == 10:
                self._cls_idx = i
        if self._reg_idx is None or self._cls_idx is None:
            raise RuntimeError(
                f"[IntDaSiamRPN] main engine outputs unexpected: {self.eng_main.out_shapes}"
            )

        # main engine 必须有 new_layer_1.weight / new_layer_2.weight 这两个 dynkern input
        # (build_engine 时把它们从 initializer 提出来了)
        for need in ("new_layer_1.weight", "new_layer_2.weight"):
            if need not in self.eng_main.in_shapes:
                raise RuntimeError(
                    f"[IntDaSiamRPN] main engine missing input '{need}'. "
                    f"available: {self.eng_main.in_names}"
                )

        # 找 search image input name (1,3,271,271)
        self._search_in_name = None
        for n, s in self.eng_main.in_shapes.items():
            if len(s) == 4 and s[1] == 3:
                self._search_in_name = n
                break
        if self._search_in_name is None:
            raise RuntimeError("main engine has no (1,3,H,W) image input")

        # anchors + hanning window (一次预算)
        self._anchors = self._generate_anchors()        # (4, ANCHOR_NUM*SS*SS)
        self._windows = self._generate_hanning()        # (ANCHOR_NUM*SS*SS,)

        # tracking state
        self.target_pos = None     # np.array([cx, cy], float)
        self.target_sz = None      # np.array([w, h], float)
        self.avg_chans = None      # np.array([B,G,R], float) for padding
        self.img_size = None       # (H, W)
        self.tracking_score = 0.0

    # ------- public interface (cv2 一致) -------

    def init(self, image: np.ndarray, bounding_box) -> None:
        """image: BGR uint8 (H,W,3); bounding_box: (x,y,w,h)."""
        if image is None or image.ndim != 3:
            raise ValueError(f"image bad shape: {None if image is None else image.shape}")
        x, y, w, h = [float(v) for v in bounding_box]
        cx = x + w * 0.5
        cy = y + h * 0.5

        self.target_pos = np.array([cx, cy], dtype=np.float32)
        self.target_sz = np.array([w, h], dtype=np.float32)
        self.img_size = (image.shape[0], image.shape[1])
        self.avg_chans = image.reshape(-1, 3).mean(axis=0).astype(np.float32)

        # 1. 抠 template region (cpp L153-156)
        wc = w + CONTEXT_AMOUNT * (w + h)
        hc = h + CONTEXT_AMOUNT * (w + h)
        sz = float(np.rint(np.sqrt(wc * hc)))   # cvRound

        z_crop = self._get_subwindow_square(
            image, self.target_pos, sz, EXEMPLAR_SIZE, self.avg_chans
        )   # (127, 127, 3) BGR float32

        # 2-5. TRT inference — push CUDA ctx (caller 可能在 worker thread)
        with _CTX_LOCK:
            self._ctx.push()
            try:
                # forward 主网络 → feature 63 (1,256,6,6) — cpp L161-163
                blob = z_crop.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
                feat = self.eng_template.infer_one_input(blob)[0]   # (1,256,6,6)

                # forward kernel networks → reg/cls kernel weights — cpp L165-169
                r1_out = self.eng_kernel_r1.infer_one_input(feat)[0]    # (1,5120,4,4)
                cls1_out = self.eng_kernel_cls1.infer_one_input(feat)[0]  # (1,2560,4,4)

                # reshape: cpp L170-173
                r1_kernel = r1_out.reshape(20, 256, 4, 4).copy()
                cls1_kernel = cls1_out.reshape(10, 256, 4, 4).copy()

                # 持久化喂给 main engine 的 dynkern input
                self.eng_main.set_input_persistent("new_layer_1.weight", r1_kernel)
                self.eng_main.set_input_persistent("new_layer_2.weight", cls1_kernel)
            finally:
                self._ctx.pop()

    def update(self, image: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        """returns (ok, (x, y, w, h))."""
        if self.target_pos is None:
            return False, (0, 0, 0, 0)
        if image is None or image.ndim != 3:
            return False, (0, 0, 0, 0)

        # 1. 算 search region (cpp trackerEval L193-201)
        w, h = float(self.target_sz[0]), float(self.target_sz[1])
        # NB: cpp 这里 wc 用 height、hc 用 width — 反序的 quirk,1:1 复刻
        wc = h + CONTEXT_AMOUNT * (w + h)
        hc = w + CONTEXT_AMOUNT * (w + h)
        sz = float(np.sqrt(wc * hc))               # 不 round (cpp L196 没 round)
        scale_z = EXEMPLAR_SIZE / sz
        search_size = (INSTANCE_SIZE - EXEMPLAR_SIZE) / 2.0   # 72
        pad = search_size / scale_z
        sx = float(np.rint(sz + 2.0 * pad))        # cvRound

        # 2. 抠 search patch (cpp getSubwindow square)
        x_crop = self._get_subwindow_square(
            image, self.target_pos, sx, INSTANCE_SIZE, self.avg_chans
        )   # (271, 271, 3)

        blob = x_crop.transpose(2, 0, 1)[None].astype(np.float32, copy=False)

        # 3. forward main engine — search image only,kernel weights 已 persistent
        # push ctx (worker thread safe). copy outs 为 owned ndarray, 出 ctx 后不动 device mem.
        with _CTX_LOCK:
            self._ctx.push()
            try:
                outs_raw = self.eng_main.infer({self._search_in_name: blob})
                delta = outs_raw[self._reg_idx].copy()  # (1, 20, 19, 19)
                score = outs_raw[self._cls_idx].copy()  # (1, 10, 19, 19)
            finally:
                self._ctx.pop()

        # 4. reshape: cpp L221-222
        delta = delta.reshape(4, ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)
        score = score.reshape(2, ANCHOR_NUM, SCORE_SIZE, SCORE_SIZE)

        pos_score = self._softmax2(score)[1]   # (ANCHOR_NUM, SS, SS)

        # cpp L226-227: targetBox.width *= scaleZ, height *= scaleZ
        # 之后 sizeCal 用这个 scaled size 跟 anchor 比较
        tgt_w_z = w * scale_z
        tgt_h_z = h * scale_z

        # 5. anchor decode (cpp L233-238)
        N = ANCHOR_NUM * SCORE_SIZE * SCORE_SIZE
        anc = self._anchors  # (4, N)
        delta_flat = delta.reshape(4, N)
        pred_x = delta_flat[0] * anc[2] + anc[0]
        pred_y = delta_flat[1] * anc[3] + anc[1]
        pred_w = np.exp(delta_flat[2]) * anc[2]
        pred_h = np.exp(delta_flat[3]) * anc[3]

        # 6. penalty (sc, rc) — cpp L240-249
        sc = self._size_cal(pred_w, pred_h) / self._size_cal(tgt_w_z, tgt_h_z)
        sc = np.maximum(sc, 1.0 / sc)
        rc = (tgt_w_z / tgt_h_z) / (pred_w / np.maximum(pred_h, 1e-8))
        rc = np.maximum(rc, 1.0 / rc)
        penalty = np.exp(-(rc * sc - 1.0) * PENALTY_K)

        # 7. window influence — cpp L251-252
        cls_flat = pos_score.reshape(N)
        pscore = penalty * cls_flat
        pscore = pscore * (1.0 - WINDOW_INFLUENCE) + self._windows * WINDOW_INFLUENCE

        # 8. argmax (cpp L254-256)
        best = int(np.argmax(pscore))

        # 9. 转回原图位移 + lr smooth (cpp L264-280)
        dx = pred_x[best] / scale_z
        dy = pred_y[best] / scale_z
        new_w = pred_w[best] / scale_z
        new_h = pred_h[best] / scale_z

        lr_eff = float(penalty[best] * cls_flat[best] * LR)
        new_cx = self.target_pos[0] + dx
        new_cy = self.target_pos[1] + dy
        sm_w = w * (1 - lr_eff) + new_w * lr_eff
        sm_h = h * (1 - lr_eff) + new_h * lr_eff

        # 10. clip (cpp L282-285)
        H_img, W_img = self.img_size
        new_cx = float(np.clip(new_cx, 0.0, float(W_img)))
        new_cy = float(np.clip(new_cy, 0.0, float(H_img)))
        sm_w = float(np.clip(sm_w, 10.0, float(W_img)))
        sm_h = float(np.clip(sm_h, 10.0, float(H_img)))

        # 11. update state
        self.target_pos = np.array([new_cx, new_cy], dtype=np.float32)
        self.target_sz = np.array([sm_w, sm_h], dtype=np.float32)
        self.tracking_score = float(cls_flat[best])

        # 12. bbox = (x, y, w, h) (cpp L181-185, int truncation)
        bbox = (
            int(new_cx - int(sm_w / 2)),
            int(new_cy - int(sm_h / 2)),
            int(sm_w),
            int(sm_h),
        )
        return True, bbox

    def getTrackingScore(self) -> float:
        return float(self.tracking_score)

    # ------- helpers (1:1 复刻 cpp) -------

    @staticmethod
    def _softmax2(x: np.ndarray) -> np.ndarray:
        """cpp softmax over axis=0 (2 elements: neg/pos)."""
        m = np.maximum(x[0], x[1])
        e0 = np.exp(x[0] - m)
        e1 = np.exp(x[1] - m)
        s = e0 + e1
        return np.stack([e0 / s, e1 / s], axis=0)

    @staticmethod
    def _size_cal(w, h):
        """cpp sizeCal: sqrt((w+pad)*(h+pad)), pad=(w+h)/2."""
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def _generate_anchors(self) -> np.ndarray:
        """cpp generateAnchors — 输出 shape (4, ANCHOR_NUM*SS*SS),
        4 行依次是 anchor_cx, anchor_cy, anchor_w, anchor_h.
        layout (k, i, j) flatten → (k*SS*SS + i*SS + j) — 跟 cpp reshape 一致.
        """
        # baseAnchor sizes (cpp L357-365)
        base = []
        size = TOTAL_STRIDE * TOTAL_STRIDE   # 64
        for r in RATIOS:
            ws = int(np.sqrt(size / r))
            hs = int(ws * r)
            base.append((ws * SCALES, hs * SCALES))

        # cpp ori = -(scoreSize/2) * totalStride
        ori = -(SCORE_SIZE // 2) * TOTAL_STRIDE   # = -72 (SS=19)

        AN, SS = ANCHOR_NUM, SCORE_SIZE
        anc = np.zeros((4, AN, SS, SS), dtype=np.float32)
        for i in range(SS):           # y
            for j in range(SS):       # x
                for k in range(AN):
                    anc[0, k, i, j] = ori + TOTAL_STRIDE * j   # cx
                    anc[1, k, i, j] = ori + TOTAL_STRIDE * i   # cy
                    anc[2, k, i, j] = base[k][0]               # w
                    anc[3, k, i, j] = base[k][1]               # h
        return anc.reshape(4, AN * SS * SS)

    def _generate_hanning(self) -> np.ndarray:
        """cpp generateHanningWindow: 2D hanning over (SS, SS), tile ANCHOR_NUM times."""
        hann = np.hanning(SCORE_SIZE).astype(np.float32)
        win2d = np.outer(hann, hann)    # (SS, SS)
        windows = np.tile(win2d[None], (ANCHOR_NUM, 1, 1)).reshape(-1)
        return windows.astype(np.float32)

    @staticmethod
    def _get_subwindow_square(
        img: np.ndarray,
        center: np.ndarray,
        original_size: float,
        out_size: int,
        avg_chans: np.ndarray,
    ) -> np.ndarray:
        """getSubwindow 方阵版 (cpp L396-427 1:1 复刻).

        - 在 image 中截 original_size × original_size 方阵, 中心=center
        - 越界用 avg_chans 填充
        - resize 到 out_size × out_size
        """
        cx, cy = float(center[0]), float(center[1])
        H, W = img.shape[:2]

        sz = float(original_size)
        c = (sz + 1) * 0.5
        # cpp 用 cvRound (banker's). np.rint 也是 banker's, 等价.
        x_min = int(np.rint(cx - c))
        y_min = int(np.rint(cy - c))
        x_max = x_min + int(sz) - 1
        y_max = y_min + int(sz) - 1

        left = max(0, -x_min)
        top = max(0, -y_min)
        right = max(0, x_max - W + 1)
        bottom = max(0, y_max - H + 1)

        x_min += left
        x_max += left
        y_min += top
        y_max += top

        if top == 0 and bottom == 0 and left == 0 and right == 0:
            crop = img[y_min:y_max + 1, x_min:x_max + 1]
        else:
            border = cv2.copyMakeBorder(
                img, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=avg_chans.tolist(),
            )
            crop = border[y_min:y_max + 1, x_min:x_max + 1]

        if crop.shape[0] != out_size or crop.shape[1] != out_size:
            crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

        return crop.astype(np.float32, copy=False)


# ---- self-test (python int_dasiamrpn.py) ------------------------------------

if __name__ == "__main__":
    import sys

    t = IntDaSiamRPN()   # uses default paths
    print(f"[OK] IntDaSiamRPN loaded:")
    print(f"     main engine inputs: {t.eng_main.in_names}")
    print(f"     main engine outputs: {[s for s in t.eng_main.out_shapes]}")
    print(f"     anchors={t._anchors.shape}, windows={t._windows.shape}")

    img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    t.init(img, (800, 400, 200, 150))
    print(f"[OK] init done")
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        ok, bbox = t.update(img)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"[OK] update bbox={bbox}, score={t.getTrackingScore():.3f}")
    a = np.array(times[5:])
    print(f"[OK] update latency avg={a.mean():.2f}ms p95={np.percentile(a,95):.2f}ms (15-frame after warmup)")
    sys.exit(0)
