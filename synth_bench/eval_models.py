#!/usr/bin/env python3
"""eval_models.py — 三模型同台考卷:SIFT vs NCC vs Siamese 判别力量化

特征层次谱系对照:
  SIFT    低层局部点(现状)
  NCC     中层整块结构(matchTemplate 归一化相关)
  Siamese CNN 抽象(IntDaSiamRPN cross-correlation 峰值 tracking_score)

考卷:gen_confusers 的目标 pattern + 分级混淆项(相似度 s 已知)。
对每项算 score → 归一化 ratio=score/target_score → 画 s-ratio 曲线 → 判别阈值 s*。
s* = ratio 跌破阈值的 s(越低=能区分越细微差异=判别力越强)。
"""
import os, sys, csv
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_confusers import make_target_pattern, perturb, render_cell

CELL_PX = 200
SEED = 7
K = 8
RATIO_THRESH = 0.70   # ratio 跌破此值算"能区分"


def build_questions():
    """复现 gen_confusers 的考卷(同 seed)。返回 [(label, img_bgr, s)]。"""
    rng = np.random.default_rng(SEED)
    target = make_target_pattern(K, rng)
    specs = [('copy', 1.0)]
    for f in [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]:
        specs.append(('color_flip', f))
    for f in [0.2, 0.4, 0.6]:
        specs.append(('shuffle', f))
    for f in [0.8, 0.5, 0.3]:
        specs.append(('subset', f))
    specs += [('mirror_h', 0), ('mirror_v', 0), ('rot180', 0)]
    items = [('TARGET', render_cell(target, CELL_PX), 1.0)]
    for kind, amt in specs:
        pat, s = perturb(target, kind, amt, rng)
        label = kind + (f'{amt:.2f}' if amt else '')
        items.append((label, render_cell(pat, CELL_PX), s))
    ref_img = render_cell(target, CELL_PX)
    return ref_img, items


# ---------- scorers ----------
class SIFTScorer:
    name = 'SIFT (低层点)'
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.bf = cv2.BFMatcher()
    def score(self, ref, query):
        g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        k1, d1 = self.sift.detectAndCompute(g1, None)
        k2, d2 = self.sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
            return 0.0
        good = 0
        for pair in self.bf.knnMatch(d1, d2, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good += 1
        return float(good)


class NCCScorer:
    name = 'NCC (中层结构)'
    def score(self, ref, query):
        g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if g1.shape != g2.shape:
            g2 = cv2.resize(g2, (g1.shape[1], g1.shape[0]))
        res = cv2.matchTemplate(g2, g1, cv2.TM_CCOEFF_NORMED)
        return float(res.max())


class ColorHistScorer:
    name = 'ColorHist (颜色)'
    def score(self, ref, query):
        def hist(img):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(h, h)
            return h.flatten()
        return float(cv2.compareHist(hist(ref), hist(query), cv2.HISTCMP_CORREL))


class SiameseClsScorer:
    """旧度量:cls objectness(已证伪 — 对什么都给高分,留作对比展示陷阱)"""
    name = 'Siamese-cls (objectness 陷阱)'
    def __init__(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'int8_quantize'))
        from int_dasiamrpn import IntDaSiamRPN
        self.t = IntDaSiamRPN()
    def score(self, ref, query):
        h, w = ref.shape[:2]
        self.t.init(ref, (0, 0, w, h))
        self.t.update(query)
        return self.t.getTrackingScore()


class SiameseEmbScorer:
    """修正度量:backbone template feature(1,256,6,6)余弦相似度 = 真实例判别"""
    name = 'Siamese-emb (backbone 特征)'
    def __init__(self):
        ip = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'int8_quantize')
        sys.path.insert(0, ip)
        from int_dasiamrpn import (IntDaSiamRPN, EXEMPLAR_SIZE,
                                   CONTEXT_AMOUNT, _CTX_LOCK)
        self.t = IntDaSiamRPN()
        self.E = EXEMPLAR_SIZE; self.CA = CONTEXT_AMOUNT; self.LOCK = _CTX_LOCK
    def _embed(self, img):
        h, w = img.shape[:2]
        pos = np.array([w / 2.0, h / 2.0], np.float32)
        wc = w + self.CA * (w + h); hc = h + self.CA * (w + h)
        sz = float(np.rint(np.sqrt(wc * hc)))
        avg = img.reshape(-1, 3).mean(0).astype(np.float32)
        z = self.t._get_subwindow_square(img, pos, sz, self.E, avg)
        blob = z.transpose(2, 0, 1)[None].astype(np.float32, copy=False)
        with self.LOCK:
            self.t._ctx.push()
            try:
                feat = self.t.eng_template.infer_one_input(blob)[0]
            finally:
                self.t._ctx.pop()
        return feat.flatten().astype(np.float64)
    def score(self, ref, query):
        fr = self._embed(ref); fq = self._embed(query)
        return float(np.dot(fr, fq) / (np.linalg.norm(fr) * np.linalg.norm(fq) + 1e-8))


def s_star(ss, ratios):
    """判别阈值:从 s=1 往低,ratio 首次 < RATIO_THRESH 的 s(线性插值)。"""
    order = np.argsort(-np.array(ss))
    ss_o = np.array(ss)[order]; rr_o = np.array(ratios)[order]
    for i in range(1, len(ss_o)):
        if rr_o[i] < RATIO_THRESH <= rr_o[i - 1]:
            # 插值
            t = (rr_o[i - 1] - RATIO_THRESH) / max(1e-6, rr_o[i - 1] - rr_o[i])
            return ss_o[i - 1] + t * (ss_o[i] - ss_o[i - 1])
    return ss_o[-1] if rr_o[-1] < RATIO_THRESH else 1.0


def main():
    out_dir = 'synth_bench/eval'
    os.makedirs(out_dir, exist_ok=True)
    ref_img, items = build_questions()
    print(f"考卷:{len(items)} 项(1 答案 + {len(items)-1} 混淆),CELL={CELL_PX}px")

    scorers = []
    for cls in (SIFTScorer, NCCScorer, ColorHistScorer, SiameseClsScorer, SiameseEmbScorer):
        try:
            scorers.append(cls())
            print(f"  [scorer] {cls.name} ready")
        except Exception as e:
            print(f"  [scorer] {cls.name} 跳过: {e}")

    results = {}  # name -> list of (label, s, score, ratio)
    csv_rows = []
    for sc in scorers:
        scores = [sc.score(ref_img, img) for (_, img, _) in items]
        tgt = scores[0] if scores[0] > 0 else max(scores + [1e-6])
        recs = []
        for (label, _, s), sco in zip(items, scores):
            ratio = sco / tgt if tgt > 0 else 0.0
            recs.append((label, s, sco, ratio))
            csv_rows.append([sc.name, label, f'{s:.3f}', f'{sco:.4f}', f'{ratio:.4f}'])
        results[sc.name] = recs

    # CSV
    with open(os.path.join(out_dir, 'eval_scores.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'item', 's', 'score', 'ratio'])
        w.writerows(csv_rows)

    # 曲线图:散点全项 + color_flip 序列连线 + s* 标注
    plt.figure(figsize=(10, 6))
    colors = {'SIFT (低层点)': 'tab:red', 'NCC (中层结构)': 'tab:green',
              'ColorHist (颜色)': 'tab:orange',
              'Siamese-cls (objectness 陷阱)': 'tab:gray',
              'Siamese-emb (backbone 特征)': 'tab:blue'}
    labels_en = {'SIFT (低层点)': 'SIFT (local pts)', 'NCC (中层结构)': 'NCC (structure)',
                 'ColorHist (颜色)': 'ColorHist', 'Siamese-cls (objectness 陷阱)': 'Siamese-cls (TRAP)',
                 'Siamese-emb (backbone 特征)': 'Siamese-emb (backbone)'}
    print("\n=== 判别阈值 s* (越低=判别力越强) ===")
    for name, recs in results.items():
        ss = [r[1] for r in recs]; rr = [r[3] for r in recs]
        c = colors.get(name, 'gray')
        plt.scatter(ss, rr, color=c, s=40, alpha=0.6)
        # color_flip 序列(干净 s 梯度)连线
        cf = sorted([(r[1], r[3]) for r in recs if r[0].startswith('color_flip')],
                    key=lambda x: -x[0])
        if cf:
            plt.plot([a for a, _ in cf], [b for _, b in cf], color=c,
                     label=labels_en.get(name, name), lw=2)
        star = s_star(ss, rr)
        plt.axvline(star, color=c, ls=':', alpha=0.5)
        print(f"  {name:30s} s* = {star:.2f}")
    plt.axhline(RATIO_THRESH, color='k', ls='--', alpha=0.4, label=f'mislock thresh {RATIO_THRESH}')
    plt.xlabel('similarity s  (1=target copy, 0=random)')
    plt.ylabel('score ratio (vs target)')
    plt.title('Discrimination: s-ratio curve (lower-right = stronger; color_flip series)')
    plt.legend(); plt.grid(alpha=0.3); plt.gca().invert_xaxis()
    png = os.path.join(out_dir, 'eval_curve.png')
    plt.savefig(png, dpi=120, bbox_inches='tight')
    print(f"\n[OUT] 曲线 → {png}")
    print(f"[OUT] 明细 → {out_dir}/eval_scores.csv")


if __name__ == '__main__':
    main()
