
import argparse
import time
import os
import sys
import json
from collections import defaultdict, deque

import numpy as np
import cv2
import yaml

# Optional: silence OpenMP duplicate warnings on some Jetson builds
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERROR] ultralytics not available. pip install ultralytics==8.*", e)
    sys.exit(1)

# ==========================
# Class-specific TTS messages
# ==========================
INSTRUCTIONS = {
    "Animal": {
        "ahead": "Please pause. Animal ahead. Give space.",
        "left":  "Animal on your left. Gently move right.",
        "right": "Animal on your right. Gently move left.",
    },
    "Bollard": {
        "ahead": "Small post ahead. Please go around.",
        "left":  "Post on your left. Gently move right.",
        "right": "Post on your right. Gently move left.",
    },
    "Broken-road": {
        "ahead": "Rough road ahead. Proceed with care; walk around.",
        "left":  "Rough patch left. Kindly move right.",
        "right": "Rough patch right. Kindly move left.",
    },
    "Car": {
        "ahead": "Please stop. Car ahead. Wait for a clear path.",
        "left":  "Car on your left. Gently move right.",
        "right": "Car on your right. Gently move left.",
    },
    "Construction-barrier": {
        "ahead": "Barrier ahead. Please pause, then turn left or right.",
        "left":  "Barrier to your left. Kindly go right.",
        "right": "Barrier to your right. Kindly go left.",
    },
    "Crosswalk": {
        "ahead": "Crosswalk ahead. Please wait for the walk signal.",
        "left":  "Crosswalk on your left. Angle slightly left to align.",
        "right": "Crosswalk on your right. Angle slightly right to align.",
    },
    "Curb": {
        "ahead": "Curb ahead. Prepare to step up or turn.",
        "left":  "Curb on your left. Gently move right.",
        "right": "Curb on your right. Gently move left.",
    },
    "Dog": {
        "ahead": "Please pause. Dog ahead. Give space.",
        "left":  "Dog on your left. Gently move right.",
        "right": "Dog on your right. Gently move left.",
    },
    "Fallen-tree": {
        "ahead": "Path blocked by a fallen tree. Please turn.",
        "left":  "Fallen tree left. Kindly go right.",
        "right": "Fallen tree right. Kindly go left.",
    },
    "Garbage": {
        "ahead": "Debris ahead. Proceed carefully; walk around.",
        "left":  "Debris on your left. Gently move right.",
        "right": "Debris on your right. Gently move left.",
    },
    "Garbage-bin": {
        "ahead": "Bin ahead. Please go around.",
        "left":  "Bin on your left. Kindly move right.",
        "right": "Bin on your right. Kindly move left.",
    },
    "Motorcycle": {
        "ahead": "Please stop. Motorcycle ahead. Wait for space.",
        "left":  "Motorcycle left. Gently move right.",
        "right": "Motorcycle right. Gently move left.",
    },
    "Muddy road": {
        "ahead": "Muddy ground ahead. Watch your step; go around.",
        "left":  "Mud on your left. Kindly move right.",
        "right": "Mud on your right. Kindly move left.",
    },
    "Obstacle": {
        "ahead": "Obstacle ahead. Hold a steady pace; pass safely.",
        "left":  "Obstacle left. Gently move right.",
        "right": "Obstacle right. Gently move left.",
    },
    "Person": {
        "ahead": "Person ahead. Please pass kindly on one side.",
        "left":  "Person on your left. Gently move right.",
        "right": "Person on your right. Gently move left.",
    },
    "Pole": {
        "ahead": "Pole ahead. Please go around.",
        "left":  "Pole on your left. Kindly move right.",
        "right": "Pole on your right. Kindly move left.",
    },
    "Pothole": {
        "ahead": "Pothole ahead. Mind your step; walk around.",
        "left":  "Pothole left. Gently move right.",
        "right": "Pothole right. Gently move left.",
    },
    "Potted-plant": {
        "ahead": "Plant ahead. Please go around.",
        "left":  "Plant on your left. Kindly move right.",
        "right": "Plant on your right. Kindly move left.",
    },
    "Railroad-crossing": {
        "ahead": "Rail crossing ahead. Please stop and listen.",
        "left":  "Crossing on your left. Angle left to approach.",
        "right": "Crossing on your right. Angle right to approach.",
    },
    "Rickshaw": {
        "ahead": "Please stop. Rickshaw ahead. Wait to pass.",
        "left":  "Rickshaw left. Gently move right.",
        "right": "Rickshaw right. Gently move left.",
    },
    "Road-barrier": {
        "ahead": "Road barrier ahead. Please pause and turn.",
        "left":  "Barrier left. Kindly go right.",
        "right": "Barrier right. Kindly go left.",
    },
    "Sidewalk": {
        "ahead": "Sidewalk ahead. Hold center and go straight.",
        "left":  "Sidewalk left. Angle softly left to center.",
        "right": "Sidewalk right. Angle softly right to center.",
    },
    "Stuck-water": {
        "ahead": "Water ahead. Tread carefully; go around.",
        "left":  "Water on your left. Kindly move right.",
        "right": "Water on your right. Kindly move left.",
    },
    "Traffic-light": {
        "ahead": "Traffic light ahead. Please wait for walk.",
        "left":  "Light on your left. Align to cross when safe.",
        "right": "Light on your right. Align to cross when safe.",
    },
    "Traffic-sign": {
        "ahead": "Sign ahead. Continue with care.",
        "left":  "Sign on your left. Gently move right.",
        "right": "Sign on your right. Gently move left.",
    },
    "Tree": {
        "ahead": "Tree ahead. Please go around.",
        "left":  "Tree on your left. Kindly move right.",
        "right": "Tree on your right. Kindly move left.",
    },
    "Upstairs": {
        "ahead": "Stairs ahead. Please stop and find the handrail.",
        "left":  "Stairs to your left. Turn left to the rail.",
        "right": "Stairs to your right. Turn right to the rail.",
    },
    "Vehicle": {
        "ahead": "Please stop. Vehicle ahead. Wait for space.",
        "left":  "Vehicle on your left. Gently move right.",
        "right": "Vehicle on your right. Gently move left.",
    },
    "Wall": {
        "ahead": "Wall ahead. Please pause and turn.",
        "left":  "Wall on your left. Kindly go right.",
        "right": "Wall on your right. Kindly go left.",
    },
}

_DEFAULTS = {
    "ahead": "Obstacle ahead. Proceed with care; pass safely.",
    "left":  "Obstacle on your left. Gently move right.",
    "right": "Obstacle on your right. Gently move left.",
}

# Map model class names to our keys where needed (optional normalization)
# e.g., unify spaces/underscores to match your dictionary.
def normalize_class(name: str) -> str:
    # Canonicalize spacing and casing without being too aggressive
    n = name.strip()
    # Common unifications:
    replacements = {
        "Muddy_road": "Muddy road",
        "Muddy-road": "Muddy road",
        "Fallen_tree": "Fallen-tree",
        "Fallen-Tree": "Fallen-tree",
        "Traffic_light": "Traffic-light",
        "Traffic_light": "Traffic-light",
        "Traffic_sign": "Traffic-sign",
        "Railroad_crossing": "Railroad-crossing",
        "Garbage_bin": "Garbage-bin",
    }
    return replacements.get(n, n)

# ---------- Soft-NMS (Gaussian) ----------
def soft_nms_gaussian(dets, sigma=0.5, Nt=0.5, score_thresh=0.001):
    if dets is None or len(dets) == 0:
        return dets
    D = dets.copy()
    N = D.shape[0]
    keep = []
    i = 0
    while i < N:
        maxpos = i + np.argmax(D[i:, 4])
        D[[i, maxpos]] = D[[maxpos, i]]
        xi1, yi1, xi2, yi2, si, ci = D[i]
        keep.append(D[i].copy())
        j = i + 1
        while j < N:
            xj1, yj1, xj2, yj2, sj, cj = D[j]
            xx1 = max(xi1, xj1); yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2); yy2 = min(yi2, yj2)
            w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
            inter = w * h
            area_i = max(0.0, (xi2 - xi1)) * max(0.0, (yi2 - yi1))
            area_j = max(0.0, (xj2 - xj1)) * max(0.0, (yj2 - yj1))
            denom = area_i + area_j - inter + 1e-9
            ovr = inter / denom if denom > 0 else 0.0
            if ovr > Nt:
                weight = np.exp(- (ovr * ovr) / sigma)
                D[j, 4] = D[j, 4] * weight
                if D[j, 4] <= score_thresh:
                    D = np.vstack([D[:j], D[j+1:]])
                    N -= 1
                    continue
            j += 1
        i += 1
    return np.array(keep, dtype=np.float32)

# ---------- Proximity + TTC ----------
class ProximityHead:
    def __init__(self, class_names, ema_momentum=0.05, min_area=1e-5, conf_thr=0.25):
        self.class_names = class_names
        self.ema_m = ema_momentum
        self.min_area = min_area
        self.conf_thr = conf_thr
        self.ema_area = defaultdict(lambda: 1.0)

    def update_ema(self, cls_ids, areas):
        for c in np.unique(cls_ids):
            v = np.median(areas[cls_ids == c]) if np.any(cls_ids == c) else None
            if v is not None and v > 0:
                self.ema_area[int(c)] = 0.95 * self.ema_area[int(c)] + 0.05 * v

    def __call__(self, det, shape):
        H, W = shape[:2]
        if det is None or len(det) == 0:
            return None, None, {}
        xyxy = det[:, :4]; conf = det[:, 4]; cls = det[:, 5].astype(int)
        w = xyxy[:, 2] - xyxy[:, 0]; h = xyxy[:, 3] - xyxy[:, 1]
        A = w * h
        A_norm = (A / (W * H)).clip(1e-9, 1e9)
        self.update_ema(cls, A_norm)
        s_c = np.array([self.ema_area[int(c)] for c in cls])
        A_proxy = A_norm / (s_c + 1e-9)
        mask = (conf >= self.conf_thr) & (A_norm >= self.min_area)
        if not np.any(mask):
            return None, None, {}
        idxs = np.where(mask)[0]
        ci = idxs[np.argmax(A_proxy[mask])]
        fi = idxs[np.argmin(A_proxy[mask])]

        def clock_dir(xc):
            if xc < W / 3: return "left"
            if xc > 2 * W / 3: return "right"
            return "center"
        def token(i):
            c = int(cls[i])
            name = self.class_names[c] if c < len(self.class_names) else str(c)
            x1, y1, x2, y2 = xyxy[i]
            return {"cls": name, "dir": clock_dir((x1 + x2) / 2), "conf": float(conf[i])}
        return ci, fi, {"closest": token(ci), "farthest": token(fi)}

class TTCProxy:
    def __init__(self, ema=0.4, dt=1/30.0, eps=1e-3):
        self.dt = dt; self.eps = eps; self.ema = ema
        self.h_ema = {}; self.dln_ema = {}

    def update(self, tid, h):
        if h <= 0: return None
        prev = self.h_ema.get(tid, h)
        h_s = self.ema*prev + (1-self.ema)*h
        self.h_ema[tid] = h_s
        dln = (np.log(h_s + self.eps) - np.log(prev + self.eps)) / self.dt
        dln_s = self.ema*self.dln_ema.get(tid, dln) + (1-self.ema)*dln
        self.dln_ema[tid] = dln_s
        ttc = -1.0 / (dln_s + self.eps)
        return max(0.0, float(ttc))

# ---------- Lightweight ID assignment (IoU matching per class) ----------
class IDAssigner:
    def __init__(self, iou_thr=0.3, ttl=10):
        self.next_id = 1
        self.tracks = {}  # id -> (cls, box, last_seen)
        self.iou_thr = iou_thr
        self.ttl = ttl
        self.frame_idx = 0

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1); ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
        area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
        union = area_a + area_b - inter + 1e-9
        return inter / union

    def update(self, det):
        self.frame_idx += 1
        assigned = {}
        used_ids = set()
        for i in range(det.shape[0]):
            box = det[i, :4]; cls = int(det[i, 5])
            best_id, best_iou = None, 0.0
            for tid, (tcls, tbox, last_seen) in self.tracks.items():
                if tcls != cls or tid in used_ids:
                    continue
                iou = self.iou(box, tbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_iou >= self.iou_thr and best_id is not None:
                assigned[i] = best_id
                used_ids.add(best_id)
                self.tracks[best_id] = (cls, box.copy(), self.frame_idx)
            else:
                tid = self.next_id; self.next_id += 1
                assigned[i] = tid
                used_ids.add(tid)
                self.tracks[tid] = (cls, box.copy(), self.frame_idx)
        dead = [tid for tid, (_, _, ls) in self.tracks.items() if self.frame_idx - ls > self.ttl]
        for d in dead:
            self.tracks.pop(d, None)
        return assigned

# ---------- TTS helpers ----------
class Speaker:
    def __init__(self, enable=False, piper_model=None, voice=None, rate=None):
        self.enable = enable
        self.piper_model = piper_model  # path to *.onnx for Piper
        self.voice = voice
        self.rate = rate
        self.have_piper = bool(piper_model and os.path.exists(piper_model))
        try:
            import pyttsx3  # noqa
            self.have_pyttsx3 = True
        except Exception:
            self.have_pyttsx3 = False

    def say(self, text):
        if not self.enable:
            return
        if self.have_piper:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav = tmp.name
            cmd = ["piper", "-m", self.piper_model, "-f", wav]
            try:
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p.communicate(input=text.encode("utf-8"), timeout=10)
                subprocess.call(["aplay", wav])
            except Exception as e:
                print("[TTS] Piper failed:", e)
        elif self.have_pyttsx3:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                if self.rate:
                    engine.setProperty('rate', self.rate)
                if self.voice:
                    engine.setProperty('voice', self.voice)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("[TTS] pyttsx3 failed:", e)
        else:
            print("[TTS disabled]", text)

# ---------- Main loop ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".engine (TRT) | .onnx | .pt")
    ap.add_argument("--data", required=True, help="data.yaml with class names")
    ap.add_argument("--source", default="0", help="webcam index or video path")
    ap.add_argument("--imgsz", type=int, default=736)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--no_softnms", action="store_true")
    ap.add_argument("--hazards", type=str, default="person,bicycle,car")
    ap.add_argument("--engine", type=str, choices=["auto","onnx","trt","pt"], default="auto")
    ap.add_argument("--show", action="store_true", help="show cv2 window")
    ap.add_argument("--save", type=str, default="", help="optional output video path")
    ap.add_argument("--speak", action="store_true")
    ap.add_argument("--piper_model", type=str, default="", help="path to piper voice onnx")
    ap.add_argument("--cooldown", type=float, default=1.5, help="seconds between TTS messages")
    args = ap.parse_args()

    # Load class names
    with open(args.data, "r", encoding="utf-8") as f:
        names = yaml.safe_load(f).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]

    # Model
    model = YOLO(args.weights)

    # Video source
    src = 0 if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        sys.exit(1)

    # Writers, helpers
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_out = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps_out, (width, height))

    prox = ProximityHead(names, conf_thr=args.conf)
    ttc_proxy = TTCProxy(dt=1/30.0)
    id_assign = IDAssigner(iou_thr=0.35, ttl=15)

    hazard_set = set([s.strip().lower() for s in args.hazards.split(',') if s.strip()])
    alpha, beta, gamma = 1.0, 1.0, 0.5

    speaker = Speaker(enable=args.speak, piper_model=args.piper_model)
    last_spoken_at = 0.0
    last_phrase = ""

    fps_pings = deque(maxlen=20)

    def dir_to_key(d: str) -> str:
        # left|center|right  → left|ahead|right
        return {"left": "left", "center": "ahead", "right": "right"}.get(d, "ahead")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        # Run model
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        im = r.orig_img
        H, W = im.shape[:2]

        if r.boxes is None or r.boxes.shape[0] == 0:
            if args.show:
                cv2.imshow("proximity", im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if writer: writer.write(im)
            continue

        det = np.concatenate(
            [r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None], r.boxes.cls.cpu().numpy()[:, None]],
            axis=1,
        ).astype(np.float32)

        if not args.no_softnms and det.shape[0] > 1:
            det = soft_nms_gaussian(det)

        # Assign IDs for TTC across frames
        id_map = id_assign.update(det)

        # Proximity tokens
        ci, fi, tokens = prox(det, im.shape)

        # Hazard selection
        hazard_score = -1.0; hazard_token = None
        for i in range(det.shape[0]):
            x1, y1, x2, y2, sc, cl = det[i]
            w = x2 - x1; h = y2 - y1
            A_norm = (w*h) / (W*H + 1e-9)
            tid = id_map.get(i, None)
            h_norm = h / (H + 1e-9)
            ttc = ttc_proxy.update(int(tid), h_norm) if tid is not None else None
            ttc_eff = ttc if (ttc is not None and np.isfinite(ttc)) else 1e9
            cls_name_raw = names[int(cl)] if int(cl) < len(names) else str(int(cl))
            cls_name = normalize_class(cls_name_raw)
            Aproxy = A_norm / (prox.ema_area[int(cl)] + 1e-9)
            score = alpha*Aproxy + beta/(ttc_eff + 1e-3) + (gamma if cls_name_raw.lower() in hazard_set else 0.0)
            if score > hazard_score:
                hazard_score = score
                dir_ = "left" if (x1+x2)/2 < W/3 else ("right" if (x1+x2)/2 > 2*W/3 else "center")
                hazard_token = {"cls": cls_name, "dir": dir_, "ttc": float(round(ttc_eff if ttc_eff<1e9 else 0.0, 2))}

        # Draw and label
        colors = {'closest': (0, 255, 0), 'farthest': (0, 0, 255), 'hazard': (0, 255, 255)}
        for tag, idx in [("closest", ci), ("farthest", fi)]:
            if idx is None: continue
            x1, y1, x2, y2 = det[idx, :4].astype(int)
            cv2.rectangle(im, (x1, y1), (x2, y2), colors[tag], 2)
            cv2.putText(im, tag, (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[tag], 2, cv2.LINE_AA)

        if hazard_token is not None:
            hx = int((det[ci,0]+det[ci,2])/2) if ci is not None else int(W/2)
            hy = int((det[ci,1]+det[ci,3])/2) if ci is not None else int(H/2)
            cv2.circle(im, (hx, hy), 6, colors['hazard'], -1)

        # HUD: FPS
        dt = time.time() - t0
        fps_pings.append(1.0/max(dt,1e-6))
        fps = sum(fps_pings)/len(fps_pings)
        cv2.putText(im, f"FPS:{fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # ===============================
        # TTS phrase crafting (per-class)
        # ===============================
        def pick_instruction(cls_name: str, d_key: str) -> str:
            bank = INSTRUCTIONS.get(cls_name, _DEFAULTS)
            return bank.get(d_key, _DEFAULTS[d_key])

        phrase = None
        # Priority 1: Imminent hazard → use class-specific ahead/left/right
        if hazard_token and 0 < hazard_token.get("ttc", 0) < 1.0:
            d_key = dir_to_key(hazard_token['dir'])
            phrase = pick_instruction(hazard_token['cls'], d_key)
        # Priority 2: Near-term hazard → also use class-specific instruction
        elif hazard_token and 1.0 <= hazard_token.get("ttc", 0) < 2.0:
            d_key = dir_to_key(hazard_token['dir'])
            phrase = pick_instruction(hazard_token['cls'], d_key)
        # Priority 3: No urgent hazard → use closest object instruction
        elif tokens and tokens.get('closest'):
            c = tokens['closest']
            cls_c = normalize_class(c['cls'])
            d_key = dir_to_key(c['dir'])
            phrase = pick_instruction(cls_c, d_key)

        # Speak with cooldown / anti-repeat
        now = time.time()
        if phrase and (now - last_spoken_at > args.cooldown) and phrase != last_phrase:
            Speaker.say(speaker, phrase)
            last_spoken_at = now
            last_phrase = phrase

        # Show/save
        if args.show:
            cv2.imshow("proximity", im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if writer:
            writer.write(im)

        # Console JSON for logging/telemetry
        out = {}
        if tokens: out.update(tokens)
        if hazard_token: out['hazard'] = hazard_token
        if phrase: out['tts'] = phrase
        if out:
            print("TTS:", json.dumps(out))

    cap.release()
    if writer: writer.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
