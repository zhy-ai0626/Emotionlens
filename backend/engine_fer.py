import os
import cv2
import numpy as np
import torch

# HSEmotion .pt files were saved on CUDA; default-patch torch.load to CPU
# so loading works in any environment. Idempotent.
if not getattr(torch.load, "_emolens_cpu_patched", False):
    _orig_torch_load = torch.load
    def _cpu_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        return _orig_torch_load(*args, **kwargs)
    _cpu_torch_load._emolens_cpu_patched = True
    torch.load = _cpu_torch_load

import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image

# HSEmotion enet_b2_7 emits classes in a different order than our EMO list.
# We use this to remap its softmax vector to our index order at inference.
HSE_LABELS = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

EMO_COLOR = {
    "neutral":    (180, 180, 180),
    "happiness":  (0, 255, 100),
    "surprise":   (255, 200, 0),
    "sadness":    (200, 80, 0),
    "anger":      (0, 0, 240),
    "disgust":    (0, 140, 60),
    "fear":       (120, 0, 180),
}

# ── Inference calibration ──────────────────────────────────────
# 1) Logit adjustment / prior correction. Training oversampled disgust
#    20× (see history.json: self_oversample=20.0) leaving a residual bias
#    toward disgust at inference. Subtracting from disgust logits restores
#    the prior — see "Long-tail learning via logit adjustment"
#    (Menon et al., NeurIPS 2020). Index order matches EMO above.
LOGIT_BIAS = [0.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0]  # disgust at idx 5

# 2) Softmax temperature > 1 flattens the distribution. argmax unchanged;
#    displayed top-1 percentage drops and runners-up become visible — wrong
#    predictions look like uncertainty instead of confident error.
SOFTMAX_TEMPERATURE = 1.5

# 3) Minimum face-crop side. MTCNN already filters at min_face_size=60;
#    this is a second gate to skip very distant faces whose classifier
#    output would just be noise.
MIN_CROP_PX = 100

_MIN_HITS   = 3
_MAX_MISS   = 8
_IOU_THRESH = 0.25

def _ema_alpha():
    """Single source of truth for the tracker's EMA smoothing factor."""
    from backend.config import EMA_GAMMA
    return EMA_GAMMA

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

class FaceTracker:
    def __init__(self):
        self._tracks = []
        self._next_id = 1

    def reset(self):
        self._tracks = []

    def update(self, raw_rects, raw_probs_list):
        a = _ema_alpha()
        used_det, used_trk = set(), set()

        for ti, t in enumerate(self._tracks):
            best_v, best_di = _IOU_THRESH, -1
            for di, r in enumerate(raw_rects):
                if di in used_det:
                    continue
                v = _iou(t['ri'], r)
                if v > best_v:
                    best_v, best_di = v, di
            if best_di >= 0:
                used_det.add(best_di)
                used_trk.add(ti)
                nr = raw_rects[best_di]
                t['rf'] = [a*nr[i] + (1-a)*t['rf'][i] for i in range(4)]
                t['ri'] = [int(v) for v in t['rf']]
                t['probs'] = a * raw_probs_list[best_di] + (1-a) * t['probs']
                t['hits'] += 1
                t['miss']  = 0
                if t['hits'] >= _MIN_HITS:
                    t['ok'] = True

        surviving = []
        for ti, t in enumerate(self._tracks):
            if ti not in used_trk:
                t['hits'] = 0
                t['miss'] += 1
            if t['miss'] <= _MAX_MISS:
                surviving.append(t)
        self._tracks = surviving

        for di, r in enumerate(raw_rects):
            if di not in used_det:
                self._tracks.append({
                    'id': self._next_id,
                    'rf': [float(v) for v in r],
                    'ri': list(r),
                    'probs': raw_probs_list[di].copy(),
                    'hits': 1, 'miss': 0, 'ok': False,
                })
                self._next_id += 1

        return [(t['id'], t['ri'], t['probs'])
                for t in self._tracks if t['ok'] and t['miss'] <= _MAX_MISS]

def load_model(ckpt_path, device, architecture="resnet18", state_key="model_state"):
    """Load a trained emotion classifier.

    Supported architectures:
      - resnet18        → torchvision ResNet-18 with custom fc (7-class)
      - efficientnet_b0 → timm EfficientNet-B0 (auto-adapts grayscale→RGB, 8→7 class)
      - effnet_tv       → torchvision EfficientNet-B0
    """
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt_data.get(state_key, ckpt_data)

    if architecture == "efficientnet_b0":
        import timm
        is_gray = sd.get("conv_stem.weight", torch.zeros(1)).shape[1] == 1
        in_chans = 1 if is_gray else 3
        num_ckpt_classes = sd.get("classifier.weight", torch.zeros(7, 1280)).shape[0]

        m = timm.create_model("efficientnet_b0", pretrained=False,
                              in_chans=in_chans, num_classes=num_ckpt_classes)
        m.load_state_dict(sd)

        # Adapt grayscale → RGB
        if is_gray:
            w = m.conv_stem.weight.data
            new_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            new_conv.weight.data = w.repeat(1, 3, 1, 1) / 3.0
            m.conv_stem = new_conv

        # Adapt 8-class (FER+) → 7-class
        if num_ckpt_classes == 8:
            old_cls = m.classifier
            new_cls = nn.Linear(old_cls.in_features, 7)
            new_cls.weight.data = old_cls.weight.data[:7, :]
            new_cls.bias.data = old_cls.bias.data[:7]
            m.classifier = new_cls
        elif num_ckpt_classes != 7:
            old_cls = m.classifier
            new_cls = nn.Linear(old_cls.in_features, 7)
            nn.init.xavier_uniform_(new_cls.weight)
            nn.init.zeros_(new_cls.bias)
            m.classifier = new_cls

        return m.to(device).eval()

    # ── Standard architectures (single load_state_dict) ──
    if architecture == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 7)
    elif architecture == "effnet_tv":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    m.load_state_dict(sd)
    return m.to(device).eval()

def preprocess_face(face_bgr, img_size):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(face_pil).unsqueeze(0)

_BIAS_CACHE = {}

def _logit_bias_on(device):
    key = str(device)
    if key not in _BIAS_CACHE:
        _BIAS_CACHE[key] = torch.tensor(LOGIT_BIAS, dtype=torch.float32, device=device)
    return _BIAS_CACHE[key]

@torch.no_grad()
def predict_batch(model, faces_tensor, device):
    if faces_tensor is None:
        return np.array([])
    x = faces_tensor.to(device)
    logits = model(x) + _logit_bias_on(device)
    return torch.softmax(logits / SOFTMAX_TEMPERATURE, 1).cpu().numpy()


def _predict_hsemotion(recognizer, face_bgr):
    """Run HSEmotion enet_b2_7 on a single BGR face crop and return a 7-vector
    of probabilities in our EMO order. No logit bias / temperature applied —
    HSEmotion is trained on AffectNet with balanced sampling, so the prior
    correction we use for best.pt is not needed (and would skew its output).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    _, scores = recognizer.predict_emotions(face_rgb, logits=False)
    # scores is in HSE_LABELS order; reindex to our EMO order.
    return np.array([scores[HSE_LABELS.index(e)] for e in EMO], dtype=np.float32)

def draw_panel(frame, faces_probs, face_rects):
    """在帧上绘制检测结果"""
    h, w = frame.shape[:2]

    for probs, (fx, fy, fw, fh) in zip(faces_probs, face_rects):
        # 取 top-1
        top_idx = int(np.argmax(probs))
        top_emo = EMO[top_idx]
        top_conf = float(probs[top_idx])
        color = EMO_COLOR.get(top_emo, (255, 255, 255))

        # 画人脸框
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

        # 顶栏：情绪 + 置信度
        label = f"{top_emo}  {top_conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (fx, fy - lh - 10), (fx + lw + 8, fy), color, -1)
        cv2.putText(frame, label, (fx + 4, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(frame, label, (fx + 4, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # top-3 概率条：优先画在右侧，空间不足则画在左侧
        bar_w = 120
        bar_h = 16
        bar_x = fx + fw + 8
        if bar_x + bar_w > w:        # 右侧超出画面，改为左侧
            bar_x = max(0, fx - bar_w - 8)
        order = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(order):
            by = fy + rank * (bar_h + 4)
            conf = float(probs[idx])
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * conf)
            cv2.rectangle(frame, (bar_x, by), (bar_x + fill_w, by + bar_h),
                          EMO_COLOR.get(EMO[idx], (200, 200, 200)), -1)
            text = f"{EMO[idx]:>10s}  {conf:.0%}"
            cv2.putText(frame, text, (bar_x + 3, by + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return frame

import threading

class EngineFER:
    def __init__(self):
        from backend.config import MODEL_BACKEND, MODEL_REGISTRY

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_size = 224
        self.model = None
        self.model_key = None
        self.model_name = None
        self.model_architecture = None
        self._lock = threading.Lock()
        self.tracker = FaceTracker()
        self._loaded: dict = {}
        self._labels: dict = {}
        self._architectures: dict = {}

        # Preload all registry models into RAM so switch_model is a pointer swap.
        # ResNet18 ~45MB each → ~270MB for 7 models, fits any demo machine.
        # HSEmotion (EfficientNet-B2) is a separate path via HSEmotionRecognizer.
        print(f"[engine] Preloading {len(MODEL_REGISTRY)} models …")
        for k, cfg in MODEL_REGISTRY.items():
            ckpt_path = os.path.abspath(cfg["path"])
            if not os.path.exists(ckpt_path):
                print(f"[engine]   {k}: SKIP (missing: {ckpt_path})")
                continue
            try:
                if cfg["architecture"] == "hsemotion_b2_7":
                    from hsemotion.facial_emotions import HSEmotionRecognizer
                    self._loaded[k] = HSEmotionRecognizer(
                        model_name="enet_b2_7", device=str(self.device),
                    )
                else:
                    self._loaded[k] = load_model(
                        ckpt_path, self.device, cfg["architecture"], cfg["state_key"]
                    )
                self._labels[k] = cfg["label"]
                self._architectures[k] = cfg["architecture"]
                print(f"[engine]   {k}: OK ({cfg['label']})")
            except Exception as e:
                print(f"[engine]   {k}: FAIL ({e})")
        print(f"[engine] Preloaded {len(self._loaded)} models: {list(self._loaded.keys())}")

        self.switch_model(MODEL_BACKEND)

        # ── Face detector: MTCNN (matches paper §5.1/§6 + training pipeline) ──
        # The personalized models were fine-tuned on MTCNN crops with the same
        # 20-px margin used at inference, so MTCNN is the source of truth.
        # Kept on CPU when device == "mps" (PyTorch issue #96056 crashes MTCNN
        # in adaptive_avg_pool2d for non-divisible sizes on Metal).
        from facenet_pytorch import MTCNN
        mtcnn_device = "cpu" if self.device == "mps" else self.device
        self.detector = MTCNN(keep_all=True, device=mtcnn_device,
                              post_process=False, min_face_size=60)
        print(f"[engine] MTCNN ready (device={mtcnn_device})")

        # ── GPU warmup so the first real frame doesn't stall on kernel compile.
        if self.device == "cuda":
            try:
                dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
                with torch.no_grad():
                    for key, m in self._loaded.items():
                        if self._architectures.get(key) == "hsemotion_b2_7":
                            continue
                        m(dummy)
                torch.cuda.synchronize()
                print("[engine] GPU warmup complete")
            except Exception as e:
                print(f"[engine] GPU warmup skipped: {e}")

    def switch_model(self, key):
        """Swap the active emotion classifier at runtime.
        Preloaded keys swap in microseconds (pointer assignment). Unknown
        paths fall through to filesystem load. Tracker is reset so the EMA
        prob buffers don't bleed across models.
        """
        from backend.config import MODEL_REGISTRY

        # Fast path: preloaded registry key
        if key in self._loaded:
            with self._lock:
                self.model = self._loaded[key]
                self.model_key = key
                self.model_name = self._labels.get(key, key)
                self.model_architecture = self._architectures.get(key)
                self.tracker.reset()
            return self.model_name

        # Slow path: resolve config / filesystem path and load on demand
        if key in MODEL_REGISTRY:
            cfg = MODEL_REGISTRY[key]
            ckpt_path = os.path.abspath(cfg["path"])
            architecture = cfg["architecture"]
            state_key = cfg["state_key"]
            label = cfg["label"]
            resolved_key = key
        else:
            ckpt_path = os.path.abspath(key)
            architecture = "resnet18"
            state_key = "model_state"
            label = os.path.basename(key)
            resolved_key = None

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[engine] Loading model on demand: {label}")
        if architecture == "hsemotion_b2_7":
            from hsemotion.facial_emotions import HSEmotionRecognizer
            new_model = HSEmotionRecognizer(
                model_name="enet_b2_7", device=str(self.device),
            )
        else:
            new_model = load_model(ckpt_path, self.device, architecture, state_key)

        with self._lock:
            self.model = new_model
            self.model_key = resolved_key
            self.model_name = label
            self.model_architecture = architecture
            self.tracker.reset()
            if resolved_key is not None:
                self._loaded[resolved_key] = new_model
                self._labels[resolved_key] = label
                self._architectures[resolved_key] = architecture
        return label

    def process_frame(self, frame, mode='m0', mode_state=None):
        """Detect faces with MTCNN, classify emotions, return per-face results.
        Crop is padded by 20px on all sides to match the training pipeline
        (paper §5.1: `face_crop.py` MTCNN + padded crop → ResNet18 features).
        Mirroring is delegated to the frontend (CSS transform).
        """
        H, W = frame.shape[:2]

        # MTCNN expects RGB PIL; we hold BGR numpy. Build a PIL once.
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = self.detector.detect(pil)

        raw_rects, raw_probs_list = [], []
        if boxes is not None and len(boxes) > 0:
            using_hse = (self.model_architecture == "hsemotion_b2_7")
            tensors, face_crops = [], []
            margin = 20
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                # Expand by margin then clip to image bounds.
                x1 = max(0, int(x1) - margin); y1 = max(0, int(y1) - margin)
                x2 = min(W, int(x2) + margin); y2 = min(H, int(y2) + margin)
                # Drop faces too small to classify reliably (see MIN_CROP_PX)
                if x2 - x1 < MIN_CROP_PX or y2 - y1 < MIN_CROP_PX:
                    continue
                if using_hse:
                    face_crops.append(frame[y1:y2, x1:x2])
                else:
                    tensors.append(preprocess_face(frame[y1:y2, x1:x2], self.img_size))
                raw_rects.append((x1, y1, x2 - x1, y2 - y1))

            if using_hse and face_crops:
                raw_probs_list = [
                    _predict_hsemotion(self.model, crop) for crop in face_crops
                ]
            elif tensors:
                batch_x = torch.cat(tensors, dim=0)
                raw_probs_list = list(predict_batch(self.model, batch_x, self.device))

        confirmed = self.tracker.update(raw_rects, raw_probs_list)

        results = []
        for tid, r_bbox, r_probs in confirmed:
            dom_idx = int(np.argmax(r_probs))
            results.append({
                'track_id': tid,
                'bbox': list(r_bbox),
                'conf': float(r_probs[dom_idx]),
                'dominant': EMO[dom_idx],
                'probs': {EMO[i]: float(r_probs[i]) for i in range(len(EMO))},
            })
        return results
