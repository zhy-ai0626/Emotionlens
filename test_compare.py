"""Standalone diagnostic — compare best.pt vs HSEmotion (enet_b2_7) side-by-side
on a live webcam feed. Use this to check whether the perceived low accuracy is
a model problem or an inference-pipeline problem.

Run:
    python test_compare.py

Keys:
    ESC / q   quit
    s         save the current frame as compare_<timestamp>.png
"""
from __future__ import annotations
import time
from pathlib import Path

# ── 1. Patch torch.load to default to CPU so HSEmotion (saved on CUDA) loads ──
import torch
_orig_torch_load = torch.load
def _cpu_load(*args, **kwargs):
    kwargs.setdefault("map_location", "cpu")
    return _orig_torch_load(*args, **kwargs)
torch.load = _cpu_load

import numpy as np
import cv2
from PIL import Image
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

ROOT = Path(__file__).resolve().parent
BEST_PT   = ROOT / "final_outputs" / "best.pt"
USER1_PT  = ROOT / "final_outputs" / "personalized_user1.pt"
MARGIN = 20

EMO_OURS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]


def load_resnet18_ckpt(path):
    """Load either best.pt or best_distilled.pt (same schema: dict with
    model_state key + 7-class classifier head)."""
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(ck["model_state"])
    return m.eval()


def preprocess_face(face_bgr, size=224):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(pil).unsqueeze(0)


@torch.no_grad()
def predict_ours(model, face_bgr):
    x = preprocess_face(face_bgr, 224)
    logits = model(x)
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return EMO_OURS[idx].lower(), float(probs[idx]), probs


def predict_hsemotion(rec, face_bgr):
    """HSEmotion expects an RGB numpy array directly (handles resize)."""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    emo, scores = rec.predict_emotions(face_rgb, logits=False)
    return emo.lower(), float(max(scores)), scores


def draw_label(img, text, org, color=(0, 255, 0)):
    cv2.rectangle(img, (org[0] - 2, org[1] - 18),
                  (org[0] + 9 * len(text), org[1] + 4), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    print("[setup] Loading best.pt (original ResNet18) …")
    m_best = load_resnet18_ckpt(BEST_PT)
    print("[setup] Loading personalized_user1.pt …")
    if USER1_PT.exists():
        m_user1 = load_resnet18_ckpt(USER1_PT)
    else:
        m_user1 = None
        print(f"  MISSING: {USER1_PT} — user1 row will be skipped")
    print("[setup] Loading HSEmotion (enet_b2_7) …")
    rec = HSEmotionRecognizer(model_name="enet_b2_7", device="cpu")
    print("[setup] Initialising MTCNN …")
    mtcnn = MTCNN(keep_all=True, device="cpu", post_process=False, min_face_size=60)

    print("[setup] Opening webcam …")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open webcam"); return

    print("\nRunning. Press ESC or q to quit, s to save snapshot.\n")
    last_log = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)  # mirror so user sees natural reflection
        H, W = frame.shape[:2]
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, _ = mtcnn.detect(pil)

        cv2.rectangle(frame, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(frame, "RED = best.pt   CYAN = personalized_user1   GREEN = HSEmotion",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                x1 = max(0, x1 - MARGIN); y1 = max(0, y1 - MARGIN)
                x2 = min(W, x2 + MARGIN); y2 = min(H, y2 + MARGIN)
                if x2 - x1 < 60 or y2 - y1 < 60:
                    continue
                face = frame[y1:y2, x1:x2]

                emo_b, p_b, _ = predict_ours(m_best, face)
                emo_h, p_h, _ = predict_hsemotion(rec, face)
                if m_user1 is not None:
                    emo_u, p_u, _ = predict_ours(m_user1, face)
                else:
                    emo_u, p_u = "n/a", 0.0

                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                draw_label(frame, f"best.pt: {emo_b} {p_b*100:.0f}%",
                           (x1, y1 - 48), (40, 40, 255))
                draw_label(frame, f"user1:   {emo_u} {p_u*100:.0f}%",
                           (x1, y1 - 26), (240, 200, 40))
                draw_label(frame, f"HSE:     {emo_h} {p_h*100:.0f}%",
                           (x1, y1 - 4), (60, 220, 100))

                now = time.time()
                if now - last_log > 1.0:
                    print(f"  best.pt → {emo_b:10s} {p_b*100:5.1f}%   |   "
                          f"user1 → {emo_u:10s} {p_u*100:5.1f}%   |   "
                          f"HSE → {emo_h:10s} {p_h*100:5.1f}%")
                    last_log = now

        cv2.imshow("EmotionLens · best.pt vs HSEmotion", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("s"):
            fn = f"compare_{int(time.time())}.png"
            cv2.imwrite(fn, frame)
            print(f"[saved] {fn}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
