"""Regression coverage for HSEmotion backend routing.

The test stubs heavyweight CV/ML dependencies so it can run in a lightweight
development environment while still exercising EngineFER.process_frame.
"""
import sys
import threading
import types
import unittest

import numpy as np


def _install_dependency_stubs():
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB = 1
    cv2.cvtColor = lambda image, code: image
    sys.modules["cv2"] = cv2

    torch = types.ModuleType("torch")

    def fake_load(*args, **kwargs):
        return {}

    class FakeTensor:
        def to(self, device):
            return self

    torch.load = fake_load
    torch.no_grad = lambda: (lambda fn: fn)
    torch.cat = lambda tensors, dim=0: FakeTensor()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = object
    torch.nn.Linear = type("Linear", (), {})
    torch.nn.Conv2d = type("Conv2d", (), {})
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn

    torchvision = types.ModuleType("torchvision")
    torchvision.models = types.ModuleType("torchvision.models")
    torchvision.transforms = types.ModuleType("torchvision.transforms")
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.models"] = torchvision.models
    sys.modules["torchvision.transforms"] = torchvision.transforms

    return FakeTensor


FakeTensor = _install_dependency_stubs()

from backend import engine_fer  # noqa: E402

engine_fer.preprocess_face = lambda face, size: FakeTensor()


class HSEmotionRecognizer:
    def predict_emotions(self, image, logits=False):
        # HSE_LABELS order: anger, disgust, fear, happiness, neutral, sadness, surprise
        return "neutral", np.array(
            [0.05, 0.05, 0.05, 0.15, 0.55, 0.10, 0.05],
            dtype=np.float32,
        )


class Detector:
    def detect(self, image):
        return (
            np.array([[20, 20, 180, 180]], dtype=np.float32),
            np.array([0.99], dtype=np.float32),
        )


class Tracker:
    def __init__(self):
        self.probabilities = None
        self.reset_count = 0

    def update(self, rectangles, probabilities):
        self.probabilities = probabilities
        return []

    def reset(self):
        self.reset_count += 1


class HSEmotionRoutingTest(unittest.TestCase):
    def test_model_switch_keeps_architecture_metadata(self):
        recognizer = HSEmotionRecognizer()
        engine = engine_fer.EngineFER.__new__(engine_fer.EngineFER)
        engine._loaded = {"general": recognizer}
        engine._labels = {"general": "General"}
        engine._architectures = {"general": "hsemotion_b2_7"}
        engine._lock = threading.Lock()
        engine.tracker = Tracker()

        label = engine.switch_model("general")

        self.assertEqual(label, "General")
        self.assertIs(engine.model, recognizer)
        self.assertEqual(engine.model_architecture, "hsemotion_b2_7")
        self.assertEqual(engine.tracker.reset_count, 1)

    def test_general_hsemotion_uses_predict_emotions_not_model_call(self):
        engine = engine_fer.EngineFER.__new__(engine_fer.EngineFER)
        engine.model_key = "general"
        engine.model_architecture = "hsemotion_b2_7"
        engine.model = HSEmotionRecognizer()
        engine.device = "cpu"
        engine.img_size = 224
        engine.detector = Detector()
        engine.tracker = Tracker()

        result = engine.process_frame(np.zeros((200, 200, 3), dtype=np.uint8))

        self.assertEqual(result, [])
        self.assertEqual(len(engine.tracker.probabilities), 1)
        # Remapped EMO order starts with neutral then happiness.
        self.assertAlmostEqual(engine.tracker.probabilities[0][0], 0.55)
        self.assertAlmostEqual(engine.tracker.probabilities[0][1], 0.15)


if __name__ == "__main__":
    unittest.main()
