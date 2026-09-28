"""
STEP 13 - ANPR + OCR

Detects vehicles with the same YOLO model already used elsewhere
(COCO classes: car=2, motorcycle=3, bus=5, truck=7), crops each
vehicle box, and runs OCR over it to pull out plate text.

This uses EasyOCR directly on the vehicle crop rather than a dedicated
plate-detector model, which keeps the dependency list small; accuracy
improves a lot if you later swap in a plate-specific detector (e.g. a
YOLO model fine-tuned on plates) to crop just the plate region before
OCR - the interface below (`read_plate`) is written so that swap only
touches this file.
"""

import cv2
import numpy as np
import easyocr

VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO)


class PlateReader:
    def __init__(self, languages=("en",), gpu: bool = False):
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def is_vehicle(self, class_id: int) -> bool:
        return class_id in VEHICLE_CLASS_IDS

    def read_plate(self, frame_bgr: np.ndarray, box) -> dict | None:
        """
        box: (x1, y1, x2, y2) vehicle bounding box from the existing
        YOLO detection loop in main.py / pipeline.py.
        Returns {"text": str, "confidence": float} or None if nothing readable.
        """

        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # focus on the lower half of the vehicle box, where plates usually sit
        h = crop.shape[0]
        lower_crop = crop[int(h * 0.55):, :]

        gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        results = self.reader.readtext(gray)

        if not results:
            return None

        # pick the highest-confidence text block that looks plate-like
        best = max(results, key=lambda r: r[2])
        text, confidence = best[1], best[2]

        cleaned = "".join(ch for ch in text if ch.isalnum()).upper()

        if len(cleaned) < 4:
            return None

        return {"text": cleaned, "confidence": float(confidence)}
