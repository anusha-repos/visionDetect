"""
YOLOv3 Object Detector Module for OpenCV Deep Learning
Provides high-performance inference, class filtering, and aesthetic visualization.
"""

import os
import cv2
import time
import base64
import threading
import numpy as np


# Categorization of COCO 80 classes for UI filtering & statistics
CLASS_CATEGORIES = {
    "person": "People",
    "bicycle": "Vehicles", "car": "Vehicles", "motorcycle": "Vehicles", "airplane": "Vehicles",
    "bus": "Vehicles", "train": "Vehicles", "truck": "Vehicles", "boat": "Vehicles",
    "traffic light": "Outdoor", "fire hydrant": "Outdoor", "stop sign": "Outdoor",
    "parking meter": "Outdoor", "bench": "Outdoor",
    "bird": "Animals", "cat": "Animals", "dog": "Animals", "horse": "Animals",
    "sheep": "Animals", "cow": "Animals", "elephant": "Animals", "bear": "Animals",
    "zebra": "Animals", "giraffe": "Animals",
    "backpack": "Accessories", "umbrella": "Accessories", "handbag": "Accessories",
    "tie": "Accessories", "suitcase": "Accessories",
    "frisbee": "Sports", "skis": "Sports", "snowboard": "Sports", "sports ball": "Sports",
    "kite": "Sports", "baseball bat": "Sports", "baseball glove": "Sports",
    "skateboard": "Sports", "surfboard": "Sports", "tennis racket": "Sports",
    "bottle": "Kitchen", "wine glass": "Kitchen", "cup": "Kitchen", "fork": "Kitchen",
    "knife": "Kitchen", "spoon": "Kitchen", "bowl": "Kitchen",
    "banana": "Food", "apple": "Food", "sandwich": "Food", "orange": "Food",
    "broccoli": "Food", "carrot": "Food", "hot dog": "Food", "pizza": "Food",
    "donut": "Food", "cake": "Food",
    "chair": "Furniture", "couch": "Furniture", "potted plant": "Furniture",
    "bed": "Furniture", "dining table": "Furniture", "toilet": "Furniture",
    "tv": "Electronics", "laptop": "Electronics", "mouse": "Electronics",
    "remote": "Electronics", "keyboard": "Electronics", "cell phone": "Electronics",
    "microwave": "Appliances", "oven": "Appliances", "toaster": "Appliances",
    "sink": "Appliances", "refrigerator": "Appliances",
    "book": "Indoor", "clock": "Indoor", "vase": "Indoor", "scissors": "Indoor",
    "teddy bear": "Indoor", "hair drier": "Indoor", "toothbrush": "Indoor"
}

def generate_palette(num_classes):
    """Generate vibrant, distinct HSL-derived colors for bounding boxes."""
    colors = []
    for i in range(num_classes):
        hue = int((i * 137.5) % 360)  # Golden angle distribution
        sat = 85
        light = 55
        # Convert HSL to RGB
        c = (1 - abs(2 * (light / 100) - 1)) * (sat / 100)
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = (light / 100) - c / 2
        
        if 0 <= hue < 60:
            r, g, b = c, x, 0
        elif 60 <= hue < 120:
            r, g, b = x, c, 0
        elif 120 <= hue < 180:
            r, g, b = 0, c, x
        elif 180 <= hue < 240:
            r, g, b = 0, x, c
        elif 240 <= hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
            
        r_val = int((r + m) * 255)
        g_val = int((g + m) * 255)
        b_val = int((b + m) * 255)
        hex_color = f"#{r_val:02x}{g_val:02x}{b_val:02x}"
        colors.append({"bgr": (b_val, g_val, r_val), "rgb": (r_val, g_val, b_val), "hex": hex_color})
    return colors


class YOLOv3Detector:
    def __init__(self, weights_path="yolov3.weights", config_path="yolov3.cfg", classes_path="yolov3.txt"):
        self.weights_path = weights_path
        self.config_path = config_path
        self.classes_path = classes_path
        self.classes = []
        self.lock = threading.Lock()
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")

        with open(classes_path, "r", encoding="utf-8") as f:
            self.classes = [line.strip() for line in f if line.strip()]

        self.palette = generate_palette(len(self.classes))

        print(f"Loading YOLOv3 network from {weights_path} and {config_path}...")
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        layer_names = self.net.getLayerNames()
        try:
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except Exception:
            self.output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        print(f"YOLOv3 detector initialized with {len(self.classes)} classes.")

    def get_classes_metadata(self):
        """Return list of classes with assigned color and category."""
        return [
            {
                "id": i,
                "name": name,
                "category": CLASS_CATEGORIES.get(name, "Other"),
                "color": self.palette[i]["hex"]
            }
            for i, name in enumerate(self.classes)
        ]

    def detect(self, image_np, conf_threshold=0.5, nms_threshold=0.4, resolution=416, allowed_classes=None):
        """
        Run inference on a BGR image array.
        
        :param image_np: numpy array (BGR)
        :param conf_threshold: float (0.01 - 0.99)
        :param nms_threshold: float (0.01 - 0.99)
        :param resolution: int (320, 416, 608)
        :param allowed_classes: set or list of class names to keep (or None for all)
        :return: dict with detections, stats, and base64 rendered image
        """
        if image_np is None or image_np.size == 0:
            raise ValueError("Invalid or empty image provided.")

        orig_h, orig_w = image_np.shape[:2]
        res = int(resolution) if resolution in (320, 416, 608) else 416

        blob = cv2.dnn.blobFromImage(
            image_np,
            scalefactor=0.00392,
            size=(res, res),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )

        with self.lock:
            self.net.setInput(blob)
            start_time = time.perf_counter()
            outs = self.net.forward(self.output_layers)
            inference_time = (time.perf_counter() - start_time) * 1000.0

        boxes = []
        confidences = []
        class_ids = []

        allowed_set = set(allowed_classes) if allowed_classes else None

        for out in outs:
            for detection in out:
                scores = detection[5:]
                cid = int(np.argmax(scores))
                confidence = float(scores[cid])

                if confidence >= conf_threshold:
                    class_name = self.classes[cid]
                    if allowed_set and class_name not in allowed_set:
                        continue

                    cx = int(detection[0] * orig_w)
                    cy = int(detection[1] * orig_h)
                    w = int(detection[2] * orig_w)
                    h = int(detection[3] * orig_h)

                    x = int(cx - (w / 2))
                    y = int(cy - (h / 2))

                    # Clamp coordinates to image boundaries
                    x_clamped = max(0, min(x, orig_w - 1))
                    y_clamped = max(0, min(y, orig_h - 1))
                    w_clamped = max(1, min(w, orig_w - x_clamped))
                    h_clamped = max(1, min(h, orig_h - y_clamped))

                    boxes.append([x_clamped, y_clamped, w_clamped, h_clamped])
                    confidences.append(confidence)
                    class_ids.append(cid)

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        final_detections = []
        annotated_image = image_np.copy()

        if len(indices) > 0:
            for i, item in enumerate(indices):
                idx = item[0] if isinstance(item, (list, tuple, np.ndarray)) else int(item)
                box = boxes[idx]
                conf = confidences[idx]
                cid = class_ids[idx]
                class_name = self.classes[cid]
                color_info = self.palette[cid]
                bgr_color = color_info["bgr"]
                hex_color = color_info["hex"]

                x, y, w, h = box

                # Draw high quality visual bounding box on annotated image
                # 1. Main bounding rectangle with 2px or 3px border
                thickness = max(2, int(min(orig_w, orig_h) / 300))
                cv2.rectangle(annotated_image, (x, y), (x + w, y + h), bgr_color, thickness)

                # 2. Modern pill label banner above or inside box
                label_text = f"{class_name} {conf * 100:.1f}%"
                font_face = cv2.FONT_HERSHEY_DUPLEX
                font_scale = max(0.45, min(orig_w, orig_h) / 1000.0)
                font_thick = 1
                (label_w, label_h), baseline = cv2.getTextSize(label_text, font_face, font_scale, font_thick)

                # Position label above box if space permits, otherwise inside
                pad = 4
                if y - label_h - (pad * 2) >= 0:
                    badge_y1 = y - label_h - (pad * 2)
                    badge_y2 = y
                    text_y = y - pad
                else:
                    badge_y1 = y
                    badge_y2 = y + label_h + (pad * 2)
                    text_y = y + label_h + pad

                badge_x1 = x
                badge_x2 = min(orig_w, x + label_w + (pad * 2))

                # Background badge with class color
                cv2.rectangle(annotated_image, (badge_x1, badge_y1), (badge_x2, badge_y2), bgr_color, -1)
                # Label text in crisp white or dark depending on color lightness
                cv2.putText(
                    annotated_image,
                    label_text,
                    (badge_x1 + pad, text_y),
                    font_face,
                    font_scale,
                    (255, 255, 255),
                    font_thick,
                    cv2.LINE_AA
                )

                # Record detection details
                final_detections.append({
                    "id": i + 1,
                    "class_id": cid,
                    "label": class_name,
                    "category": CLASS_CATEGORIES.get(class_name, "Other"),
                    "confidence": round(conf, 4),
                    "confidence_pct": round(conf * 100, 1),
                    "color": hex_color,
                    "box": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    },
                    "normalized_box": {
                        "x": round(x / orig_w, 4),
                        "y": round(y / orig_h, 4),
                        "width": round(w / orig_w, 4),
                        "height": round(h / orig_h, 4)
                    }
                })

        # Sort detections by confidence descending
        final_detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Encode annotated image to JPEG base64
        _, buffer = cv2.imencode(".jpg", annotated_image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        b64_annotated = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

        # Encode original image to JPEG base64 for before/after comparison
        _, orig_buffer = cv2.imencode(".jpg", image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        b64_original = "data:image/jpeg;base64," + base64.b64encode(orig_buffer).decode("utf-8")

        unique_labels = sorted(list(set(d["label"] for d in final_detections)))
        category_counts = {}
        for d in final_detections:
            cat = d["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "success": True,
            "detections": final_detections,
            "stats": {
                "total_detections": len(final_detections),
                "inference_time_ms": round(inference_time, 1),
                "image_width": orig_w,
                "image_height": orig_h,
                "input_resolution": f"{res}x{res}",
                "conf_threshold": conf_threshold,
                "nms_threshold": nms_threshold,
                "unique_classes": unique_labels,
                "category_counts": category_counts
            },
            "annotated_image": b64_annotated,
            "original_image": b64_original
        }
