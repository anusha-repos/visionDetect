"""
Flask Web Application for YOLOv3 Object Detection on Localhost
Provides RESTful API and responsive single-page web dashboard.
"""

import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from yolo_detector import YOLOv3Detector

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload limit

# Base workspace directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "static", "samples")

# Initialize detector once at startup
print("Initializing YOLOv3 Object Detector...")
detector = YOLOv3Detector(
    weights_path=os.path.join(BASE_DIR, "yolov3.weights"),
    config_path=os.path.join(BASE_DIR, "yolov3.cfg"),
    classes_path=os.path.join(BASE_DIR, "yolov3.txt")
)
print("YOLOv3 detector ready!")

SAMPLES_META = [
    {
        "id": "dog",
        "filename": "dog.jpg",
        "title": "Dog, Bike & Truck",
        "description": "Classic YOLO benchmark test image with multiple overlapping objects",
        "url": "/static/samples/dog.jpg"
    },
    {
        "id": "cat",
        "filename": "cat-raising-paw.webp",
        "title": "Cat Raising Paw",
        "description": "High resolution playful cat photo for animal class detection",
        "url": "/static/samples/cat-raising-paw.webp"
    }
]


def decode_image_from_request(req):
    """Extract and decode image from file upload, JSON base64, or sample name."""
    # 1. File upload
    if "file" in req.files:
        file = req.files["file"]
        if file.filename != "":
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img, file.filename

    # 2. JSON or Form Data
    data = req.get_json(silent=True) or req.form

    # Check for sample_name
    sample_name = data.get("sample_name") if data else None
    if sample_name:
        safe_name = os.path.basename(sample_name)
        sample_path = os.path.join(SAMPLE_DIR, safe_name)
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, safe_name)
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            if img is not None:
                return img, safe_name

    # Check for base64 encoded image
    b64_image = data.get("image") if data else None
    if b64_image:
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(b64_image)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img, "captured_frame.jpg"
        except Exception as e:
            print("Error decoding base64 image:", e)

    return None, None


@app.route("/")
def index():
    """Render main application page."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check and model capabilities."""
    return jsonify({
        "status": "online",
        "model": "YOLOv3 (Darknet)",
        "classes_count": len(detector.classes),
        "opencv_version": cv2.__version__,
        "python_version": sys.version.split()[0],
        "default_resolution": "416x416"
    })


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Get all 80 COCO classes with categories and colors."""
    return jsonify({
        "success": True,
        "classes": detector.get_classes_metadata()
    })


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Get list of pre-configured sample images."""
    return jsonify({
        "success": True,
        "samples": SAMPLES_META
    })


@app.route("/api/detect", methods=["POST"])
def detect_objects():
    """Main object detection endpoint."""
    try:
        img, filename = decode_image_from_request(request)
        if img is None:
            return jsonify({
                "success": False,
                "error": "No valid image provided. Please upload an image, select a sample, or capture from webcam."
            }), 400

        # Parse inference parameters
        data = request.get_json(silent=True) or request.form

        conf_threshold = float(data.get("conf_threshold", 0.5))
        nms_threshold = float(data.get("nms_threshold", 0.4))
        resolution = int(data.get("resolution", 416))

        # Clamp thresholds
        conf_threshold = max(0.01, min(0.99, conf_threshold))
        nms_threshold = max(0.01, min(0.99, nms_threshold))
        if resolution not in (320, 416, 608):
            resolution = 416

        # Parse class filters
        allowed_classes = data.get("allowed_classes")
        if isinstance(allowed_classes, str) and allowed_classes.strip():
            allowed_classes = [c.strip().lower() for c in allowed_classes.split(",") if c.strip()]
        elif not isinstance(allowed_classes, list):
            allowed_classes = None

        result = detector.detect(
            image_np=img,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            resolution=resolution,
            allowed_classes=allowed_classes
        )

        result["filename"] = filename or "image.jpg"
        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f" YOLOv3 Object Detection Web App running on Localhost")
    print(f" URL: http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
