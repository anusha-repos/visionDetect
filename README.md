# YOLO Vision AI • Deep Learning Object Detection with OpenCV

A modern, interactive deep learning object detection application powered by **YOLOv3 (Darknet)** and **OpenCV DNN**, packaged with a clean **Sky Blue & White** web interface.

![YOLO Object Detection](cat.jpg)

## ✨ Features

* **🎯 Real-Time 80-Class Detection**

  * Detects 80 different object classes from the MS COCO dataset.
  * Supports people, vehicles, animals, electronics, food, sports items, and many more.

* **💻 Interactive Web Dashboard**

  * **4 Input Sources:**

    * Built-in sample images
    * Drag-and-drop image upload
    * Live webcam streaming
    * Image URL input

  * **Customizable Sensitivity**

    * Confidence threshold: **10%–95%**
    * NMS IoU threshold adjustment

  * **4 View Modes:**

    1. **Annotated** — Displays bounding boxes and detected object labels.
    2. **Interactive HUD** — Interactive detection boxes synchronized with detection cards.
    3. **Compare Split Slider** — Compares the original image with the detected image.
    4. **Original View** — Displays the original image without annotations.

* **⚡ Performance Metrics**

  * Inference time
  * Number of detected objects
  * Detected COCO categories
  * Confidence scores

* **📥 One-Click Export**

  * Download annotated images.
  * Export detection results as structured JSON.

* **🌐 REST API**

  * Easily integrate object detection into other applications.

* **📟 CLI Support**

  * Supports direct object detection through the command line using `yolo_opencv.py`.

## 🧠 Technology Stack

| Technology          | Purpose                           |
| ------------------- | --------------------------------- |
| Python              | Core programming language         |
| YOLOv3              | Object detection model            |
| Darknet             | YOLO model architecture           |
| OpenCV DNN          | Deep learning inference           |
| Flask               | Web application and REST API      |
| NumPy               | Numerical processing              |
| HTML/CSS/JavaScript | Interactive web interface         |
| MS COCO             | 80-class object detection dataset |

## 📂 Project Structure

```text
YOLO-Vision-AI/
│
├── app.py
├── yolo_opencv.py
├── yolov3.cfg
├── yolov3.weights
├── yolov3.txt
│
├── cat.jpg
├── dog.jpg
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── uploads/
│
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd YOLO-Vision-AI
```

### 2. Install Dependencies

Make sure Python 3.8 or later is installed.

```bash
pip install flask opencv-python numpy
```

### 3. Download YOLOv3 Weights

Download the pre-trained **YOLOv3 weights** file.

The file is approximately **248 MB**.

Official Darknet weights:

[Download YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights?utm_source=chatgpt.com)

Place the downloaded file in the project root directory:

```text
YOLO-Vision-AI/
└── yolov3.weights
```

### 4. Verify Required Files

Make sure these files are available:

```text
app.py
yolo_opencv.py
yolov3.cfg
yolov3.weights
yolov3.txt
cat.jpg
```

### 5. Start the Application

Run:

```bash
python app.py
```

The Flask server will start locally.

Open your browser and visit:

```text
http://localhost:5000
```

## 📸 Object Detection

The application can detect multiple objects in an image.

For example, an image containing a cat may produce:

```text
Object: cat
Confidence: 92%
```

Bounding boxes are automatically drawn around detected objects.

## 📟 CLI Usage

You can also run object detection directly from the command line:

```bash
python yolo_opencv.py --image cat.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```

This processes the image and displays the detected objects with their bounding boxes and confidence scores.

## 🌐 REST API

The application provides the following API endpoints:

| Endpoint       | Method | Description                     |
| -------------- | ------ | ------------------------------- |
| `/`            | GET    | Main web application            |
| `/api/detect`  | POST   | Runs YOLO object detection      |
| `/api/health`  | GET    | Returns model and system health |
| `/api/classes` | GET    | Returns all 80 COCO classes     |
| `/api/samples` | GET    | Returns available sample images |

### `/api/detect`

Runs YOLO inference on:

* Uploaded images
* Sample images
* Base64 encoded frames

Example response:

```json
{
  "success": true,
  "detections": [
    {
      "class": "cat",
      "confidence": 0.92,
      "box": {
        "x": 120,
        "y": 80,
        "width": 300,
        "height": 250
      }
    }
  ]
}
```

## 🎯 Supported Object Categories

YOLOv3 is trained on the **MS COCO dataset** and can recognize 80 object classes, including:

* Person
* Car
* Bus
* Truck
* Motorcycle
* Bicycle
* Dog
* Cat
* Horse
* Bird
* Bottle
* Chair
* Laptop
* Cell phone
* TV
* Keyboard
* Mouse
* Backpack
* Suitcase
* Sports equipment
* Food items
* Household objects

and many more.

## 📊 Detection Workflow

```text
Input Image
     ↓
OpenCV Image Processing
     ↓
YOLOv3 Model
     ↓
Object Detection
     ↓
Confidence Filtering
     ↓
Non-Maximum Suppression
     ↓
Bounding Boxes
     ↓
Detected Objects + Confidence
```

## 🔍 How It Works

The application uses the YOLOv3 deep learning model to identify objects inside images.

1. The user provides an image.
2. OpenCV loads and preprocesses the image.
3. The image is passed to the YOLOv3 neural network.
4. YOLO predicts possible objects and their locations.
5. Confidence scores are calculated.
6. Low-confidence detections are removed.
7. Non-Maximum Suppression removes overlapping duplicate boxes.
8. The final bounding boxes and labels are displayed.
9. Detection information can be exported as JSON.

## ⚙️ Detection Settings

The web interface allows users to adjust:

### Confidence Threshold

Controls how confident the model must be before displaying a detection.

```text
Lower value  → More detections
Higher value → Fewer but more confident detections
```

### NMS IoU Threshold

Controls how overlapping bounding boxes are filtered.

This helps prevent multiple boxes from being displayed around the same object.

## 📈 Performance Metrics

The dashboard provides:

* Detection count
* Inference time
* Object categories
* Confidence scores
* Bounding box coordinates

These metrics help understand the performance of the object detection model.

## 🖼️ Example

The application can process a cat image and produce an output similar to:

```text
Input:
Cat image

Output:
┌───────────────────────────┐
│                           │
│        ┌─────────┐        │
│        │   CAT   │        │
│        │  92%    │        │
│        └─────────┘        │
│                           │
└───────────────────────────┘
```

## 💡 Applications

YOLO object detection can be used in many real-world applications, including:

* Smart surveillance
* Traffic monitoring
* Autonomous vehicles
* Wildlife monitoring
* Retail analytics
* Industrial inspection
* Security systems
* Robotics
* Smart cameras
* Crowd monitoring
* Image analysis

## 🛠️ Troubleshooting

### YOLO weights not found

Make sure:

```text
yolov3.weights
```

exists in the project root directory.

### OpenCV not installed

Run:

```bash
pip install opencv-python
```

### Flask not installed

Run:

```bash
pip install flask
```

### NumPy not installed

Run:

```bash
pip install numpy
```

### Port already in use

If port `5000` is already being used, stop the existing Flask process or configure the application to use another port.

## 📌 Project Highlights

* YOLOv3-based object detection
* OpenCV DNN inference
* 80-class COCO detection
* Interactive web dashboard
* Image upload support
* Webcam support
* URL-based image detection
* Adjustable confidence threshold
* Adjustable NMS threshold
* Bounding box visualization
* Detection statistics
* JSON export
* REST API
* Command-line support

## 📄 License

This project is licensed under the **MIT License**.

---

## 👩‍💻 Project

**YOLO Vision AI**

Deep Learning Object Detection using **YOLOv3 + OpenCV + Flask**.

Built for learning, experimentation, and real-world computer vision applications.
