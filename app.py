import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

# ------------------- CONFIG -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_SIZE = 224
MODEL_PATH = os.path.join(BASE_DIR, "modelnet_model.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "jpg", "jpeg", "png"}

LABELS = sorted(os.listdir(DATA_DIR)) if os.path.isdir(DATA_DIR) else []

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["TEMPLATES_AUTO_RELOAD"] = True
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- MEDIAPIPE HANDS -------------------
try:
    from mediapipe.python.solutions import hands as mp_hands_module
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
    mp_hands = mp_hands_module
    MEDIAPIPE_AVAILABLE = True
    print("[OK] MediaPipe loaded successfully")
except (ImportError, ModuleNotFoundError) as e:
    MEDIAPIPE_AVAILABLE = False
    mp_hands = None
    mp_drawing = None
    mp_drawing_styles = None
    print(f"[WARN] MediaPipe not available: {e}. Using full frame for prediction.")

if MEDIAPIPE_AVAILABLE:
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
else:
    hands_detector = None

# ------------------- TRAIN MODEL -------------------
def build_and_train_model():
    global LABELS
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential

    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(
            f"[ERROR] Data folder not found at: {DATA_DIR}\n"
            "Please create a 'data/' folder with one subfolder per sign class."
        )

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    train_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="categorical", subset="training"
    )
    val_gen = datagen.flow_from_directory(
        DATA_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="categorical", subset="validation"
    )

    LABELS = list(train_gen.class_indices.keys())
    print(f"[OK] Detected {len(LABELS)} classes: {LABELS}")

    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(LABELS), activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, epochs=5, validation_data=val_gen)
    model.save(MODEL_PATH)
    print(f"[OK] Model trained and saved to {MODEL_PATH}")
    return model

# ------------------- LOAD OR TRAIN -------------------
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        try:
            loaded = load_model(MODEL_PATH)
            print("[OK] Loaded pre-trained model from", MODEL_PATH)
            return loaded
        except Exception as e:
            print(f"[WARN] Failed to load saved model: {e}")
            print("[WARN] Rebuilding and training a new model from the dataset.")
    else:
        print("[INFO] Model not found at", MODEL_PATH, "- starting training...")
    return build_and_train_model()

model = load_or_train_model()

# ------------------- HELPERS -------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_crop_hand(frame):
    if not MEDIAPIPE_AVAILABLE or hands_detector is None:
        return frame
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    h, w, _ = frame.shape
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        x_min = max(0, int(min(x_coords)) - 20)
        x_max = min(w, int(max(x_coords)) + 20)
        y_min = max(0, int(min(y_coords)) - 20)
        y_max = min(h, int(max(y_coords)) + 20)
        cropped = frame[y_min:y_max, x_min:x_max]
        return cropped if cropped.size > 0 else frame
    return None

def preprocess_frame(frame):
    if frame is None or frame.size == 0:
        return None
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_frame(frame):
    if frame is None or frame.size == 0:
        return "Invalid Frame", 0.0
    cropped = detect_and_crop_hand(frame)
    processed = preprocess_frame(cropped if cropped is not None else frame)
    if processed is None:
        return "Preprocessing Error", 0.0
    preds = model.predict(processed, verbose=0)
    class_index = int(np.argmax(preds))
    confidence = float(np.max(preds))
    if class_index >= len(LABELS):
        return "Unknown", confidence
    return LABELS[class_index], confidence

def extract_frames_and_predict(video_path, step=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video file"
    sequence = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            label, _ = predict_frame(frame)
            if label not in ("No Hand Detected", "Invalid Frame", "Preprocessing Error"):
                sequence.append(label)
        frame_count += 1
    cap.release()
    collapsed = []
    for char in sequence:
        if not collapsed or char != collapsed[-1]:
            collapsed.append(char)
    return " ".join(collapsed) if collapsed else "No signs detected"

# ------------------- FLASK ROUTES -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400
    label, conf = predict_frame(img)
    return jsonify({"prediction": label, "confidence": round(conf, 4)})

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(video_path)
    sequence = extract_frames_and_predict(video_path, step=5)
    return jsonify({"prediction": sequence})

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break
        label, conf = predict_frame(frame)
        display_text = f"{label} ({conf:.2f})"
        cv2.putText(frame, display_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route("/predict_live")
def predict_live():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------------- RUN APP -------------------
if __name__ == "__main__":
    print(f"[INFO] Labels loaded: {LABELS}")
    print(f"[INFO] MediaPipe available: {MEDIAPIPE_AVAILABLE}")
    print("[INFO] Starting Flask on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
