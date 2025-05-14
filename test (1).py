import datetime
import threading
import flask
from flask import Flask, request, jsonify, render_template
import cv2
# *** ADDED BACK ***
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import math
import base64
import numpy as np
import io
from PIL import Image
from typing import Dict, Optional, Any, Tuple, List
import torch
import torchvision.transforms.functional as F # Keep for DeepLab preprocess
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation # Keep SegFormer
import traceback
import os
import time
from flask_cors import CORS
import cv2
import numpy as np
from scipy import ndimage
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# *** UPDATED *** Reflecting hybrid approach
RAW_IMAGE_DIR = "raw_measure_images"


global current_req, request_no
request_no = 0
current_req = "requests"

if not os.path.exists(current_req):
    os.makedirs(current_req); print(f"Created debug directory: {current_req}")
# *** NEW *** Create raw image directory
if not os.path.exists(RAW_IMAGE_DIR):
    os.makedirs(RAW_IMAGE_DIR); print(f"Created raw image directory: {RAW_IMAGE_DIR}")
# --- Model Setup (DeepLabV3, SegFormer, MediaPipe) ---

# --- DeepLabV3 Model Setup (for Masking) ---
DEEPLAB_MODEL_LOADED = False
# ... (Keep DeepLabV3 loading code as before) ...
deeplab_model = None
deeplab_preprocess = None
DEEPLAB_PERSON_CLASS_INDEX = -1
print("Loading DeepLabV3 model (for masking)...")
start_time = time.time()
try:
    deeplab_weights = DeepLabV3_ResNet101_Weights.DEFAULT
    deeplab_model = deeplabv3_resnet101(weights=deeplab_weights)
    deeplab_model.eval()
    deeplab_model.to(DEVICE)
    deeplab_preprocess = deeplab_weights.transforms()
    deeplab_class_names = deeplab_weights.meta["categories"]
    try:
        DEEPLAB_PERSON_CLASS_INDEX = deeplab_class_names.index('person')
        print(f"DeepLab 'person' class index: {DEEPLAB_PERSON_CLASS_INDEX}")
        DEEPLAB_MODEL_LOADED = True
    except ValueError: print("ERROR: 'person' class not found in DeepLab categories.")
except Exception as e: print(f"FATAL ERROR: Could not load DeepLabV3 model: {e}")
print(f"DeepLabV3 loaded: {DEEPLAB_MODEL_LOADED} ({time.time() - start_time:.2f}s)")

# --- SegFormer Model Setup (for Bounding Box) ---
SEGFORMER_MODEL_LOADED = False
# ... (Keep SegFormer loading code as before) ...
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
segformer_processor = None
segformer_model = None
SEGFORMER_PERSON_CLASS_ID = -1
print(f"Loading SegFormer model ({SEGFORMER_MODEL_NAME}) for bounding box...")
start_time = time.time()
try:
    segformer_processor = AutoImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    segformer_model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME).to(DEVICE)
    segformer_model.eval()
    id2label = segformer_model.config.id2label
    found_person = False
    for id_val_str, label in id2label.items():
        try:
            id_val = int(id_val_str)
            if label.lower() == 'person': SEGFORMER_PERSON_CLASS_ID = id_val; found_person = True; break
        except ValueError: continue
    if not found_person: print(f"Warning: SegFormer 'person' ID not verified. Assuming {SEGFORMER_PERSON_CLASS_ID}.")
    if SEGFORMER_PERSON_CLASS_ID == -1: SEGFORMER_PERSON_CLASS_ID = 12
    SEGFORMER_MODEL_LOADED = True
except Exception as e: print(f"FATAL ERROR: Could not load SegFormer model: {e}")
print(f"SegFormer loaded: {SEGFORMER_MODEL_LOADED} ({time.time() - start_time:.2f}s)")

# --- MediaPipe Pose Setup (for Landmark Relative Y) ---
# *** ADDED BACK ***
MP_POSE_LOADED = False
# ... (Keep MediaPipe Pose loading code as before) ...
mp_pose = mp.solutions.pose
pose_processor = None
print("Loading MediaPipe Pose model...")
start_time = time.time()
try:
    pose_processor = mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
    MP_POSE_LOADED = True
except Exception as e: print(f"FATAL ERROR: Could not load MediaPipe Pose model: {e}")
print(f"MediaPipe Pose loaded: {MP_POSE_LOADED} ({time.time() - start_time:.2f}s)")


# --- Body Part Localization Config ---
# <<<--- REMOVED BODY_PART_RELATIVE_Y --- >>>
# *** ADDED *** Offset ratios for Chest/Waist relative to Shoulder/Hip landmarks
# These might need tuning based on visual results
CHEST_Y_OFFSET_RATIO = 0.15 # % of shoulder-hip distance below shoulder landmark
WAIST_Y_OFFSET_RATIO = 0.15 # % of shoulder-hip distance above hip landmark

# --- Calibration Data (Ground Truth) ---
# ... (Keep calibration_guide as before) ...
calibration_guide = {
    'shoulder_width': 40.64, 'chest_circ': 96.52, 'hip_width': 48.10,
    'waist_circ': 92.0, 'thigh_circ': 56.0,
}

# --- Global State (Multi-Factor) ---
# ... (Keep calibration_factors and is_calibrated as before) ...
calibration_factors: Dict[str, Optional[float]] = {
    'shoulder_width': None, 'hip_width': None, 'chest_circ': None,
    'waist_circ': None, 'thigh_circ': None,
}
is_calibrated: bool = False

COUNTER_FILE = os.path.join(os.path.dirname(__file__), "request_counter.txt") # Store alongside script
request_counter = 0
counter_lock = threading.Lock() # To prevent race conditions if using threads/multiple workers



import math

# ─── Size Labels ────────────────────────────────────────────────
SIZE_MAPPING          = {"XS": 0, "S": 1, "M": 2, "L": 3, "XL": 4, "XXL": 5}
SIZE_REVERSE_MAPPING  = {v: k for k, v in SIZE_MAPPING.items()}
SIZES                 = ["XS", "S", "M", "L", "XL", "XXL"]

# ─── Asian-fit Chest Thresholds (upper bounds), in CM ──────────
# (Dropshipping.com chart: XS 32–34″, S 34–35.5″, M 35.5–37.5″, L 37.5–40″, XL 40–42″)
# 1″ = 2.54 cm
THRESHOLDS = {
    'asian': {
        'chest': [
            34 * 2.54,    # XS max → 86.36 cm
            35.5 * 2.54,  # S  max → 90.17 cm
            37.5 * 2.54,  # M  max → 95.25 cm
            40 * 2.54,    # L  max → 101.60 cm
            42 * 2.54,    # XL max → 106.68 cm
            math.inf      # XXL and above
        ]
    }
}

# ─── Exact Asian→Western Letter-Size Conversion ────────────────
# (From your “Clothing Size Conversion” table)
ASIAN_TO_WESTERN = {
    "XS": "XS",
    "S":  "S",
    "M":  "S",   # Asian M → Western S
    "L":  "M",   # Asian L → Western M
    "XL": "M",   # Asian XL → Western M
    "XXL":"L",  # Asian XXL → Western L
}

# ─── Exact Asian→European Letter-Size Conversion ───────────────
# (From your “Clothing Size Conversion” table)
ASIAN_TO_EUROPEAN = {
    "XS": "XXS",
    "S":  "XS",
    "M":  "S",
    "L":  "M",
    "XL": "L",
    "XXL":"XL",
}


def _get_size_from_measurement(chest_cm: float) -> Optional[str]:
    """Finds the Asian size label given a chest measurement in cm."""
    if chest_cm <= 0:
        return None
    for idx, upper_bound in enumerate(THRESHOLDS['asian']['chest']):
        if chest_cm <= upper_bound:
            return SIZES[idx]
    return SIZES[-1]  # XXL for anything above


def get_regional_clothing_sizes_enhanced(measurements: Dict[str, float]) -> Dict[str, Optional[str]]:
    """
    Given a dict with 'Chest Circumference' in cm, returns:
      - 'asian':   the direct Asian fit size
      - 'western': the mapped Western size
      - 'european': the mapped European size
    """
    chest = measurements.get("Chest Circumference", 0)
    asian_size = _get_size_from_measurement(chest)
    return {
        'asian':    asian_size,
        'western':  ASIAN_TO_WESTERN.get(asian_size) if asian_size else None,
        'european': ASIAN_TO_EUROPEAN.get(asian_size) if asian_size else None
    }




def load_or_initialize_counter():
    """Loads counter from file or initializes to 0 if file doesn't exist/invalid."""
    global request_counter
    try:
        if os.path.exists(COUNTER_FILE):
            with open(COUNTER_FILE, 'r') as f:
                content = f.read().strip()
                request_counter = int(content)
                print(f"Loaded request counter: {request_counter}")
        else:
            request_counter = 0 # Start from 0 if file doesn't exist
            print("Counter file not found, initializing counter to 0.")
            # Optionally write the initial value
            save_counter(request_counter)
    except (ValueError, IOError) as e:
        print(f"Error loading counter file '{COUNTER_FILE}': {e}. Resetting counter to 0.")
        request_counter = 0
    except Exception as e:
        print(f"Unexpected error loading counter: {e}. Resetting counter to 0.")
        request_counter = 0

def save_counter(count):
    """Saves the current counter value to the file."""
    try:
        # Write operation should be atomic enough for this simple case,
        # but lock ensures safety if multiple threads/workers exist.
        with counter_lock:
            with open(COUNTER_FILE, 'w') as f:
                f.write(str(count))
    except IOError as e:
        print(f"Error saving counter file '{COUNTER_FILE}': {e}")
    except Exception as e:
        print(f"Unexpected error saving counter: {e}")

# --- Helper Functions ---

# base64_to_image, generate_human_mask_deeplab, calculate_mask_width_at_y
# remain unchanged. Include them here.
# ... (Paste functions here) ...
def base64_to_image(image_base64: str) -> Optional[np.ndarray]:
    """Convert base64 string to OpenCV image (BGR)."""
    try:
        image_bytes = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e: print(f"Error converting base64 to image: {e}"); return None

def generate_human_mask_deeplab(image_np: np.ndarray, debug_filename_prefix: Optional[str] = None) -> Optional[np.ndarray]:
    """Generates DeepLab mask and optional overlay."""
    if not DEEPLAB_MODEL_LOADED: return None # Simplified check
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        original_width, original_height = image_pil.size
        input_tensor = deeplab_preprocess(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)
        with torch.no_grad(): output = deeplab_model(input_batch)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        binary_mask_np = np.where(output_predictions == DEEPLAB_PERSON_CLASS_INDEX, 255, 0).astype(np.uint8)
        binary_mask_resized = cv2.resize(binary_mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        if debug_filename_prefix:
            try:
                colored_mask_viz = np.zeros_like(image_np); colored_mask_viz[binary_mask_resized == 255] = (0, 255, 0)
                overlay_img = cv2.addWeighted(image_np, 1, colored_mask_viz, 0.4, 0)
                cv2.imwrite(os.path.join(current_req, f"debug_{debug_filename_prefix}_deeplab_overlay.jpg"), overlay_img)
                
            except Exception as e_overlay: print(f"Warning: Failed DeepLab overlay save: {e_overlay}")
        return binary_mask_resized
    except Exception as e: print(f"Error during DeepLab mask generation: {e}"); return None

def calculate_mask_width_at_y(mask: np.ndarray, y: int, band_height: int = 5) -> Optional[float]:
    """Calculates average mask width in a horizontal band."""
    if mask is None: return None
    if y < 0 or y >= mask.shape[0]: return None
    if band_height <= 0: band_height = 1
    height = mask.shape[0]; half_band = band_height // 2
    start_row = max(0, y - half_band); end_row = min(height, y + half_band + (band_height % 2))
    row_widths = []
    for current_y in range(start_row, end_row):
        row = mask[current_y, :]; white_pixels = np.where(row == 255)[0]
        if len(white_pixels) > 0:
            width = float(np.max(white_pixels) - np.min(white_pixels) + 1)
            if width > 0: row_widths.append(width)
    if not row_widths: return 0.0
    return np.mean(row_widths)

# --- Constants for Distance Check (ADD THESE near the top) ---
# Ratios relative to image height. Tune these based on testing!
MIN_PERSON_HEIGHT_RATIO = 0.40  # Person must occupy at least 40% of image height
MAX_PERSON_HEIGHT_RATIO = 1.0  # Person shouldn't occupy more than 90% (avoids extreme closeups)
MIN_CONTOUR_AREA_PIXELS = 10000 # Minimum pixel area for a contour to be considered a person (tune this)

# --- UPDATED Y-Coordinate Function (Hybrid Approach) ---
def get_body_part_y_coordinates_hybrid(
    image_np: np.ndarray,
    debug_filename_prefix: Optional[str] = None
) -> Tuple[Optional[str], Optional[Dict[str, int]]]: # <-- UPDATED RETURN TYPE
    """
    Hybrid approach: Uses SegFormer for bounding box checks & MediaPipe for relative landmarks.
    Calculates absolute Y coordinates for measurement.
    Returns:
        Tuple[Optional[str], Optional[Dict[str, int]]]:
            - str: Error code ('no_person', 'multiple_people', 'too_far', 'too_close',
                   'no_landmarks', 'segformer_error', 'mediapipe_error', 'calculation_error') or None if successful.
            - Dict: Dictionary of Y-coordinates if successful, otherwise None.
    """
    if not SEGFORMER_MODEL_LOADED or not MP_POSE_LOADED:
        print("Error: SegFormer or MediaPipe model not loaded for hybrid localization.")
        return "model_load_error", None

    image_height, image_width, _ = image_np.shape
    estimated_y_coords = {}

    # --- 1. Get Bounding Box & Perform Checks using SegFormer ---
    try:
        image_pil_seg = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        inputs = segformer_processor(images=image_pil_seg, return_tensors="pt").to(DEVICE)
        with torch.no_grad(): outputs = segformer_model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(image_height, image_width), mode="bilinear", align_corners=False)
        seg_map = logits.argmax(dim=1)[0].cpu().numpy()
        person_mask = (seg_map == SEGFORMER_PERSON_CLASS_ID).astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= MIN_CONTOUR_AREA_PIXELS]

        # *** DETECTION LOGIC ***
        if not valid_contours:
            print("SegFormer found no valid contours meeting minimum area.")
            # Optionally save the mask for debugging why no one was found
            if debug_filename_prefix:
                 cv2.imwrite(os.path.join(current_req, f"debug_{debug_filename_prefix}_segformer_mask_no_person.jpg"), person_mask * 255)
            return "no_person", None # No person detected

        if len(valid_contours) > 1:
            print(f"SegFormer found {len(valid_contours)} valid contours. Assuming multiple people.")
             # Optionally draw all contours on debug image
            if debug_filename_prefix:
                debug_multi = image_np.copy()
                cv2.drawContours(debug_multi, valid_contours, -1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(current_req, f"debug_{debug_filename_prefix}_segformer_multiple_people.jpg"), debug_multi)
            return "multiple_people", None # Multiple people detected

        # --- Exactly one valid contour found ---
        main_contour = valid_contours[0] # Use the single valid contour
        bbox_x, bbox_y, bbox_w, bbox_h = cv2.boundingRect(main_contour)

        if bbox_h <= 0:
            print("SegFormer bounding box height is zero.")
            return "bbox_error", None # Invalid bounding box

        print(f"Hybrid: SegFormer BBox: y={bbox_y}, h={bbox_h}, w={bbox_w}")

        # *** DISTANCE CHECK ***
        person_height_ratio = bbox_h / image_height
        print(f"Hybrid: Person height ratio: {person_height_ratio:.2f}")
        if person_height_ratio < MIN_PERSON_HEIGHT_RATIO:
            print("Person detected, but appears too far away.")
            return "too_far", None
        if person_height_ratio > MAX_PERSON_HEIGHT_RATIO:
            print("Person detected, but appears too close.")
            return "too_close", None

    except Exception as e_seg:
        print(f"Error during SegFormer bounding box detection: {e_seg}")
        return "segformer_error", None

    # --- 2. Get Landmarks from MediaPipe (Only if SegFormer checks passed) ---
    try:
        image_rgb_mp = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        image_rgb_mp.flags.writeable = False
        results = pose_processor.process(image_rgb_mp)
        image_rgb_mp.flags.writeable = True

        if not results.pose_landmarks:
            print("MediaPipe detected no landmarks.")
            return "no_landmarks", None # Changed error code
        landmarks = results.pose_landmarks.landmark
        keypoints_abs_y = {} # Store absolute Y pixel coordinates
        min_visibility = 0.3

        required_lm_enums = {
            'shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER, # Use consistent side if possible
            'hip': mp_pose.PoseLandmark.RIGHT_HIP,
            'knee': mp_pose.PoseLandmark.RIGHT_KNEE
        }
        missing_landmarks = []
        for name, lm_enum in required_lm_enums.items():
            lm = landmarks[lm_enum.value]
            if lm.visibility >= min_visibility:
                keypoints_abs_y[name] = int(lm.y * image_height)
            else:
                print(f"Warning: MediaPipe landmark '{lm_enum.name}' low visibility ({lm.visibility:.2f}).")
                missing_landmarks.append(lm_enum.name)

        # Check if essential landmarks were found
        if not all(k in keypoints_abs_y for k in ['shoulder', 'hip', 'knee']):
             print(f"Error: Missing essential high-visibility MediaPipe landmarks. Need shoulder, hip, knee. Found: {list(keypoints_abs_y.keys())}. Missing/LowVis: {missing_landmarks}")
             return "no_landmarks", None

        y_shoulder_abs = keypoints_abs_y['shoulder']
        y_hip_abs = keypoints_abs_y['hip']
        y_knee_abs = keypoints_abs_y['knee']
        print(f"Hybrid: MediaPipe Abs Y: Shoulder={y_shoulder_abs}, Hip={y_hip_abs}, Knee={y_knee_abs}")

    except Exception as e_mp:
        print(f"Error during MediaPipe landmark detection: {e_mp}")
        return "mediapipe_error", None

    # --- 3. Calculate Relative Y positions within BBox ---
    rel_y = {}
    try:
        # (Keep the existing logic for calculating rel_y['shoulder'], rel_y['hip'], etc.)
        # ...
        # Calculate relative position only if landmark is within bbox vertically
        if bbox_y <= y_shoulder_abs < bbox_y + bbox_h:
            rel_y['shoulder'] = (y_shoulder_abs - bbox_y) / bbox_h
        else: print("Warning: Shoulder landmark outside SegFormer bbox Y range."); rel_y['shoulder'] = 0.1 # Fallback

        if bbox_y <= y_hip_abs < bbox_y + bbox_h:
             rel_y['hip'] = (y_hip_abs - bbox_y) / bbox_h
        else: print("Warning: Hip landmark outside SegFormer bbox Y range."); rel_y['hip'] = 0.6 # Fallback

        # Knee relative pos isn't directly used for a measurement line, but needed for thigh calc
        rel_y_knee = -1 # Default invalid
        if bbox_y <= y_knee_abs < bbox_y + bbox_h:
            rel_y_knee = (y_knee_abs - bbox_y) / bbox_h
        else: print("Warning: Knee landmark outside SegFormer bbox Y range.")

        # Check relative order
        if rel_y.get('shoulder', 1.0) >= rel_y.get('hip', 0.0):
             print("Warning: Relative Shoulder Y not above Relative Hip Y. Using fallbacks.")
             # Provide safe fallbacks if order is wrong
             rel_y['shoulder'] = 0.15
             rel_y['hip'] = 0.55

        relative_shoulder_hip_dist = rel_y['hip'] - rel_y['shoulder']
        if relative_shoulder_hip_dist <= 0: # Avoid division by zero or negative offset
             print("Warning: Relative shoulder-hip distance invalid. Using fixed offsets.")
             relative_shoulder_hip_dist = 0.4 # Example typical relative distance

        # Calculate target relative Ys based on landmark relatives and offsets
        rel_y['chest'] = rel_y['shoulder'] + CHEST_Y_OFFSET_RATIO * relative_shoulder_hip_dist
        rel_y['waist'] = rel_y['hip'] - WAIST_Y_OFFSET_RATIO * relative_shoulder_hip_dist

        # Thigh: Midpoint between relative hip and relative knee
        # Use absolute knee Y if relative was invalid, but only if knee is below hip
        if rel_y_knee >= 0 and rel_y_knee > rel_y['hip']: # Knee found and below hip
             rel_y['thigh'] = rel_y['hip'] + (rel_y_knee - rel_y['hip']) / 2
        elif y_knee_abs > y_hip_abs: # Absolute knee below absolute hip, use absolute midpoint's relative pos
             abs_y_thigh_mid = (y_hip_abs + y_knee_abs) / 2
             rel_y['thigh'] = (abs_y_thigh_mid - bbox_y) / bbox_h
        else: # Fallback if knee data is unusable
             print("Warning: Knee position invalid for thigh calculation. Using offset from hip.")
             rel_y['thigh'] = rel_y['hip'] + 0.15 # Offset below hip landmark relative pos

        # Ensure calculated relative Ys are within [0, 1] range
        for k in ['chest', 'waist', 'thigh']:
            rel_y[k] = max(0.0, min(rel_y[k], 1.0))

        print(f"Hybrid: Calculated Relative Ys: {rel_y}")
        # ... (end of existing relative Y calculation)

    except ZeroDivisionError:
        print("Error: Division by zero calculating relative Y (bbox_h likely zero).")
        return "calculation_error", None
    except KeyError as e_key:
         print(f"Error: Missing relative key during calculation: {e_key}")
         return "calculation_error", None
    except Exception as e_calc:
        print(f"Error during relative Y calculation: {e_calc}")
        return "calculation_error", None


    # --- 4. Calculate Final Absolute Y Coordinates ---
    final_y_coords = {}
    try:
        # (Keep existing logic for converting rel_y to final_y_coords)
        # ...
        final_y_coords['y_shoulder'] = int(bbox_y + bbox_h * rel_y['shoulder'])
        final_y_coords['y_chest'] = int(bbox_y + bbox_h * rel_y['chest'])
        final_y_coords['y_waist'] = int(bbox_y + bbox_h * rel_y['waist'])
        final_y_coords['y_hip'] = int(bbox_y + bbox_h * rel_y['hip'])
        final_y_coords['y_thigh'] = int(bbox_y + bbox_h * rel_y['thigh'])

        # Clamp final absolute coords to image bounds
        for key, y_val in final_y_coords.items():
            final_y_coords[key] = max(0, min(y_val, image_height - 1))
        # ...
    except Exception as e_final_y:
        print(f"Error calculating final absolute Y coordinates: {e_final_y}")
        return "calculation_error", None

    # --- 5. Debug Drawing (Optional enhancement: Draw BBox and distance status) ---
    if debug_filename_prefix:
        try:
            debug_img_hybrid = image_np.copy()
            # Draw SegFormer BBox (Green if OK)
            cv2.rectangle(debug_img_hybrid, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 255, 0), 2) # Green BBox
            # Draw MediaPipe Landmarks
            if results.pose_landmarks:
                 mp_drawing.draw_landmarks(
                    debug_img_hybrid, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,100,0), thickness=1, circle_radius=2), # Blueish landmarks
                    mp_drawing.DrawingSpec(color=(200,200,200), thickness=1)) # Light connections
            # Draw Final Measurement Lines
            line_colors = {'y_shoulder': (0, 255, 0),'y_chest':(0, 255, 0),'y_waist':(0, 255, 255),'y_hip':(0, 0, 255),'y_thigh':(255, 0, 255)}
            for key, y_val in final_y_coords.items():
                label = key.replace('y_', '').capitalize()
                color = line_colors.get(key, (255, 255, 255))
                cv2.line(debug_img_hybrid, (bbox_x, y_val), (bbox_x + bbox_w, y_val), color, 1)
                cv2.putText(debug_img_hybrid, f"{label} (y={y_val})", (bbox_x + bbox_w + 5, y_val + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Add text about distance check result
            cv2.putText(debug_img_hybrid, f"Height Ratio: {person_height_ratio:.2f} (OK)", (10, image_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


            debug_path = os.path.join(current_req, f"debug_{debug_filename_prefix}_hybrid_localization_success.jpg")
            cv2.imwrite(debug_path, debug_img_hybrid)
        except Exception as e_dbg:
            print(f"Warning: Failed to save hybrid debug image: {e_dbg}")

    # --- Success ---
    return None, final_y_coords # No error code, return coordinates
# --- UPDATED Measurement Conversion ---
# calculate_measurements_cm_multi_factor remains the same as V5
# ... (Paste calculate_measurements_cm_multi_factor here) ...
def calculate_measurements_cm_multi_factor(
    pixel_measurements: Dict[str, Optional[float]], # Use Optional since measurements might fail
    factors: Dict[str, float]
    ) -> Dict[str, Optional[float]]:
    """Converts pixel measurements to cm using specific factors and estimates circumferences."""
    required_factors = ['shoulder_width', 'hip_width', 'chest_circ', 'waist_circ', 'thigh_circ']
    if not all(factors.get(f) is not None and factors[f] > 0 for f in required_factors):
        print(f"Error: Missing/invalid factors: {factors}. Req: {required_factors}")
        return {key: None for key in ['Shoulder Width', 'Chest Circumference', 'Hip Width', 'Hip Circumference', 'Waist Circumference', 'Thigh Circumference']}

    results_cm = {'Shoulder Width': None, 'Chest Circumference': None,'Hip Circumference': None, 'Waist Circumference': None, 'Thigh Circumference': None,}

    try:
        sw_px = pixel_measurements.get('shoulder_width_px')
        if sw_px is not None and sw_px > 0: results_cm['Shoulder Width'] = round(sw_px / factors['shoulder_width'], 2)

        hw_px = pixel_measurements.get('hip_width_px')
        if hw_px is not None and hw_px > 0: results_cm['Hip Width'] = round(hw_px / factors['hip_width'], 2)

        ct_px = pixel_measurements.get('chest_thickness_px')
        if sw_px is not None and ct_px is not None and sw_px > 0 and ct_px > 0:
             chest_circ_px = math.pi * (sw_px + ct_px) / 2
             results_cm['Chest Circumference'] = round(chest_circ_px / factors['chest_circ'], 2)
        else: print("Warn: Skip Chest Circ.")

        ww_px = pixel_measurements.get('waist_width_px')
        wt_px = pixel_measurements.get('waist_thickness_px')
        if ww_px is not None and wt_px is not None and ww_px > 0 and wt_px > 0:
             waist_circ_px = math.pi * (ww_px + wt_px) / 2
             results_cm['Waist Circumference'] = round(waist_circ_px / factors['waist_circ'], 2)
        else: print("Warn: Skip Waist Circ.")

        thw_px = pixel_measurements.get('thigh_width_px')
        tht_px = pixel_measurements.get('thigh_thickness_px')
        if thw_px is not None and tht_px is not None and thw_px > 0 and tht_px > 0:
             thigh_circ_px = math.pi * (thw_px + tht_px) / 2
             results_cm['Thigh Circumference'] = round(thigh_circ_px / factors['thigh_circ'], 2)
        else: print("Warn: Skip Thigh Circ.")

        ht_px = pixel_measurements.get('hip_thickness_px')
        hip_width_cm = results_cm.get('Hip Width')
        hip_thickness_cm = None
        if ht_px is not None and ht_px > 0: hip_thickness_cm = ht_px / factors['hip_width'] # APPROX

        if hip_width_cm is not None and hip_thickness_cm is not None:
            hip_circ_cm = math.pi * (hip_width_cm + hip_thickness_cm) / 2
            results_cm['Hip Circumference'] = round(hip_circ_cm, 2)
        else: print("Warn: Skip Hip Circ.")

    except KeyError as e: print(f"Error: Missing key in conversion: {e}")
    except ZeroDivisionError: print("Error: Factor is zero.")
    except Exception as e: print(f"Error in conversion: {e}")
    return results_cm







# --- Flask App ---
app = Flask(__name__)
CORS(app)

# --- UPDATED Calibration Route ---
@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Calibrates using FRONT/SIDE images. Hybrid Y coords. Per-part factors.
    Handles tuple return from coordinate estimation.
    """
    global calibration_factors, is_calibrated
    data = request.get_json()
    if not data or 'image_calibrate_front' not in data or 'image_calibrate_side' not in data: return jsonify({"error": "Missing calibration image data"}), 400
    if not DEEPLAB_MODEL_LOADED or not SEGFORMER_MODEL_LOADED or not MP_POSE_LOADED:
         return jsonify({"error": "Models not loaded. Cannot calibrate."}), 500

    try:
        image_cal_front_np = base64_to_image(data['image_calibrate_front'])
        image_cal_side_np = base64_to_image(data['image_calibrate_side'])

        # Re-evaluate rotation necessity
        image_cal_front_np = ndimage.rotate(image_cal_front_np, -90)
        image_cal_side_np = ndimage.rotate(image_cal_side_np, -90)

        if image_cal_front_np is None or image_cal_side_np is None: return jsonify({"error": "Error decoding calibration images"}), 400

        height_cal_front, width_cal_front, _ = image_cal_front_np.shape
        height_cal_side, width_cal_side, _ = image_cal_side_np.shape

        # 1. Mask both images
        print("Calibration: Generating DeepLab masks...")
        cal_front_mask = generate_human_mask_deeplab(image_cal_front_np, debug_filename_prefix="calibrate_front")
        cal_side_mask = generate_human_mask_deeplab(image_cal_side_np, debug_filename_prefix="calibrate_side")
        if cal_front_mask is None or cal_side_mask is None: return jsonify({"error": "Failed DeepLab mask generation during calibration"}), 500 # Added context

        # 2. Find y-coordinates using HYBRID method on FRONT view
        print("Calibration: Estimating part locations & performing checks (Hybrid)...")
        # *** FIX: Unpack the tuple correctly ***
        cal_error_code, actual_y_coords = get_body_part_y_coordinates_hybrid(
            image_cal_front_np,
            debug_filename_prefix="calibrate_front"
        )

        # *** FIX: Check the error code first ***
        if cal_error_code:
            print(f"Calibration localization/detection failed with code: {cal_error_code}")
            # Map error codes to user-friendly calibration errors
            if cal_error_code == "no_person":
                return jsonify({"error": "Calibration failed: No person detected in the front image."}), 400
            elif cal_error_code == "multiple_people":
                return jsonify({"error": "Calibration failed: Multiple people detected in the front image."}), 400
            elif cal_error_code == "too_far":
                return jsonify({"error": "Calibration failed: Person appears too far away in the front image."}), 400
            elif cal_error_code == "too_close":
                # This might be the actual intended error based on your previous log (ratio 0.95 vs max 0.90?)
                # Double check MAX_PERSON_HEIGHT_RATIO constant value. It's 0.99 in the code above.
                return jsonify({"error": "Calibration failed: Person appears too close in the front image."}), 400
            elif cal_error_code == "no_landmarks":
                 return jsonify({"error": "Calibration failed: Could not detect key body points clearly in the front image."}), 400
            else: # Handle other internal/model errors as 500
                 return jsonify({"error": f"Calibration failed due to internal analysis error ({cal_error_code})."}), 500

        # *** FIX: Check the coordinate dictionary existence (safety) ***
        if actual_y_coords is None:
             print("CRITICAL: Calibration localization returned no error code but None coordinates.")
             return jsonify({"error": "Internal error during calibration coordinate processing."}), 500

        # *** FIX: Define required_y_keys and use actual_y_coords for checking and unpacking ***
        required_y_keys = ['y_shoulder', 'y_chest', 'y_waist', 'y_hip', 'y_thigh']
        # This check might be redundant if the function guarantees returning all keys on success, but good practice
        if not all(k in actual_y_coords for k in required_y_keys):
             print(f"CRITICAL ERROR: Calibration coordinates dictionary missing expected keys. Found: {actual_y_coords.keys()}")
             return jsonify({"error": f"Internal error: Coordinate estimation result incomplete."}), 500

        print(f"Calibration: Estimated Y Coords (Hybrid): {actual_y_coords}")
        # Use the actual dictionary for unpacking
        y_shoulder, y_chest, y_waist, y_hip, y_thigh = (actual_y_coords[k] for k in required_y_keys)

        # 3. Validate Y-coordinates against BOTH cal image bounds (using the unpacked values)
        # Note: Use actual_y_coords dict for the check if you prefer checking the dict directly
        valid_y_cal_front = all(0 <= actual_y_coords[k] < height_cal_front for k in required_y_keys)
        valid_y_cal_side = all(0 <= actual_y_coords[k] < height_cal_side for k in required_y_keys) # Use required_y_keys here too for consistency
        if not valid_y_cal_front: return jsonify({"error": f"Cal Y coords out of bounds for FRONT cal ({height_cal_front}). Check pose/lighting."}), 400
        if not valid_y_cal_side: return jsonify({"error": f"Cal Y coords out of bounds for SIDE cal ({height_cal_side}). Check pose/lighting."}), 400

        # 4. Measure relevant pixel dimensions from CALIBRATION masks
        print("Calibration: Measuring pixel dimensions...")
        cal_band_height = 7
        cal_pixel_measurements = {}
        px_errors = []
        # Use the unpacked y_shoulder, y_chest etc.
        cal_pixel_measurements['shoulder_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_shoulder, cal_band_height)
        cal_pixel_measurements['hip_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_hip, cal_band_height)
        cal_pixel_measurements['waist_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_waist, cal_band_height)
        cal_pixel_measurements['thigh_width_px'] = calculate_mask_width_at_y(cal_front_mask, y_thigh, cal_band_height)
        # Chest Y is needed for side mask
        cal_pixel_measurements['chest_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_chest, cal_band_height)
        cal_pixel_measurements['waist_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_waist, cal_band_height)
        cal_pixel_measurements['thigh_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_thigh, cal_band_height)
        # Hip Y is needed for side mask if you calculate hip thickness (currently not used for factors but good practice)
        cal_pixel_measurements['hip_thickness_px'] = calculate_mask_width_at_y(cal_side_mask, y_hip, cal_band_height)


        for key, value in cal_pixel_measurements.items():
             if value is None or value <= 0: px_errors.append(f"Invalid px measurement for {key}: {value}")
        if px_errors: return jsonify({"error": "Calibration measurement error(s): " + "; ".join(px_errors)}), 400
        print(f"Calibration: Measured Px Dimensions: {cal_pixel_measurements}")

        # Estimate Circumferences in PIXELS
        estimated_circ_px = {}
        # Need to handle potential None or zero values from pixel measurements gracefully
        try:
            if cal_pixel_measurements.get('shoulder_width_px') and cal_pixel_measurements.get('chest_thickness_px'):
                estimated_circ_px['chest'] = math.pi * (cal_pixel_measurements['shoulder_width_px'] + cal_pixel_measurements['chest_thickness_px']) / 2
            if cal_pixel_measurements.get('waist_width_px') and cal_pixel_measurements.get('waist_thickness_px'):
                estimated_circ_px['waist'] = math.pi * (cal_pixel_measurements['waist_width_px'] + cal_pixel_measurements['waist_thickness_px']) / 2
            if cal_pixel_measurements.get('thigh_width_px') and cal_pixel_measurements.get('thigh_thickness_px'):
                estimated_circ_px['thigh'] = math.pi * (cal_pixel_measurements['thigh_width_px'] + cal_pixel_measurements['thigh_thickness_px']) / 2
            # Hip circ estimate needs hip width and thickness
            if cal_pixel_measurements.get('hip_width_px') and cal_pixel_measurements.get('hip_thickness_px'):
                 estimated_circ_px['hip'] = math.pi * (cal_pixel_measurements['hip_width_px'] + cal_pixel_measurements['hip_thickness_px']) / 2

        except TypeError as te:
             print(f"Error during pixel circumference estimation: {te}")
             return jsonify({"error": f"Internal error calculating pixel circumferences. Check measurements."}), 500
        print(f"Calibration: Estimated Circ Px: {estimated_circ_px}")


        # 5. Calculate and Store Per-Part Factors
        print("Calibration: Calculating factors...")
        factors_calculated = {}
        calculation_errors = []
        for key in calibration_guide.keys(): # Iterate through reference guide keys
            known_cm = calibration_guide.get(key)
            measured_px = None
            # Map reference key to calculated pixel value key
            if key == 'shoulder_width': measured_px = cal_pixel_measurements.get('shoulder_width_px')
            elif key == 'hip_width': measured_px = cal_pixel_measurements.get('hip_width_px')
            elif key == 'chest_circ': measured_px = estimated_circ_px.get('chest')
            elif key == 'waist_circ': measured_px = estimated_circ_px.get('waist')
            elif key == 'thigh_circ': measured_px = estimated_circ_px.get('thigh')
            # Add hip circ factor calculation if needed (and if hip_circ is in calibration_guide)
            # elif key == 'hip_circ': measured_px = estimated_circ_px.get('hip')
            else:
                print(f"Note: Skipping factor calculation for unmapped reference key '{key}'")
                continue

            if known_cm is None or known_cm <= 0: calculation_errors.append(f"Invalid reference cm value for '{key}'."); continue
            if measured_px is None or measured_px <= 0: calculation_errors.append(f"Invalid or missing pixel measurement for factor '{key}' (Px: {measured_px})."); continue

            try:
                factors_calculated[key] = measured_px / known_cm
            except ZeroDivisionError:
                calculation_errors.append(f"Division by zero calculating factor for '{key}'.")
            except Exception as e_factor:
                calculation_errors.append(f"Unexpected error calculating factor '{key}': {e_factor}")

        # Validate factors were calculated for all guide keys
        missing_factors = [k for k in calibration_guide.keys() if k not in factors_calculated and k in ['shoulder_width', 'hip_width', 'chest_circ', 'waist_circ', 'thigh_circ']] # Check only relevant ones
        if missing_factors:
            calculation_errors.append(f"Factors could not be calculated for: {', '.join(missing_factors)}")

        if len(calculation_errors) > 0:
             error_message = "Calibration factor calculation errors: " + "; ".join(calculation_errors)
             print(f"ERROR: {error_message}")
             calibration_factors = {key: None for key in calibration_factors}; is_calibrated = False
             return jsonify({"error": error_message}), 500 # Return 500 as it's a calculation failure

        # --- Success ---
        # Update the global state atomically (though locks aren't strictly needed for dict update/bool assignment if not multi-threaded writes)
        with counter_lock: # Reuse lock for safety, though maybe overkill here
            calibration_factors.update(factors_calculated)
            is_calibrated = True
        print(f"Calibration successful. Factors: {calibration_factors}")

        # Return results
        return jsonify({
            "status": "calibrated",
            "calculated_factors": calibration_factors,
            "reference_values_cm": calibration_guide,
            "detected_values_px": cal_pixel_measurements,
            "estimated_circ_px": estimated_circ_px,
            "estimated_y_coordinates": actual_y_coords # Return the actual dictionary
        })

    except Exception as e:
        print(f"Error during calibration: {e}"); traceback.print_exc()
        # Reset calibration status on any unexpected error
        with counter_lock: # Ensure status is reset safely
            is_calibrated = False
            # Optionally clear factors too
            # calibration_factors = {key: None for key in calibration_factors}
        return jsonify({"error": f"Calibration encountered an unexpected server error: {str(e)}"}), 500

# --- UPDATED Measurement Route ---
# --- UPDATED Measurement Route ---
@app.route('/measure', methods=['POST'])
def measure():
    """ Measures using front/side images. Includes detection checks."""
    global calibration_factors, is_calibrated, request_counter
    data = request.get_json()
    current_request_no = -1 # Default
    global current_req, request_no
    request_no+=1
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    current_req = f"requests/{request_no}_{timestamp}"
    if not os.path.exists(current_req):
        os.makedirs(current_req); print(f"Created debug directory: {current_req}")

    # --- Request Number Handling ---
    try:
        with counter_lock:
            request_counter += 1
            current_request_no = request_counter
        save_counter(current_request_no)
        print(f"Processing measure request #{current_request_no}")
    except Exception as e_counter:
         print(f"CRITICAL ERROR updating/saving request counter: {e_counter}")
         # Decide if you want to proceed or return error
         return jsonify({"error": "Internal server error processing request counter"}), 500

    # --- Input and Calibration Checks ---
    if not is_calibrated: return jsonify({"error": "[Not Calibrated] Please calibrate the system first!"}), 400 # Adjusted message
    if not all(calibration_factors.get(f) is not None for f in calibration_factors): return jsonify({"error": "Calibration incomplete or factors missing."}), 400
    if not data or 'image_front' not in data or 'image_side' not in data: return jsonify({"error": "Missing front or side image data"}), 400
    if not DEEPLAB_MODEL_LOADED or not SEGFORMER_MODEL_LOADED or not MP_POSE_LOADED: return jsonify({"error": "Required models not loaded on server."}), 500

    try:
        # --- Decode Images ---
        image_front_base64 = data['image_front']
        image_side_base64 = data['image_side']
        image_front_np = base64_to_image(image_front_base64)
        image_side_np = base64_to_image(image_side_base64)

        # *** IMPORTANT: Re-evaluate rotation ***
        # If images from frontend are already correctly oriented (e.g., landscape), REMOVE these lines.
        # If they are portrait and need rotation, KEEP them. Verify this!
        image_front_np = ndimage.rotate(image_front_np, -90)
        image_side_np = ndimage.rotate(image_side_np, -90)
        # ****************************************

        if image_front_np is None or image_side_np is None: return jsonify({"error": "Error decoding base64 images"}), 400

        # --- Save Raw Input Images ---
        if current_request_no != -1:
            try:
                # 1. Get current timestamp
                now = datetime.datetime.now()
                # 2. Format it into a string suitable for filenames (YYYYMMDD_HHMMSS)
                timestamp_str = now.strftime("%Y%m%d")
                # Optional: Add microseconds for even higher uniqueness if needed
                # timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")

                # 3. Create unique filenames using timestamp and request number
                front_filename = os.path.join(
                    current_req,
                    f"front_{timestamp_str}_req{current_request_no}.jpg"
                )
                side_filename = os.path.join(
                    current_req,
                    f"side_{timestamp_str}_req{current_request_no}.jpg"
                )

                # 4. Ensure the directory exists (optional but good practice here)
                os.makedirs(RAW_IMAGE_DIR, exist_ok=True)

                # 5. Save the images
                cv2.imwrite(front_filename, image_front_np)
                cv2.imwrite(side_filename, image_side_np)
                print(f"Saved raw images: {os.path.basename(front_filename)}, {os.path.basename(side_filename)}")
            except Exception as e_save:
                print(f"Warning: Failed to save raw input images for request #{current_request_no}: {e_save}")

        height_front, width_front, _ = image_front_np.shape; height_side, width_side, _ = image_side_np.shape

        # --- Perform Localization & Detection Checks (using FRONT image) ---
        print("Measure: Estimating part locations & performing checks (Hybrid)...")
        # *** CALL MODIFIED FUNCTION ***
        error_code, y_coords = get_body_part_y_coordinates_hybrid(
            image_front_np,
            debug_filename_prefix=f"measure_{current_request_no}_front" # Include req number in debug
        )

        # *** HANDLE RETURNED ERROR CODE ***
        if error_code:
            print(f"Localization/Detection failed with code: {error_code}")
            if error_code == "no_person":
                return jsonify({"error": "No person detected. Please stand clearly in the center of the frame."}), 400
            elif error_code == "multiple_people":
                return jsonify({"error": "Multiple people detected. Please ensure only one person is in the frame."}), 400
            elif error_code == "too_far":
                return jsonify({"error": "You seem too far away. Please move closer to the camera."}), 400
            elif error_code == "too_close":
                return jsonify({"error": "You seem too close to the camera. Please move further back."}), 400
            elif error_code == "no_landmarks":
                 return jsonify({"error": "Could not detect key body points clearly. Please check posture, lighting, and ensure body is visible."}), 400 # User-actionable
            elif error_code in ["segformer_error", "mediapipe_error", "calculation_error", "bbox_error", "model_load_error"]:
                 # These are more likely server-side or setup issues
                 return jsonify({"error": f"Image analysis failed internally ({error_code}). Please try again or contact support."}), 500
            else:
                 # Generic fallback for unexpected codes
                 return jsonify({"error": f"An unexpected error occurred during image analysis ({error_code})."}), 500

        # --- If error_code is None, y_coords should be valid ---
        if y_coords is None: # Should not happen if error_code is None, but safety check
             print("CRITICAL: Localization returned no error code but None coordinates.")
             return jsonify({"error": "Internal error during coordinate processing."}), 500
        required_y_keys = ['y_shoulder', 'y_chest', 'y_waist', 'y_hip', 'y_thigh']
        print(f"Measure: Estimated Y Coords (Hybrid): {y_coords}")
        y_shoulder, y_chest, y_waist, y_hip, y_thigh = (y_coords[k] for k in required_y_keys) # required_y_keys defined earlier

        # --- Proceed with Masking, Measurement, Conversion (using the validated y_coords) ---

        print("Measure: Generating DeepLab masks...")
        # Pass request number for unique debug filenames
        front_mask = generate_human_mask_deeplab(image_front_np, f"measure_{current_request_no}_front")
        side_mask = generate_human_mask_deeplab(image_side_np, f"measure_{current_request_no}_side")
        if front_mask is None or side_mask is None:
             return jsonify({"error": "Failed to generate segmentation mask for measurement."}), 500 # More specific

        # Validate Y-coordinates against BOTH image bounds (still useful)
        valid_y_front = all(0 <= y_coords[k] < height_front for k in required_y_keys)
        valid_y_side = all(0 <= y_coords[k] < height_side for k in required_y_keys) # Check against side height too
        if not valid_y_front: return jsonify({"error": f"Calculated Y coords out of bounds for FRONT image ({height_front}). Check calibration/pose."}), 400
        if not valid_y_side: return jsonify({"error": f"Calculated Y coords out of bounds for SIDE image ({height_side}). Check calibration/pose."}), 400

        # --- Measure Pixel Dimensions ---
        print("Measure: Calculating pixel dimensions...")
        measure_band = 5; measurements_px = {}; px_errors = []
        # ... (Pixel measurement logic remains the same, using the validated y_coords) ...
        measurements_px['shoulder_width_px'] = calculate_mask_width_at_y(front_mask, y_shoulder, measure_band)
        measurements_px['hip_width_px'] = calculate_mask_width_at_y(front_mask, y_hip, measure_band)
        measurements_px['waist_width_px'] = calculate_mask_width_at_y(front_mask, y_waist, measure_band)
        measurements_px['thigh_width_px'] = calculate_mask_width_at_y(front_mask, y_thigh, measure_band)
        measurements_px['chest_thickness_px'] = calculate_mask_width_at_y(side_mask, y_chest, measure_band)
        measurements_px['hip_thickness_px'] = calculate_mask_width_at_y(side_mask, y_hip, measure_band) # You might need this for Hip Circ if hip_width factor isn't ideal
        measurements_px['waist_thickness_px'] = calculate_mask_width_at_y(side_mask, y_waist, measure_band)
        measurements_px['thigh_thickness_px'] = calculate_mask_width_at_y(side_mask, y_thigh, measure_band)


        for key, value in measurements_px.items():
             if value is None or value <= 0: px_errors.append(f"Invalid px for {key}: {value}")
        # Decide if invalid px measurements should be a hard error or just a warning + None in results
        if px_errors:
             print(f"Warning: Invalid pixel measurements detected: {'; '.join(px_errors)}. Results might be incomplete.")
             # Option 1: Return error (stricter)
             # return jsonify({"error": f"Measurement failed for some parts: {'; '.join(px_errors)}"}), 400
             # Option 2: Continue, results will have None for failed parts (current behavior)

        print(f"Measure: Pixel Measurements: {measurements_px}")

        # --- Convert to CM ---
        print("Measure: Converting to CM...")
        cm_measurements = calculate_measurements_cm_multi_factor(measurements_px, calibration_factors)
        print(f"Measure: CM Measurements: {cm_measurements}")

        # --- Calculate Sizing ---
        sizing = get_regional_clothing_sizes_enhanced(cm_measurements)
        print(f"Measure: Recommended Sizing: {sizing}")

        # --- DEBUG Drawing ---
        try: # Wrap debug drawing in try/except
            debug_front_mask_lines = cv2.cvtColor(front_mask, cv2.COLOR_GRAY2BGR)
            debug_side_mask_lines = cv2.cvtColor(side_mask, cv2.COLOR_GRAY2BGR)
            # ... (draw_measurement_line helper and drawing calls remain the same) ...
            def draw_measurement_line(img, y, label, value_px, color, width): # Keep helper
                 if value_px is not None and value_px > 0: # Only draw valid lines
                     # Calculate start/end based on mask at that row to be more precise
                     row = img[y, :, 0] # Use one channel
                     white_pixels = np.where(row > 128)[0] # Find mask pixels
                     if len(white_pixels) > 1:
                        x_start = np.min(white_pixels)
                        x_end = np.max(white_pixels)
                        cv2.line(img, (x_start, y), (x_end, y), color, 1)
                        cv2.putText(img, f"{label}: {value_px:.1f}px", (x_end + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                     else: # Fallback if mask is weird
                        cv2.line(img, (0, y), (width, y), color, 1)
                        cv2.putText(img, f"{label}: {value_px:.1f}px (FullW)", (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                 elif value_px is not None: # Draw line even if 0, but mark it
                     cv2.line(img, (0, y), (width, y), (100, 100, 100), 1)
                     cv2.putText(img, f"{label}: {value_px:.1f}px (Invalid)", (10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

            # Draw lines using current y_coords
            draw_measurement_line(debug_front_mask_lines, y_shoulder, "SW", measurements_px.get('shoulder_width_px'), (0, 255, 0), width_front)
            draw_measurement_line(debug_front_mask_lines, y_waist, "WW", measurements_px.get('waist_width_px'), (0, 255, 255), width_front)
            draw_measurement_line(debug_front_mask_lines, y_hip, "HW", measurements_px.get('hip_width_px'), (0, 0, 255), width_front)
            draw_measurement_line(debug_front_mask_lines, y_thigh, "ThW", measurements_px.get('thigh_width_px'), (255, 0, 255), width_front)
            # Also draw Chest Y on front mask for reference
            draw_measurement_line(debug_front_mask_lines, y_chest, "ChestY", None, (200, 200, 0), width_front)

            draw_measurement_line(debug_side_mask_lines, y_chest, "CT", measurements_px.get('chest_thickness_px'), (0, 255, 0), width_side)
            draw_measurement_line(debug_side_mask_lines, y_waist, "WT", measurements_px.get('waist_thickness_px'), (0, 255, 255), width_side)
            draw_measurement_line(debug_side_mask_lines, y_hip, "HT", measurements_px.get('hip_thickness_px'), (0, 0, 255), width_side)
            draw_measurement_line(debug_side_mask_lines, y_thigh, "ThT", measurements_px.get('thigh_thickness_px'), (255, 0, 255), width_side)
             # Also draw Shoulder Y on side mask for reference
            draw_measurement_line(debug_side_mask_lines, y_shoulder, "ShoulderY", None, (200, 200, 0), width_side)


            # Save debug images with request number
            cv2.imwrite(os.path.join(current_req, f"debug_measure_{current_request_no}_front_mask_levels.jpg"), debug_front_mask_lines)
            cv2.imwrite(os.path.join(current_req, f"debug_measure_{current_request_no}_side_mask_levels.jpg"), debug_side_mask_lines)
        except Exception as e_debug_draw:
            print(f"Warning: Failed to draw/save debug images for req #{current_request_no}: {e_debug_draw}")


        # Remove Hip Width before sending? Your code does this. Keep if intended.
        if 'Hip Width' in cm_measurements:
            cm_measurements.pop('Hip Width')
            print("Note: Removed 'Hip Width' from final response.")


        # --- Return success results ---
        return jsonify({
            "status": "success",
            "measurements_cm": cm_measurements,
            "recommended_sizing": sizing,
            # "calibration_factors_used": calibration_factors, # Optional: Exclude if not needed by frontend
            # "pixel_measurements_raw": measurements_px,      # Optional: Exclude
            # "estimated_y_coordinates": y_coords,            # Optional: Exclude
            "request_number": current_request_no
        })

    except Exception as e:
        print(f"Error in /measure endpoint (Req #{current_request_no}): {e}"); traceback.print_exc()
        # Provide the request number in the error if possible
        error_msg = f"Unexpected measurement error occurred (Req #{current_request_no}). Please try again." if current_request_no != -1 else "Unexpected measurement error occurred. Please try again."
        # You could add more detail based on the exception `e` if needed, but avoid leaking sensitive info
        # error_msg += f" Details: {str(e)[:100]}" # Example: Truncated details
        return jsonify({"error": error_msg}), 500

# --- Other routes and main execution block remain the same ---
# ...
@app.route('/')
def index():
    try:
        return render_template('index.html', is_calibrated=is_calibrated, factors=calibration_factors)
    except Exception as e_template:
        print(f"Error rendering template: {e_template}")
        status_text = f"Calibrated: {is_calibrated} (Factors: {calibration_factors})" if is_calibrated else "Not Calibrated"
        return f"<html><body><h1>Measurement App V7 (Hybrid Y)</h1><p>Status: {status_text}</p><p>(Error rendering template)</p></body></html>"


# --- Main Execution ---
# Remains the same
if __name__ == '__main__':
    # *** UPDATED *** Check all 3 models
    models_ok = DEEPLAB_MODEL_LOADED and SEGFORMER_MODEL_LOADED and MP_POSE_LOADED
    if not models_ok:
        print("\nFATAL WARNING: One or more required models failed to load.")
        print(f"  DeepLab: {DEEPLAB_MODEL_LOADED}, SegFormer: {SEGFORMER_MODEL_LOADED}, MediaPipe: {MP_POSE_LOADED}")
        print("---> Application will likely FAIL. <---")
    app.run(host='0.0.0.0', port=7000, debug=False)