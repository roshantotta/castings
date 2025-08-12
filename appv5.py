import os
import sys
import ctypes
import time
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, redirect, url_for, flash, session
from ultralytics import YOLO

# Append the path for the camera control module
sys.path.append(r"c:/Program Files (x86)/MVS/Development/Samples/Python/MvImport")
import MvCameraControl_class as MvCC

app = Flask(__name__)
app.secret_key = 'dtc123'

# ----------------------------- Global Variable -----------------------------
# Store captured frames (a dictionary with keys "cam1", "cam2", "cam3")
captured_frames = {}


# ----------------------------- Load YOLO Models -----------------------------
try:
    det_model = YOLO(r"C:\Users\ADMIN\Desktop\New_folder\models\hleicoil2_4_6.pt")
    clf_model = YOLO(r"C:\Users\ADMIN\Desktop\New_folder\models\hcp2cls.pt")
except Exception as e:
    print(f"Failed to load models: {e}")
    sys.exit(1)


# ----------------------------- Device Enumeration -----------------------------
def enumerate_devices():
    pstDevList = MvCC.MV_CC_DEVICE_INFO_LIST()
    nTLayerType = MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE
    ret = MvCC.MvCamera.MV_CC_EnumDevices(nTLayerType, pstDevList)
    if ret != 0 or pstDevList.nDeviceNum < 1:
        return {}

    devices = {}
    for i in range(pstDevList.nDeviceNum):
        device_info_ptr = ctypes.cast(pstDevList.pDeviceInfo[i], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO))
        device_info = device_info_ptr.contents

        # Get serial number based on device type
        serial = ""
        if device_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
            serial = bytes(device_info.SpecialInfo.stGigEInfo.chSerialNumber).decode('utf-8', errors='ignore').strip('\x00')
        elif device_info.nTLayerType == MvCC.MV_USB_DEVICE:
            serial = bytes(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode('utf-8', errors='ignore').strip('\x00')
        
        if serial:
            devices[serial] = device_info

    return devices


# ----------------------------- Camera Utilities -----------------------------
def initialize_camera_from_device(device_info):
    cam = MvCC.MvCamera()
    if cam.MV_CC_CreateHandle(device_info) != 0:
        raise Exception("CreateHandle failed")
    if cam.MV_CC_OpenDevice(MvCC.MV_ACCESS_Exclusive, 0) != 0:
        raise Exception("OpenDevice failed")
    cam.MV_CC_SetEnumValue("ExposureAuto", MvCC.MV_EXPOSURE_AUTO_MODE_CONTINUOUS)
    if cam.MV_CC_StartGrabbing() != 0:
        raise Exception("StartGrabbing failed")
    return cam

def release_camera(cam):
    try:
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    except Exception as e:
        print(f"Error releasing camera: {e}")

def get_frame(cam):
    stOutFrame = MvCC.MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
    ret = cam.MV_CC_GetImageBuffer(stFrame=stOutFrame, nMsec=3000)
    if ret == 0 and stOutFrame.pBufAddr:
        buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
        ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
        image_data = np.frombuffer(buf_cache, dtype=np.uint8)
        image = image_data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
        pixel_format = stOutFrame.stFrameInfo.enPixelType

        # Handle some common pixel formats. Adjust as needed.
        if pixel_format == MvCC.PixelType_Gvsp_BayerRG8:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
        elif pixel_format == MvCC.PixelType_Gvsp_BayerGB8:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_GB2RGB)
        elif pixel_format == MvCC.PixelType_Gvsp_BayerGR8:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_GR2RGB)
        elif pixel_format == MvCC.PixelType_Gvsp_BayerBG8:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cam.MV_CC_FreeImageBuffer(stOutFrame)
        return image_rgb
    return None

def process_frame(image):
    crops_results = []
    results = det_model(image)
    annotated_image = image.copy()

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            clf_results = clf_model(cropped)
            # Assuming clf_results[0].probs.top1 provides the index (adjust if your API differs)
            class_index = int(clf_results[0].probs.top1)
            class_label = clf_results[0].names[class_index]
            color = (0, 255, 0) if class_label.lower() == "present" else (0, 0, 255)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_image, class_label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            crops_results.append((cropped, class_label))
    return annotated_image, crops_results

def img_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded}"

def save_processed_images(processed_results, shift, batch, object_number):
    # Create directory structure: results/<shift>/<batch>
    base_dir = os.path.join("results", f"shift_{shift}")
    batch_dir = os.path.join(base_dir, f"batch_{batch}_object_{object_number}")
    os.makedirs(batch_dir, exist_ok=True)

    # Mapping of cam labels to human-readable prefix
    prefix_map = {
        "cam1": "left",
        "cam2": "top",
        "cam3": "right"
    }

    for cam_key, result in processed_results.items():
        if "error" in result:
            continue

        # Get original image from captured frames
        original = captured_frames.get(cam_key)
        annotated_data = result["annotated"]
        _, encoded = annotated_data.split(',', 1)
        annotated_bytes = base64.b64decode(encoded)
        annotated = np.frombuffer(annotated_bytes, dtype=np.uint8)
        annotated = cv2.imdecode(annotated, cv2.IMREAD_COLOR)

        prefix = prefix_map.get(cam_key, cam_key)
        # orig_filename = os.path.join(batch_dir, f"{prefix}_original.jpg")
        ann_filename = os.path.join(batch_dir, f"{prefix}_annotated.jpg")

        # cv2.imwrite(orig_filename, original)
        cv2.imwrite(ann_filename, annotated)


# ----------------------------- Login & Authentication -----------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Example: hardcoded credentials (username: admin, password: admin)
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin':
            session['logged_in'] = True
            return redirect(url_for('shift_batch'))
        else:
            flash("Invalid credentials. Try again.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ----------------------------- Shift and Batch Input -----------------------------
@app.route('/shift_batch', methods=['GET', 'POST'])
def shift_batch():
    # Ensure the user is logged in
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    if request.method == 'POST':
        shift = request.form.get('shift')
        batch = request.form.get('batch')
        if not shift or not batch:
            flash("Please enter both shift and batch.", "warning")
            return redirect(url_for('shift_batch'))
        session['shift'] = shift
        session['batch'] = batch
        return redirect(url_for('capture_form'))
    return render_template('shift.html')

# ----------------------------- Object Number Input & Capture Page -----------------------------
@app.route('/capture_form', methods=['GET', 'POST'])
def capture_form():
    # Ensure shift and batch are provided
    if not session.get('shift') or not session.get('batch'):
        flash("Please input shift and batch first.", "warning")
        return redirect(url_for('shift_batch'))
    if request.method == 'POST':
        object_id = request.form.get('object_id')
        if not object_id:
            flash("Please enter the object number.", "warning")
            return redirect(url_for('capture_form'))
        session['object_number'] = object_id
        # Redirect to capture images
        return redirect(url_for('capture'))
    return render_template('capture_form.html')

# ----------------------------- Routes for Capturing and Processing -----------------------------

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if not session.get('shift') or not session.get('batch') or not session.get('object_number'):
        flash("Missing shift, batch or object number. Please provide those details.", "warning")
        return redirect(url_for('shift_batch'))

    global captured_frames
    device_map = enumerate_devices()

    expected_serials = {
        # "cam1": "DA5211138",  # left
        "cam2": "DA5843333",  # top
        "cam3": "DA5843329"   # right
    }

    frames = {}
    cams = {}

    for cam_label, serial in expected_serials.items():
        device_info = device_map.get(serial)
        if device_info is None:
            flash(f"Camera with serial {serial} ({cam_label}) not found.", "danger")
            frames[cam_label] = None
            continue

        try:
            cam = initialize_camera_from_device(device_info)
            frame = get_frame(cam)
            release_camera(cam)
            cams[cam_label] = cam
            frames[cam_label] = frame
        except Exception as e:
            flash(f"Error with {cam_label} ({serial}): {e}", "danger")
            frames[cam_label] = None

    captured_frames = frames

    # Convert frames to base64 for display
    display_frames = {k: img_to_base64(v) if v is not None else None for k, v in frames.items()}

    return render_template('results.html', mode="capture", results=display_frames)

@app.route('/process', methods=['POST'])
def process_images():
    # Ensure captured_frames is available and valid.
    global captured_frames
    if not captured_frames or any(captured_frames.get(f"cam{i+1}") is None for i in range(3)):
        flash("Please capture frames first.", "warning")
        return redirect(url_for('capture_form'))
    
    processed_results = {}
    for i in range(3):
        key = f"cam{i+1}"
        frame = captured_frames.get(key)
        if frame is None:
            processed_results[key] = {"error": "Frame capture failed."}
        else:
            annotated, _ = process_frame(frame)
            processed_results[key] = {
                "original": img_to_base64(frame),
                "annotated": img_to_base64(annotated)
            }
    
    # Save processed data to folder structure based on shift and batch
    shift = session.get('shift')
    batch = session.get('batch')
    object_number = session.get('object_number')
    save_processed_images(processed_results, shift, batch, object_number)

    return render_template('results.html', mode="process", results=processed_results)

# ----------------------------- Main -----------------------------
if __name__ == '__main__':
    app.run(debug=True)