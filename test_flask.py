# import os
# import sys
# import ctypes
# import time
# import cv2
# import numpy as np
# import base64
# import logging
# import uuid
# import threading
# from datetime import datetime
# from logging.handlers import RotatingFileHandler
# from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
# from werkzeug.security import check_password_hash, generate_password_hash
# from functools import wraps
# import yaml
# from dotenv import load_dotenv
# from waitress import serve
# import sentry_sdk
# from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary, Text, ForeignKey, Boolean
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, relationship
# from sqlalchemy.sql import func

# sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=1.0)

# sys.path.append(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport")
# import MvCameraControl_class as MvCC

# # Load environment variables and configuration
# load_dotenv()

# # Setup logging
# log_directory = "logs"
# os.makedirs(log_directory, exist_ok=True)
# log_file = os.path.join(log_directory, "app.log")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] - %(message)s",
#     handlers=[
#         RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load configuration
# try:
#     with open("config.yaml", "r") as config_file:
#         config = yaml.safe_load(config_file)
# except Exception as e:
#     logger.critical(f"Failed to load configuration: {e}")
#     sys.exit(1)

# # Initialize Flask application
# app = Flask(__name__)
# app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
# app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session timeout in seconds

# # Global variables
# global captured_frames
# captured_frames = {}

# # Streaming control
# streaming_active = False
# camera_instances = {}
# stream_lock = threading.Lock()

# # ----------------------------- Database Models -----------------------------
# Base = declarative_base()

# class Batch(Base):
#     __tablename__ = 'batches'
    
#     id = Column(Integer, primary_key=True)
#     shift = Column(String(50), nullable=False)
#     batch_number = Column(Integer, nullable=False)
#     timestamp = Column(DateTime, default=func.now())
#     created_by = Column(String(100), nullable=False)
    
#     objects = relationship("Object", back_populates="batch", cascade="all, delete-orphan")
    
#     def __repr__(self):
#         return f"<Batch(shift='{self.shift}', batch_number={self.batch_number})>"

# class Object(Base):
#     __tablename__ = 'objects'
    
#     id = Column(Integer, primary_key=True)
#     object_number = Column(Integer, nullable=False)
#     timestamp = Column(DateTime, default=func.now())
#     batch_id = Column(Integer, ForeignKey('batches.id'), nullable=False)
    
#     batch = relationship("Batch", back_populates="objects")
#     images = relationship("Image", back_populates="object", cascade="all, delete-orphan")
    
#     def __repr__(self):
#         return f"<Object(object_number={self.object_number}, batch_id={self.batch_id})>"

# class Image(Base):
#     __tablename__ = 'images'
    
#     id = Column(Integer, primary_key=True)
#     camera_label = Column(String(50), nullable=False)
#     image_type = Column(String(20), nullable=False)
#     data = Column(LargeBinary, nullable=False)
#     format = Column(String(10), default='jpg')
#     timestamp = Column(DateTime, default=func.now())
#     object_id = Column(Integer, ForeignKey('objects.id'), nullable=False)
    
#     object = relationship("Object", back_populates="images")
#     detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")
    
#     def __repr__(self):
#         return f"<Image(camera={self.camera_label}, type={self.image_type}, object_id={self.object_id})>"

# class Detection(Base):
#     __tablename__ = 'detections'
    
#     id = Column(Integer, primary_key=True)
#     x1 = Column(Integer, nullable=False)
#     y1 = Column(Integer, nullable=False)
#     x2 = Column(Integer, nullable=False)
#     y2 = Column(Integer, nullable=False)
#     class_label = Column(String(50), nullable=False)
#     confidence = Column(String(50), nullable=True)
#     crop_data = Column(LargeBinary, nullable=True)
#     timestamp = Column(DateTime, default=func.now())
#     image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    
#     image = relationship("Image", back_populates="detections")
    
#     def __repr__(self):
#         return f"<Detection(class={self.class_label}, image_id={self.image_id})>"

# # Database connection setup
# class DatabaseManager:
#     def __init__(self, connection_string):
#         self.engine = create_engine(connection_string)
#         self.Session = sessionmaker(bind=self.engine)
        
#     def create_tables(self):
#         Base.metadata.create_all(self.engine)
        
#     def get_session(self):
#         return self.Session()

# # Initialize database connection
# def init_database():
#     db_config = config.get("database", {})
#     db_type = db_config.get("type", "sqlite")
    
#     if db_type == "sqlite":
#         db_path = db_config.get("path", "camera_system.db")
#         connection_string = f"sqlite:///{db_path}"
#     elif db_type == "mysql":
#         connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
#     elif db_type == "postgresql":
#         connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
#     else:
#         logger.critical(f"Unsupported database type: {db_type}")
#         sys.exit(1)
    
#     try:
#         db_manager = DatabaseManager(connection_string)
#         db_manager.create_tables()
#         logger.info(f"Database initialized successfully ({db_type})")
#         return db_manager
#     except Exception as e:
#         logger.critical(f"Failed to initialize database: {e}")
#         sys.exit(1)

# # ----------------------------- YOLO Models -----------------------------
# try:
#     det_model_path = config.get("detection_model_path")
#     clf_model_path = config.get("classification_model_path")
    
#     from ultralytics import YOLO
#     det_model = YOLO(det_model_path)
#     clf_model = YOLO(clf_model_path)
#     logger.info("YOLO models loaded successfully")
# except Exception as e:
#     logger.critical(f"Failed to load models: {e}")
#     sys.exit(1)

# # ----------------------------- Authentication -----------------------------
# def login_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#         if not session.get('logged_in'):
#             flash("Please log in first.", "warning")
#             return redirect(url_for('login'))
#         return f(*args, **kwargs)
#     return decorated_function

# # ----------------------------- Camera & Image Processing -----------------------------
# class CameraManager:
#     @staticmethod
#     def enumerate_devices():
#         pstDevList = MvCC.MV_CC_DEVICE_INFO_LIST()
#         nTLayerType = MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE
#         ret = MvCC.MvCamera.MV_CC_EnumDevices(nTLayerType, pstDevList)
#         if ret != 0:
#             logger.error(f"Failed to enumerate devices, error code: {ret}")
#             return {}
        
#         devices = {}
#         for i in range(pstDevList.nDeviceNum):
#             device_info_ptr = ctypes.cast(pstDevList.pDeviceInfo[i], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO))
#             device_info = device_info_ptr.contents
#             serial = ""
#             if device_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
#                 serial = bytes(device_info.SpecialInfo.stGigEInfo.chSerialNumber).decode('utf-8', errors='ignore').strip('\x00')
#             elif device_info.nTLayerType == MvCC.MV_USB_DEVICE:
#                 serial = bytes(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode('utf-8', errors='ignore').strip('\x00')
            
#             if serial:
#                 devices[serial] = device_info
#         return devices

#     @staticmethod
#     def initialize_camera(device_info):
#         cam = MvCC.MvCamera()
#         if cam.MV_CC_CreateHandle(device_info) != 0:
#             raise Exception("CreateHandle failed")
#         if cam.MV_CC_OpenDevice(MvCC.MV_ACCESS_Exclusive, 0) != 0:
#             cam.MV_CC_DestroyHandle()
#             raise Exception("OpenDevice failed")
#         cam.MV_CC_SetEnumValue("ExposureAuto", MvCC.MV_EXPOSURE_AUTO_MODE_OFF)
#         if cam.MV_CC_StartGrabbing() != 0:
#             cam.MV_CC_CloseDevice()
#             cam.MV_CC_DestroyHandle()
#             raise Exception("StartGrabbing failed")
#         return cam

#     @staticmethod
#     def release_camera(cam):
#         try:
#             cam.MV_CC_StopGrabbing()
#             cam.MV_CC_CloseDevice()
#             cam.MV_CC_DestroyHandle()
#         except: pass

#     @staticmethod
#     def get_frame(cam, timeout=3000):
#         stOutFrame = MvCC.MV_FRAME_OUT()
#         ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
#         ret = cam.MV_CC_GetImageBuffer(stFrame=stOutFrame, nMsec=timeout)
#         if ret != 0: return None
        
#         try:
#             buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
#             ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
#             image_data = np.frombuffer(buf_cache, dtype=np.uint8)
#             image = image_data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            
#             if stOutFrame.stFrameInfo.enPixelType in [MvCC.PixelType_Gvsp_BayerRG8, MvCC.PixelType_Gvsp_BayerGB8,
#                                                      MvCC.PixelType_Gvsp_BayerGR8, MvCC.PixelType_Gvsp_BayerBG8]:
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
#             else:
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#             return image_rgb
#         finally:
#             cam.MV_CC_FreeImageBuffer(stOutFrame)

#     @staticmethod
#     def get_frame_continuous(cam, timeout=3000):
#         """Optimized version for continuous streaming"""
#         stOutFrame = MvCC.MV_FRAME_OUT()
#         ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
#         ret = cam.MV_CC_GetImageBuffer(stFrame=stOutFrame, nMsec=timeout)
#         if ret != 0: return None
        
#         try:
#             # More efficient memory handling for streaming
#             image_data = np.frombuffer(
#                 ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)).contents,
#                 dtype=np.uint8
#             )
#             image = image_data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            
#             if stOutFrame.stFrameInfo.enPixelType in [MvCC.PixelType_Gvsp_BayerRG8, MvCC.PixelType_Gvsp_BayerGB8,
#                                                      MvCC.PixelType_Gvsp_BayerGR8, MvCC.PixelType_Gvsp_BayerBG8]:
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
#             else:
#                 image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#             return image_rgb
#         finally:
#             cam.MV_CC_FreeImageBuffer(stOutFrame)

# class ImageProcessor:
#     @staticmethod
#     def process_frame(image):
#         if image is None: return None, []
#         crops_results = []
        
#         try:
#             results = det_model(image)
#             annotated_image = image.copy()
            
#             for result in results:
#                 for box in result.boxes.xyxy:
#                     x1, y1, x2, y2 = map(int, box[:4])
#                     if x1 >= x2 or y1 >= y2: continue
                    
#                     cropped = image[y1:y2, x1:x2]
#                     if cropped.size == 0: continue
                    
#                     clf_results = clf_model(cropped)
#                     class_index = int(clf_results[0].probs.top1)
#                     class_label = clf_results[0].names[class_index]
#                     confidence = f"{clf_results[0].probs.top1conf:.2f}"
                    
#                     color = (0, 255, 0) if class_label.lower() == "present" else (0, 0, 255)
#                     cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
#                     cv2.putText(annotated_image, f"{class_label} ({confidence})", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
#                     _, encoded_crop = cv2.imencode('.jpg', cropped)
#                     crops_results.append((x1, y1, x2, y2, class_label, confidence, encoded_crop.tobytes()))
            
#             return annotated_image, crops_results
#         except Exception as e:
#             logger.error(f"Image processing error: {e}")
#             return image, []

#     @staticmethod
#     def img_to_base64(img):
#         """Convert OpenCV image to base64 string"""
#         try:
#             _, buffer = cv2.imencode('.jpg', img)
#             return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
#         except Exception as e:
#             logger.error(f"Error converting image to base64: {e}")
#             return None

#     @staticmethod
#     def base64_to_image(base64_str):
#         """Convert base64 image string to OpenCV image"""
#         try:
#             if base64_str.startswith('data:image'):
#                 # Remove the data URL prefix if present
#                 base64_str = base64_str.split(',', 1)[1]
#             img_data = base64.b64decode(base64_str)
#             img_array = np.frombuffer(img_data, np.uint8)
#             return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#         except Exception as e:
#             logger.error(f"Error converting base64 to image: {e}")
#             return None

#     @staticmethod
#     def img_to_bytes(img):
#         """Convert OpenCV image to bytes (JPEG format)"""
#         try:
#             _, buffer = cv2.imencode('.jpg', img)
#             return buffer.tobytes()
#         except Exception as e:
#             logger.error(f"Error converting image to bytes: {e}")
#             return None

#     @staticmethod
#     def save_to_db(db_session, processed_results, shift, batch_number, object_number, username):
#         try:
#             batch = db_session.query(Batch).filter_by(shift=shift, batch_number=batch_number).first()
#             if not batch:
#                 batch = Batch(shift=shift, batch_number=batch_number, created_by=username)
#                 db_session.add(batch)
#                 db_session.flush()
            
#             object_record = Object(object_number=object_number, batch_id=batch.id)
#             db_session.add(object_record)
#             db_session.flush()
            
#             for cam_key, result in processed_results.items():
#                 if "error" in result: continue
                
#                 camera_label = {"cam1": "left", "cam2": "top", "cam3": "right"}.get(cam_key, cam_key)
                
#                 if "original" in result and result["original"]:
#                     original_img = ImageProcessor.base64_to_image(result["original"])
#                     if original_img is not None:
#                         db_session.add(Image(
#                             camera_label=camera_label,
#                             image_type="original",
#                             data=ImageProcessor.img_to_bytes(original_img),
#                             object_id=object_record.id
#                         ))
                
#                 if "annotated" in result and result["annotated"]:
#                     annotated_img = ImageProcessor.base64_to_image(result["annotated"])
#                     if annotated_img is not None:
#                         annotated_image = Image(
#                             camera_label=camera_label,
#                             image_type="annotated",
#                             data=ImageProcessor.img_to_bytes(annotated_img),
#                             object_id=object_record.id
#                         )
#                         db_session.add(annotated_image)
#                         db_session.flush()
                        
#                         if "detection_details" in result:
#                             for det in result["detection_details"]:
#                                 db_session.add(Detection(
#                                     x1=det[0], y1=det[1], x2=det[2], y2=det[3],
#                                     class_label=det[4], confidence=det[5],
#                                     crop_data=det[6], image_id=annotated_image.id
#                                 ))
            
#             db_session.commit()
#             return True
#         except Exception as e:
#             db_session.rollback()
#             logger.error(f"Database save error: {e}")
#             return False

# # ----------------------------- Streaming Functions -----------------------------
# def generate_frames(cam_label):
#     """Generator function that yields frames from the camera"""
#     global streaming_active, camera_instances, stream_lock
    
#     with stream_lock:
#         cam = camera_instances.get(cam_label)
#         if not cam:
#             device_map = CameraManager.enumerate_devices()
#             camera_config = config.get("cameras", {})
#             serial = camera_config.get(cam_label)
#             if not serial:
#                 return
            
#             try:
#                 cam = CameraManager.initialize_camera(device_map[serial])
#                 camera_instances[cam_label] = cam
#             except Exception as e:
#                 logger.error(f"Camera initialization error {cam_label}: {e}")
#                 return
    
#     while streaming_active:
#         with stream_lock:
#             if not streaming_active:
#                 break
                
#             frame = CameraManager.get_frame_continuous(cam)
#             if frame is None:
#                 continue
                
#             # Convert frame to JPEG
#             ret, buffer = cv2.imencode('.jpg', frame)
#             if not ret:
#                 continue
                
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# # ----------------------------- Routes -----------------------------
# @app.route('/', methods=['GET', 'POST'])
# def login():
#     if session.get('logged_in'):
#         return redirect(url_for('shift_batch'))
    
#     if request.method == 'POST':
#         username = request.form.get('username')
#         password = request.form.get('password')
#         users = config.get("users", {})
        
#         if check_password_hash(users.get(username, ''), password):
#             session['logged_in'] = True
#             session['username'] = username
#             return redirect(url_for('shift_batch'))
#         else:
#             flash("Invalid credentials", "danger")
    
#     return render_template('login.html')

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login'))

# @app.route('/shift_batch', methods=['GET', 'POST'])
# @login_required
# def shift_batch():
#     if request.method == 'POST':
#         session['shift'] = request.form.get('shift')
#         session['batch'] = request.form.get('batch')
#         return redirect(url_for('capture_form'))
#     return render_template('shift.html')

# @app.route('/capture_form', methods=['GET', 'POST'])
# @login_required
# def capture_form():
#     if request.method == 'POST':
#         session['object_number'] = request.form.get('object_id')
#         return redirect(url_for('stream'))  # Redirect to streaming page instead of direct capture
#     return render_template('capture_form.html')

# @app.route('/stream')
# @login_required
# def stream():
#     """Show the streaming page"""
#     if not all(k in session for k in ['shift', 'batch', 'object_number']):
#         flash("Please complete all previous steps", "warning")
#         return redirect(url_for('capture_form'))
    
#     camera_config = config.get("cameras", {})
#     return render_template('stream.html', cameras=camera_config.keys())

# @app.route('/capture', methods=['GET'])
# @login_required
# def capture():
#     """Capture frames from all cameras"""
#     global captured_frames, camera_instances, stream_lock
    
#     with stream_lock:
#         if not camera_instances:
#             flash("Please start the stream first", "warning")
#             return redirect(url_for('stream'))
        
#         captured_frames = {}
#         for cam_label, cam in camera_instances.items():
#             frame = CameraManager.get_frame(cam)
#             if frame is not None:
#                 captured_frames[cam_label] = frame
    
#     if not captured_frames:
#         flash("Failed to capture frames", "danger")
#         return redirect(url_for('stream'))
    
#     display_frames = {k: ImageProcessor.img_to_base64(v) for k, v in captured_frames.items()}
#     return render_template('results.html', mode="capture", results=display_frames)

# @app.route('/video_feed/<cam_label>')
# @login_required
# def video_feed(cam_label):
#     """Video streaming route that returns MJPEG frames"""
#     return Response(generate_frames(cam_label),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/start_stream')
# @login_required
# def start_stream():
#     """Start the video stream"""
#     global streaming_active
#     streaming_active = True
#     return '', 204

# @app.route('/stop_stream')
# @login_required
# def stop_stream():
#     """Stop the video stream and release cameras"""
#     global streaming_active, camera_instances, stream_lock
    
#     with stream_lock:
#         streaming_active = False
#         time.sleep(0.5)  # Give time for streams to stop
        
#         for cam_label, cam in camera_instances.items():
#             try:
#                 CameraManager.release_camera(cam)
#             except Exception as e:
#                 logger.error(f"Error releasing camera {cam_label}: {e}")
        
#         camera_instances = {}
#     return '', 204

# @app.route('/capture', methods=['GET'])
# @login_required
# def capture():
#     """Capture frames from all cameras"""
#     global captured_frames, camera_instances, stream_lock
    
#     if not camera_instances:
#         flash("Please start the stream first", "warning")
#         return redirect(url_for('stream'))
    
#     with stream_lock:
#         captured_frames = {}
#         for cam_label, cam in camera_instances.items():
#             frame = CameraManager.get_frame(cam)
#             if frame is not None:
#                 captured_frames[cam_label] = frame
    
#     if not captured_frames:
#         flash("Failed to capture frames", "danger")
#         return redirect(url_for('stream'))
    
#     display_frames = {k: ImageProcessor.img_to_base64(v) for k, v in captured_frames.items()}
#     return render_template('results.html', mode="capture", results=display_frames)

# @app.route('/process', methods=['POST'])
# @login_required
# def process_images():
#     global captured_frames
#     if not captured_frames:
#         flash("No frames captured", "warning")
#         return redirect(url_for('capture_form'))
    
#     processed_results = {}
#     for cam_key, frame in captured_frames.items():
#         if frame is None:
#             processed_results[cam_key] = {"error": "Capture failed"}
#             continue
        
#         annotated, crops = ImageProcessor.process_frame(frame)
#         processed_results[cam_key] = {
#             "original": ImageProcessor.img_to_base64(frame),
#             "annotated": ImageProcessor.img_to_base64(annotated),
#             "detection_details": crops
#         }
    
#     db_session = db_manager.get_session()
#     try:
#         success = ImageProcessor.save_to_db(
#             db_session,
#             processed_results,
#             session['shift'],
#             session['batch'],
#             session['object_number'],
#             session['username']
#         )
#         flash("Processed successfully" if success else "Processing failed", "success" if success else "danger")
#     finally:
#         db_session.close()
    
#     return render_template('results.html', mode="process", results=processed_results)

# @app.route('/history')
# @login_required
# def view_history():
#     db_session = db_manager.get_session()
#     try:
#         batches = db_session.query(Batch).order_by(Batch.timestamp.desc()).limit(20).all()
#         batch_data = [{
#             'id': b.id,
#             'shift': b.shift,
#             'batch_number': b.batch_number,
#             'timestamp': b.timestamp,
#             'created_by': b.created_by,
#             'objects_count': len(b.objects)
#         } for b in batches]
#         return render_template('history.html', batches=batch_data)
#     finally:
#         db_session.close()

# @app.route('/view_batch/<int:batch_id>')
# @login_required
# def view_batch(batch_id):
#     db_session = db_manager.get_session()
#     try:
#         batch = db_session.query(Batch).get(batch_id)
#         if not batch: return redirect(url_for('view_history'))
        
#         object_data = [{
#             'id': obj.id,
#             'object_number': obj.object_number,
#             'timestamp': obj.timestamp,
#             'images_count': len(obj.images)
#         } for obj in batch.objects]
        
#         return render_template('batch_view.html', batch=batch, objects=object_data)
#     finally:
#         db_session.close()

# @app.route('/view_object/<int:object_id>')
# @login_required
# def view_object(object_id):
#     db_session = db_manager.get_session()
#     try:
#         obj = db_session.query(Object).get(object_id)
#         if not obj: return redirect(url_for('view_history'))
        
#         images = {}
#         for img in obj.images:
#             img_base64 = base64.b64encode(img.data).decode('utf-8')
#             if img.camera_label not in images:
#                 images[img.camera_label] = {}
            
#             images[img.camera_label][img.image_type] = f"data:image/jpeg;base64,{img_base64}"
            
#             if img.image_type == "annotated":
#                 detections = []
#                 for det in img.detections:
#                     crop_base64 = base64.b64encode(det.crop_data).decode('utf-8') if det.crop_data else None
#                     detections.append({
#                         'coordinates': [det.x1, det.y1, det.x2, det.y2],
#                         'class': det.class_label,
#                         'confidence': det.confidence,
#                         'crop': f"data:image/jpeg;base64,{crop_base64}" if crop_base64 else None
#                     })
#                 images[img.camera_label]['detections'] = detections
        
#         return render_template('object_view.html', 
#                              object=obj,
#                              batch=obj.batch,
#                              images=images)
#     finally:
#         db_session.close()

# # ----------------------------- App Initialization -----------------------------
# def init_app():
#     global db_manager
#     db_manager = init_database()
    
#     if not config.get("users"):
#         config["users"] = {"admin": generate_password_hash(os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin"))}
#         with open("config.yaml", "w") as f:
#             yaml.dump(config, f)
    
#     return app

# if __name__ == '__main__':
#     app = init_app()
#     serve(app, host=config.get('host', '0.0.0.0'), port=config.get('port', 5000), threads=4)



import os
import sys
import ctypes
import time
import cv2
import numpy as np
import base64
import logging
import uuid
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import yaml
from dotenv import load_dotenv
from waitress import serve
import sentry_sdk
from sqlalchemy import create_engine, Column, Integer, String, DateTime, LargeBinary, Text, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.sql import func

# Initialize Sentry
sentry_sdk.init(dsn=os.environ.get("SENTRY_DSN"), traces_sample_rate=1.0)

# Fix path issues for MVS SDK
sys.path.append(r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport")
try:
    import MvCameraControl_class as MvCC
    # Fix path escape sequences
    MvCC.MvCamCtrldll = ctypes.WinDLL(r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64\MvCameraControl.dll")
except Exception as e:
    logging.error(f"Failed to load MVS SDK: {e}")
    sys.exit(1)

# Load environment variables and configuration
load_dotenv()

# Setup logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_file = os.path.join(log_directory, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    logger.critical(f"Failed to load configuration: {e}")
    sys.exit(1)

# Initialize Flask application
app = Flask(__name__, template_folder= "templates")
app.secret_key = os.environ.get("SECRET_KEY", uuid.uuid4().hex)
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # Session timeout in seconds

# Global variables
captured_frames = {}
streaming_active = False
camera_instances = {}
stream_lock = threading.Lock()

# Database Models
Base = declarative_base()

class Batch(Base):
    __tablename__ = 'batches'
    id = Column(Integer, primary_key=True)
    shift = Column(String(50), nullable=False)
    batch_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    created_by = Column(String(100), nullable=False)
    objects = relationship("Object", back_populates="batch", cascade="all, delete-orphan")

class Object(Base):
    __tablename__ = 'objects'
    id = Column(Integer, primary_key=True)
    object_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=func.now())
    batch_id = Column(Integer, ForeignKey('batches.id'), nullable=False)
    batch = relationship("Batch", back_populates="objects")
    images = relationship("Image", back_populates="object", cascade="all, delete-orphan")

class Image(Base):
    __tablename__ = 'images'
    id = Column(Integer, primary_key=True)
    camera_label = Column(String(50), nullable=False)
    image_type = Column(String(20), nullable=False)
    data = Column(LargeBinary, nullable=False)
    format = Column(String(10), default='jpg')
    timestamp = Column(DateTime, default=func.now())
    object_id = Column(Integer, ForeignKey('objects.id'), nullable=False)
    object = relationship("Object", back_populates="images")
    detections = relationship("Detection", back_populates="image", cascade="all, delete-orphan")

class Detection(Base):
    __tablename__ = 'detections'
    id = Column(Integer, primary_key=True)
    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)
    class_label = Column(String(50), nullable=False)
    confidence = Column(String(50), nullable=True)
    crop_data = Column(LargeBinary, nullable=True)
    timestamp = Column(DateTime, default=func.now())
    image_id = Column(Integer, ForeignKey('images.id'), nullable=False)
    image = relationship("Image", back_populates="detections")

# Database connection setup
class DatabaseManager:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        Base.metadata.create_all(self.engine)
        
    def get_session(self):
        return self.Session()

def init_database():
    db_config = config.get("database", {})
    db_type = db_config.get("type", "sqlite")
    
    if db_type == "sqlite":
        db_path = db_config.get("path", "camera_system.db")
        connection_string = f"sqlite:///{db_path}"
    elif db_type == "mysql":
        connection_string = f"mysql+pymysql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    elif db_type == "postgresql":
        connection_string = f"postgresql://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    else:
        logger.critical(f"Unsupported database type: {db_type}")
        sys.exit(1)
    
    try:
        db_manager = DatabaseManager(connection_string)
        db_manager.create_tables()
        logger.info(f"Database initialized successfully ({db_type})")
        return db_manager
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}")
        sys.exit(1)

# Initialize YOLO models
try:
    det_model_path = config.get("detection_model_path")
    clf_model_path = config.get("classification_model_path")
    from ultralytics import YOLO
    det_model = YOLO(det_model_path)
    clf_model = YOLO(clf_model_path)
    logger.info("YOLO models loaded successfully")
except Exception as e:
    logger.critical(f"Failed to load models: {e}")
     
    sys.exit(1)

# Authentication
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash("Please log in first.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Camera Management
class CameraManager:
    @staticmethod
    def enumerate_devices():
        pstDevList = MvCC.MV_CC_DEVICE_INFO_LIST()
        nTLayerType = MvCC.MV_GIGE_DEVICE | MvCC.MV_USB_DEVICE
        ret = MvCC.MvCamera.MV_CC_EnumDevices(nTLayerType, pstDevList)
        if ret != 0:
            logger.error(f"Failed to enumerate devices, error code: {ret}")
            return {}
        
        devices = {}
        for i in range(pstDevList.nDeviceNum):
            device_info_ptr = ctypes.cast(pstDevList.pDeviceInfo[i], ctypes.POINTER(MvCC.MV_CC_DEVICE_INFO))
            device_info = device_info_ptr.contents
            serial = ""
            if device_info.nTLayerType == MvCC.MV_GIGE_DEVICE:
                serial = bytes(device_info.SpecialInfo.stGigEInfo.chSerialNumber).decode('utf-8', errors='ignore').strip('\x00')
            elif device_info.nTLayerType == MvCC.MV_USB_DEVICE:
                serial = bytes(device_info.SpecialInfo.stUsb3VInfo.chSerialNumber).decode('utf-8', errors='ignore').strip('\x00')
            
            if serial:
                devices[serial] = device_info
        return devices

    @staticmethod
    def initialize_camera(serial_number):
        device_map = CameraManager.enumerate_devices()
        if serial_number not in device_map:
            raise Exception(f"Camera with serial {serial_number} not found")
        
        try:
            cam = MvCC.MvCamera()
            if cam.MV_CC_CreateHandle(device_map[serial_number]) != 0:
                raise Exception("CreateHandle failed")
            if cam.MV_CC_OpenDevice(MvCC.MV_ACCESS_Exclusive, 0) != 0:
                cam.MV_CC_DestroyHandle()
                raise Exception("OpenDevice failed")
            cam.MV_CC_SetEnumValue("ExposureAuto", MvCC.MV_EXPOSURE_AUTO_MODE_OFF)
            if cam.MV_CC_StartGrabbing() != 0:
                cam.MV_CC_CloseDevice()
                cam.MV_CC_DestroyHandle()
                raise Exception("StartGrabbing failed")
            return cam
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            raise

    @staticmethod
    def release_camera(cam):
        try:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
        except Exception as e:
            logger.error(f"Error releasing camera: {e}")

    @staticmethod
    def get_frame(cam, timeout=3000):
        stOutFrame = MvCC.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
        ret = cam.MV_CC_GetImageBuffer(stFrame=stOutFrame, nMsec=timeout)
        if ret != 0: return None
        
        try:
            buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
            ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
            image_data = np.frombuffer(buf_cache, dtype=np.uint8)
            image = image_data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            
            if stOutFrame.stFrameInfo.enPixelType in [MvCC.PixelType_Gvsp_BayerRG8, MvCC.PixelType_Gvsp_BayerGB8,
                                                     MvCC.PixelType_Gvsp_BayerGR8, MvCC.PixelType_Gvsp_BayerBG8]:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image_rgb
        finally:
            cam.MV_CC_FreeImageBuffer(stOutFrame)

    @staticmethod
    def get_frame_continuous(cam, timeout=3000):
        stOutFrame = MvCC.MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))
        ret = cam.MV_CC_GetImageBuffer(stFrame=stOutFrame, nMsec=timeout)
        if ret != 0: return None
        
        try:
            image_data = np.frombuffer(
                ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)).contents,
                dtype=np.uint8
            )
            image = image_data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            
            if stOutFrame.stFrameInfo.enPixelType in [MvCC.PixelType_Gvsp_BayerRG8, MvCC.PixelType_Gvsp_BayerGB8,
                                                     MvCC.PixelType_Gvsp_BayerGR8, MvCC.PixelType_Gvsp_BayerBG8]:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image_rgb
        finally:
            cam.MV_CC_FreeImageBuffer(stOutFrame)

# Image Processing
class ImageProcessor:
    @staticmethod
    def process_frame(image):
        if image is None or image.size == 0:
            return None, []
        crops_results = []
        
        try:
            results = det_model.predict(source=image,imgsz=1280)
            annotated_image = image.copy()
            
            for result in results:
                for box in result.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box[:4])
                    if x1 >= x2 or y1 >= y2: continue
                    
                    cropped = image[y1:y2, x1:x2]
                    if cropped.size == 0: continue
                    
                    clf_results = clf_model(cropped)
                    class_index = int(clf_results[0].probs.top1)
                    class_label = clf_results[0].names[class_index]
                    confidence = f"{clf_results[0].probs.top1conf:.2f}"
                    
                    color = (0, 255, 0) if class_label.lower() == "present" else (0, 0, 255)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 5)
                    cv2.putText(annotated_image, f"{class_label} ({confidence})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    _, encoded_crop = cv2.imencode('.jpg', cropped)
                    crops_results.append((x1, y1, x2, y2, class_label, confidence, encoded_crop.tobytes()))
            
            return annotated_image, crops_results
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return image, []

    @staticmethod
    def img_to_base64(img):
        try:
            if img is None or img.size == 0:
                raise ValueError("Empty image provided")
            _, buffer = cv2.imencode('.jpg', img)
            return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return None

    @staticmethod
    def base64_to_image(base64_str):
        try:
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',', 1)[1]
            img_data = base64.b64decode(base64_str)
            img_array = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            return None

    @staticmethod
    def img_to_bytes(img):
        try:
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}")
            return None

    @staticmethod
    def save_to_db(db_session, processed_results, shift, batch_number, object_number, username):
        try:
            batch = db_session.query(Batch).filter_by(shift=shift, batch_number=batch_number).first()
            if not batch:
                batch = Batch(shift=shift, batch_number=batch_number, created_by=username)
                db_session.add(batch)
                db_session.flush()
            
            object_record = Object(object_number=object_number, batch_id=batch.id)
            db_session.add(object_record)
            db_session.flush()
            
            for cam_key, result in processed_results.items():
                if "error" in result: continue
                
                camera_label = {"cam1": "left", "cam2": "top", "cam3": "right"}.get(cam_key, cam_key)
                
                if "original" in result and result["original"]:
                    original_img = ImageProcessor.base64_to_image(result["original"])
                    if original_img is not None:
                        db_session.add(Image(
                            camera_label=camera_label,
                            image_type="original",
                            data=ImageProcessor.img_to_bytes(original_img),
                            object_id=object_record.id
                        ))
                
                if "annotated" in result and result["annotated"]:
                    annotated_img = ImageProcessor.base64_to_image(result["annotated"])
                    if annotated_img is not None:
                        annotated_image = Image(
                            camera_label=camera_label,
                            image_type="annotated",
                            data=ImageProcessor.img_to_bytes(annotated_img),
                            object_id=object_record.id
                        )
                        db_session.add(annotated_image)
                        db_session.flush()
                        
                        if "detection_details" in result:
                            for det in result["detection_details"]:
                                db_session.add(Detection(
                                    x1=det[0], y1=det[1], x2=det[2], y2=det[3],
                                    class_label=det[4], confidence=det[5],
                                    crop_data=det[6], image_id=annotated_image.id
                                ))
            
            db_session.commit()
            return True
        except Exception as e:
            db_session.rollback()
            logger.error(f"Database save error: {e}")
            return False

# Streaming Functions
def generate_frames(cam_label):
    global streaming_active, camera_instances, stream_lock
    
    with stream_lock:
        cam = camera_instances.get(cam_label)
        if not cam:
            camera_config = config.get("cameras", {})
            serial = camera_config.get(cam_label)
            if not serial:
                logger.error(f"No serial number configured for camera {cam_label}")
                return
            
            try:
                cam = CameraManager.initialize_camera(serial)
                camera_instances[cam_label] = cam
            except Exception as e:
                logger.error(f"Camera initialization error {cam_label}: {e}")
                return
    
    while streaming_active:
        with stream_lock:
            if not streaming_active:
                break
                
            frame = CameraManager.get_frame_continuous(cam)
            if frame is None:
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('shift_batch'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        users = config.get("users", {})
        
        if check_password_hash(users.get(username, ''), password):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('shift_batch'))
        else:
            flash("Invalid credentials", "danger")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/shift_batch', methods=['GET', 'POST'])
@login_required
def shift_batch():
    if request.method == 'POST':
        session['shift'] = request.form.get('shift')
        session['batch'] = request.form.get('batch')
        return redirect(url_for('capture_form'))
    return render_template('shift.html')

@app.route('/capture_form', methods=['GET', 'POST'])
@login_required
def capture_form():
    if request.method == 'POST':
        session['object_number'] = request.form.get('object_id')
        return redirect(url_for('stream'))
    return render_template('capture_form.html')

@app.route('/stream',endpoint='stream')
@login_required
def stream():
    if not all(k in session for k in ['shift', 'batch', 'object_number']):
        flash("Please complete all previous steps", "warning")
        return redirect(url_for('capture_form'))
    
    camera_config = config.get("cameras", {})
    return render_template('stream.html', cameras=camera_config.keys())

@app.route('/video_feed/<cam_label>')
@login_required
def video_feed(cam_label):
    return Response(generate_frames(cam_label),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream')
@login_required
def start_stream():
    global streaming_active, camera_instances, stream_lock
    
    with stream_lock:
        if streaming_active:
            return '', 204
            
        streaming_active = True
        
        # Initialize all configured cameras
        camera_config = config.get("cameras", {})
        for cam_label, serial in camera_config.items():
            if cam_label in camera_instances:
                continue
                
            try:
                cam = CameraManager.initialize_camera(serial)
                camera_instances[cam_label] = cam
            except Exception as e:
                logger.error(f"Error initializing camera {cam_label}: {e}")
                continue
    
    return '', 204

@app.route('/stop_stream')
@login_required
def stop_stream():
    global streaming_active, camera_instances, stream_lock
    
    with stream_lock:
        streaming_active = False
        time.sleep(0.5)
        
        for cam_label, cam in camera_instances.items():
            try:
                CameraManager.release_camera(cam)
            except Exception as e:
                logger.error(f"Error releasing camera {cam_label}: {e}")
        
        camera_instances = {}
    return '', 204

@app.route('/capture')
@login_required
def capture():
    global captured_frames, camera_instances, stream_lock
    
    with stream_lock:
        if not camera_instances:
            flash("Please start the stream first", "warning")
            return redirect(url_for('stream'))
        
        captured_frames = {}
        for cam_label, cam in camera_instances.items():
            frame = CameraManager.get_frame(cam)
            if frame is not None:
                captured_frames[cam_label] = frame
    
    if not captured_frames:
        flash("Failed to capture frames", "danger")
        return redirect(url_for('stream'))
    
    display_frames = {k: ImageProcessor.img_to_base64(v) for k, v in captured_frames.items()}
    return render_template('results.html', mode="capture", results=display_frames)

@app.route('/process', methods=['POST'])
@login_required
def process_images():
    global captured_frames
    if not captured_frames:
        flash("No frames captured", "warning")
        return redirect(url_for('capture_form'))
    
    processed_results = {}
    for cam_key, frame in captured_frames.items():
        if frame is None:
            processed_results[cam_key] = {"error": "Capture failed"}
            continue
        
        annotated, crops = ImageProcessor.process_frame(frame)
        processed_results[cam_key] = {
            "original": ImageProcessor.img_to_base64(frame),
            "annotated": ImageProcessor.img_to_base64(annotated),
            "detection_details": crops
        }
    
    db_session = db_manager.get_session()
    try:
        success = ImageProcessor.save_to_db(
            db_session,
            processed_results,
            session['shift'],
            session['batch'],
            session['object_number'],
            session['username']
        )
        flash("Processed successfully" if success else "Processing failed", "success" if success else "danger")
    finally:
        db_session.close()
    
    return render_template('results.html', mode="process", results=processed_results)

@app.route('/history')
@login_required
def view_history():
    db_session = db_manager.get_session()
    try:
        batches = db_session.query(Batch).order_by(Batch.timestamp.desc()).limit(20).all()
        batch_data = [{
            'id': b.id,
            'shift': b.shift,
            'batch_number': b.batch_number,
            'timestamp': b.timestamp,
            'created_by': b.created_by,
            'objects_count': len(b.objects)
        } for b in batches]
        return render_template('history.html', batches=batch_data)
    finally:
        db_session.close()

@app.route('/view_batch/<int:batch_id>')
@login_required
def view_batch(batch_id):
    db_session = db_manager.get_session()
    try:
        batch = db_session.query(Batch).get(batch_id)
        if not batch: return redirect(url_for('view_history'))
        
        object_data = [{
            'id': obj.id,
            'object_number': obj.object_number,
            'timestamp': obj.timestamp,
            'images_count': len(obj.images)
        } for obj in batch.objects]
        
        return render_template('batch_view.html', batch=batch, objects=object_data)
    finally:
        db_session.close()

@app.route('/view_object/<int:object_id>')
@login_required
def view_object(object_id):
    db_session = db_manager.get_session()
    try:
        obj = db_session.query(Object).get(object_id)
        if not obj: return redirect(url_for('view_history'))
        
        images = {}
        for img in obj.images:
            img_base64 = base64.b64encode(img.data).decode('utf-8')
            if img.camera_label not in images:
                images[img.camera_label] = {}
            
            images[img.camera_label][img.image_type] = f"data:image/jpeg;base64,{img_base64}"
            
            if img.image_type == "annotated":
                detections = []
                for det in img.detections:
                    crop_base64 = base64.b64encode(det.crop_data).decode('utf-8') if det.crop_data else None
                    detections.append({
                        'coordinates': [det.x1, det.y1, det.x2, det.y2],
                        'class': det.class_label,
                        'confidence': det.confidence,
                        'crop': f"data:image/jpeg;base64,{crop_base64}" if crop_base64 else None
                    })
                images[img.camera_label]['detections'] = detections
        
        return render_template('object_view.html', 
                             object=obj,
                             batch=obj.batch,
                             images=images)
    finally:
        db_session.close()

# App Initialization
def init_app():
    global db_manager
    db_manager = init_database()
    
    if not config.get("users"):
        config["users"] = {"admin": generate_password_hash(os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin"))}
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)
    
    return app

if __name__ == '__main__':
    app = init_app()
    serve(app, host=config.get('host', '0.0.0.0'), port=config.get('port', 5000), threads=4)
