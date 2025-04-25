# respiration_monitor.py
# Core classes for video capture, calibration, motion detection, and signal processing.
# v7: Fixed PCA input shape and fallback IndexError in Optical Flow.

import cv2
import numpy as np
from scipy import signal as sp_signal
from scipy.optimize import curve_fit
import collections
import time
import warnings
import json # Import json library
import os   # Import os library for path handling
import traceback # For detailed error printing

# Suppress RuntimeWarning from mean of empty slice etc.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Default values (used if keys are missing in JSON) ---
MINIMAL_DEFAULTS = {
    'VIDEO_SOURCE': 0,
    'DEFAULT_MOTION_METHOD': 'ROI_Average',
    'SIGNAL_BUFFER_SIZE': 300,
    'LOWPASS_FILTER_CUTOFF_HZ': 2.0,
    'LOWPASS_FILTER_ORDER': 4,
    'MAX_EXPECTED_BPM': 120,
    'PEAK_FINDING_DISTANCE_FACTOR': 0.5,
    'PEAK_FINDING_PROMINENCE_FACTOR': 0.1,
    'OPTICAL_FLOW_PARAMS': { # Need defaults for nested dicts too
        'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
        'lk_params': {'winSize': [15, 15], 'maxLevel': 2, 'criteria': [3, 10, 0.03]} # Use list/int for JSON compatibility
    },
    # Add other critical defaults if necessary
    'CALIBRATION_METHOD': 'MotionVariance',
    'CALIBRATION_DURATION_FRAMES': 150,
    'ROI_AVG_PREFILTER': 'Median',
    'ROI_AVG_MEDIAN_KSIZE': 5,
    'ROI_AVG_GAUSSIAN_KERNEL': (3, 3),
    'ROI_REFINE_SIZE': False,
    'USE_GAUSSIAN_FIT': False,
    'GAUSSIAN_FIT_WINDOW_FACTOR': 3,
    'GAUSSIAN_FIT_MIN_STDDEV': 1.0,
    'GAUSSIAN_FIT_MAX_STDDEV': 50.0,
    'USE_OPENCL': False,
    'USE_ADAPTIVE_CONTROL': False,
    'ADAPTIVE_FPS_THRESHOLD': 15,
    'ADAPTIVE_BPM_STABILITY_THRESHOLD': 15,
    'ADAPTIVE_HYSTERESIS_FRAMES': 60,
    'DEBUG_PRINT_VALUES': False,
    'DEBUG_CALIBRATION_VIZ': False,
    'EVM_PYRAMID_LEVEL': 2,
    'EVM_LOW_FREQ_HZ': 0.4,
    'EVM_HIGH_FREQ_HZ': 2.0,
}


# --- Helper function to load and merge config ---
def load_config(filepath):
    """Loads config from JSON file and merges with minimal defaults."""
    config = MINIMAL_DEFAULTS.copy() # Start with minimal defaults
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                loaded_config = json.load(f)
            # Merge loaded config, overwriting defaults
            config.update(loaded_config)
            print(f"Loaded configuration from: {filepath}")
            # Special handling for nested dicts like OPTICAL_FLOW_PARAMS
            if 'OPTICAL_FLOW_PARAMS' in loaded_config and isinstance(loaded_config['OPTICAL_FLOW_PARAMS'], dict):
                 default_ofp = MINIMAL_DEFAULTS.get('OPTICAL_FLOW_PARAMS', {})
                 loaded_ofp = loaded_config['OPTICAL_FLOW_PARAMS']
                 merged_fp = default_ofp.get('feature_params', {}).copy()
                 merged_fp.update(loaded_ofp.get('feature_params', {}))
                 merged_lk = default_ofp.get('lk_params', {}).copy()
                 loaded_lk = loaded_ofp.get('lk_params', {})
                 merged_lk.update(loaded_lk)
                 config['OPTICAL_FLOW_PARAMS'] = {'feature_params': merged_fp, 'lk_params': merged_lk}

            # Convert list from JSON back to tuple for specific OpenCV params if needed AFTER merging
            if 'lk_params' in config.get('OPTICAL_FLOW_PARAMS', {}) and \
               'winSize' in config['OPTICAL_FLOW_PARAMS']['lk_params'] and \
               isinstance(config['OPTICAL_FLOW_PARAMS']['lk_params']['winSize'], list):
                config['OPTICAL_FLOW_PARAMS']['lk_params']['winSize'] = tuple(config['OPTICAL_FLOW_PARAMS']['lk_params']['winSize'])
            if 'ROI_AVG_GAUSSIAN_KERNEL' in config and isinstance(config['ROI_AVG_GAUSSIAN_KERNEL'], list):
                 config['ROI_AVG_GAUSSIAN_KERNEL'] = tuple(config['ROI_AVG_GAUSSIAN_KERNEL'])
        else:
            print(f"Warning: Config file not found at '{filepath}'. Using minimal defaults.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filepath}': {e}. Using minimal defaults.")
    except Exception as e:
        print(f"Error loading config file '{filepath}': {e}. Using minimal defaults.")
    return config

# --- Gaussian Function for Fitting ---
def gaussian(x, amplitude, mean, stddev):
    """Gaussian function for curve fitting."""
    stddev = max(abs(stddev), 1e-6) # Avoid division by zero or negative stddev
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# --- Module: VideoInput ---
# (No changes)
class VideoInput:
    """Handles video capture from webcam or file."""
    def __init__(self, config):
        self.config = config; self.capture = None; self.fps = 30; self.use_opencl = False; self.frame_width = 0; self.frame_height = 0
    def initialize(self):
        """Initializes the video capture source."""
        try:
            source = self.config.get('VIDEO_SOURCE', 0); self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened(): raise IOError(f"Cannot open video source: {source}")
            self.fps = self.capture.get(cv2.CAP_PROP_FPS); self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)); self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self.fps <= 0: self.fps = 30; print("Warning: Could not get FPS, defaulting to 30.")
            self.use_opencl = self.config.get('USE_OPENCL', False) and cv2.ocl.haveOpenCL(); cv2.ocl.setUseOpenCL(self.use_opencl)
            print(f"Video source initialized. Res: {self.frame_width}x{self.frame_height}, FPS: {self.fps:.2f}, OpenCL: {self.use_opencl}")
            return True
        except IOError as e: print(f"Error initializing video input (IOError): {e}"); self.capture = None; return False
        except cv2.error as e: print(f"Error initializing video input (OpenCV Error): {e}"); self.capture = None; return False
        except Exception as e: print(f"Unexpected error initializing video input: {e}"); self.capture = None; return False
    def get_frame(self):
        """Retrieves the next frame from the source."""
        if self.capture is None or not self.capture.isOpened(): return False, None
        try:
            success, frame = self.capture.read()
            if not success: return False, None
            if self.use_opencl:
                try: return success, cv2.UMat(frame)
                except (cv2.error, Exception) as e: print(f"Warning: UMat conversion failed ({e}), using NumPy."); self.use_opencl = False; cv2.ocl.setUseOpenCL(False); return success, frame
            else: return success, frame
        except cv2.error as e: print(f"Error reading frame (OpenCV Error): {e}"); return False, None
        except Exception as e: print(f"Unexpected error reading frame: {e}"); return False, None
    def get_fps(self): return self.fps
    def get_frame_size(self): return self.frame_width, self.frame_height
    def release(self):
        """Releases the video capture object."""
        if self.capture:
            try: self.capture.release(); print("Video source released.")
            except Exception as e: print(f"Error releasing video source: {e}")

# --- Module: Calibration ---
# (No changes)
class Calibration:
    """Handles ROI selection based on motion analysis."""
    def __init__(self, video_input, config):
        self.video_input = video_input; self.config = config; self.use_opencl = video_input.use_opencl; self.debug_viz = config.get('DEBUG_CALIBRATION_VIZ', False)
    def run_calibration(self):
        """Runs the calibration process to find the ROI."""
        method = self.config.get('CALIBRATION_METHOD', 'MotionVariance'); print(f"Starting calibration using '{method}' method...")
        frames_for_calib_gray = []; frames_for_calib_color = []; start_time = time.time(); calibration_duration = self.config.get('CALIBRATION_DURATION_FRAMES', 150)
        try:
            for i in range(calibration_duration):
                success, frame = self.video_input.get_frame()
                if not success: print("Warning: Video source ended during calibration."); break
                frame_np = frame.get() if isinstance(frame, cv2.UMat) else frame; frames_for_calib_color.append(frame_np)
                if isinstance(frame, cv2.UMat): gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else: gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_for_calib_gray.append(gray_frame)
                if self.debug_viz and i % 15 == 0:
                     viz_frame = frame_np.copy(); cv2.putText(viz_frame, f"Calibrating... Frame {i+1}/{calibration_duration}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1); cv2.imshow("Calibration Progress", viz_frame); cv2.waitKey(1)
        except Exception as e: print(f"Error during frame collection for calibration: {e}"); return None
        finally:
             if self.debug_viz: cv2.destroyWindow("Calibration Progress"); cv2.waitKey(1) # Ensure window closes
        if len(frames_for_calib_gray) < 20: print("Error: Not enough frames for calibration."); return None
        print(f"Collected {len(frames_for_calib_gray)} frames for calibration in {time.time() - start_time:.2f} seconds.")
        try:
            if method == 'SimplifiedEVM': roi = self._perform_simplified_evm(frames_for_calib_gray)
            elif method == 'MotionVariance': roi = self._find_roi_by_motion_variance(frames_for_calib_gray)
            else: print(f"Error: Unknown calibration method '{method}'."); return None
            if roi is not None and self.config.get('ROI_REFINE_SIZE', False):
                 print("Performing ROI Size Refinement..."); refine_frames_count = min(len(frames_for_calib_color), int(self.video_input.get_fps() * 3))
                 if refine_frames_count > 10: refined_roi = self._refine_roi_size(roi, frames_for_calib_color[:refine_frames_count]); roi = refined_roi or roi
                 else: print("Not enough frames for ROI refinement.")
            if roi is None: print("Calibration failed to find ROI.")
            else: print(f"Calibration complete. Selected ROI: {roi}")
            return roi
        except cv2.error as e_cv: print(f"OpenCV error during calibration ({method}): {e_cv}"); return None
        except Exception as e: print(f"Unexpected error during calibration ({method}): {e}"); traceback.print_exc(); return None
        finally:
            if self.debug_viz: cv2.destroyWindow("Variance Map (Blurred - EVM)"); cv2.destroyWindow("Variance Map (Raw - EVM)"); cv2.destroyWindow("Variance Map (Blurred - MotionVar)"); cv2.destroyWindow("Variance Map (Raw - MotionVar)"); cv2.waitKey(1)
    def _build_gaussian_pyramid(self, frame, levels):
        """Builds a Gaussian pyramid using OpenCV."""
        pyramid = [frame]; current_level = frame
        for i in range(levels):
            try:
                if isinstance(current_level, cv2.UMat):
                     try: next_level = cv2.pyrDown(current_level)
                     except cv2.error: next_level = cv2.pyrDown(current_level.get())
                else: next_level = cv2.pyrDown(current_level)
                pyramid.append(next_level); current_level = next_level
            except cv2.error as e: print(f"Error in pyrDown at level {i}: {e}"); raise
        return pyramid
    def _temporal_filter(self, data_sequence, fs, low_hz, high_hz):
        """Applies a temporal bandpass filter using SciPy."""
        if data_sequence.shape[0] < 10: return None
        try: sos = sp_signal.butter(4, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos'); return sp_signal.sosfiltfilt(sos, data_sequence, axis=0)
        except ValueError as e: print(f"Error designing/applying temporal filter (ValueError): {e}."); return None
        except Exception as e: print(f"Error during temporal filtering (Exception): {e}"); return None
    def _visualize_variance_map(self, variance_map, window_title="Variance Map"):
         """Helper function to display variance maps for debugging."""
         if variance_map is None or variance_map.size == 0: return
         try: norm_map = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U); color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET); cv2.imshow(window_title, color_map)
         except cv2.error as e: print(f"Error visualizing variance map '{window_title}': {e}")
         except Exception as e: print(f"Unexpected error visualizing variance map '{window_title}': {e}")
    def _perform_simplified_evm(self, gray_frames):
        """Simplified EVM approach: Temporal filtering of a specific pyramid level."""
        target_level = self.config.get('EVM_PYRAMID_LEVEL', 2); fs = self.video_input.get_fps(); low_hz = self.config.get('EVM_LOW_FREQ_HZ', 0.4); high_hz = self.config.get('EVM_HIGH_FREQ_HZ', 2.0)
        if fs <= 0 or high_hz >= fs / 2.0: print(f"Error: Invalid frequencies for temporal filter."); return self._find_roi_by_motion_variance(gray_frames)
        pyramids = []
        for frame in gray_frames:
            pyramid = self._build_gaussian_pyramid(frame, target_level)
            if len(pyramid) > target_level: pyramids.append(pyramid[target_level])
            else: print(f"Warning: Could not build pyramid to level {target_level}"); return self._find_roi_by_motion_variance(gray_frames)
        if not pyramids: return None
        target_shape = pyramids[0].shape; pyramid_data_list = [p.get() if isinstance(p, cv2.UMat) else p for p in pyramids if p.shape == target_shape]
        if len(pyramid_data_list) < 10: print("Error: Not enough consistent pyramid levels."); return self._find_roi_by_motion_variance(gray_frames)
        pyramid_sequence = np.stack(pyramid_data_list, axis=0)
        filtered_sequence = self._temporal_filter(pyramid_sequence, fs, low_hz, high_hz)
        if filtered_sequence is None: print("Temporal filtering failed."); return self._find_roi_by_motion_variance(gray_frames)
        variance_map = np.var(filtered_sequence, axis=0); variance_map_blurred = cv2.GaussianBlur(variance_map.astype(np.float32), (15, 15), 0)
        if self.debug_viz: self._visualize_variance_map(variance_map, "Variance Map (Raw - EVM)"); self._visualize_variance_map(variance_map_blurred, "Variance Map (Blurred - EVM)"); cv2.waitKey(1)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(variance_map_blurred)
        print(f"Simplified EVM - Max Variance: {maxVal:.4f} at {maxLoc} (Level {target_level})")
        if maxVal <= 1e-6: print("Warning: No significant variance found in filtered signal."); return self._find_roi_by_motion_variance(gray_frames)
        level_h, level_w = target_shape[:2]; orig_w, orig_h = self.video_input.get_frame_size(); scale_factor = 2**target_level
        roi_w_orig = max(20, orig_w // 4); roi_h_orig = max(20, orig_h // 4); center_x_orig = maxLoc[0] * scale_factor; center_y_orig = maxLoc[1] * scale_factor
        roi_x_orig = max(0, min(orig_w - roi_w_orig, center_x_orig - roi_w_orig // 2)); roi_y_orig = max(0, min(orig_h - roi_h_orig, center_y_orig - roi_h_orig // 2))
        return (int(roi_x_orig), int(roi_y_orig), int(roi_w_orig), int(roi_h_orig))
    def _find_roi_by_motion_variance(self, frames):
        """Finds ROI based on variance of frame differences."""
        if len(frames) < 5: return None
        frames_np = [(f.get() if isinstance(f, cv2.UMat) else f) for f in frames]; frame_h, frame_w = frames_np[0].shape[:2]
        diffs = [cv2.absdiff(frames_np[i], frames_np[i+1]) for i in range(len(frames_np)-1)]
        if not diffs: return None
        diff_stack = np.stack(diffs, axis=0); variance_map = np.var(diff_stack, axis=0); variance_map_blurred = cv2.GaussianBlur(variance_map.astype(np.float32), (15, 15), 0)
        if self.debug_viz: self._visualize_variance_map(variance_map, "Variance Map (Raw - MotionVar)"); self._visualize_variance_map(variance_map_blurred, "Variance Map (Blurred - MotionVar)"); cv2.waitKey(1)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(variance_map_blurred)
        print(f"Motion Variance - Max Variance: {maxVal:.4f} at {maxLoc}")
        if maxVal <= 1e-6: print("Warning: No significant motion variance found."); roi_w = max(20, frame_w // 4); roi_h = max(20, frame_h // 4); roi_x = max(0, frame_w // 2 - roi_w // 2); roi_y = max(0, frame_h // 2 - roi_h // 2); return (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
        roi_w = max(20, frame_w // 4); roi_h = max(20, frame_h // 4); roi_x = max(0, min(frame_w - roi_w, maxLoc[0] - roi_w // 2)); roi_y = max(0, min(frame_h - roi_h, maxLoc[1] - roi_h // 2))
        return (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
    def _calculate_roi_signal_variance(self, roi, frames_np):
        """Helper to calculate variance of ROI average signal for refinement."""
        x, y, w, h = roi; motion_signal = []; prefilter = self.config.get('ROI_AVG_PREFILTER'); median_ksize = self.config.get('ROI_AVG_MEDIAN_KSIZE', 5); gaussian_kernel = self.config.get('ROI_AVG_GAUSSIAN_KERNEL', (3,3))
        if w <= 0 or h <= 0: return 0
        for frame in frames_np:
            try:
                if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0: continue
                actual_y_end = min(frame.shape[0], y + h); actual_x_end = min(frame.shape[1], x + w); actual_y = max(0, y); actual_x = max(0, x)
                if actual_y_end <= actual_y or actual_x_end <= actual_x: continue
                roi_frame = frame[actual_y:actual_y_end, actual_x:actual_x_end]
                if roi_frame.shape[0] == 0 or roi_frame.shape[1] == 0: continue
                filtered_roi = roi_frame
                try:
                    if prefilter == 'Median': filtered_roi = cv2.medianBlur(roi_frame, median_ksize)
                    elif prefilter == 'Gaussian': filtered_roi = cv2.GaussianBlur(roi_frame, gaussian_kernel, 0)
                    if len(filtered_roi.shape) == 3: gray_roi = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2GRAY)
                    else: gray_roi = filtered_roi
                    motion_signal.append(cv2.mean(gray_roi)[0])
                except (cv2.error, Exception): continue # Ignore errors during inner processing
            except Exception as e_outer: print(f"Warning: Error processing frame for ROI signal variance: {e_outer}"); continue
        if len(motion_signal) < 5: return 0
        return np.var(motion_signal)
    def _refine_roi_size(self, initial_roi, frames_np):
         """Tests ROI size variations and selects based on signal variance."""
         orig_x, orig_y, orig_w, orig_h = initial_roi; frame_h, frame_w = frames_np[0].shape[:2]; best_roi = initial_roi
         max_variance = self._calculate_roi_signal_variance(initial_roi, frames_np); print(f"  Initial ROI variance: {max_variance:.4f}")
         scale_factors = [0.8, 0.9, 1.1, 1.2]
         for scale_w in scale_factors:
             for scale_h in scale_factors:
                 if scale_w == 1.0 and scale_h == 1.0: continue
                 new_w = int(orig_w * scale_w); new_h = int(orig_h * scale_h); new_x = orig_x + (orig_w - new_w) // 2; new_y = orig_y + (orig_h - new_h) // 2
                 new_x = max(0, min(frame_w - new_w, new_x)); new_y = max(0, min(frame_h - new_h, new_y)); new_w = max(10, min(frame_w - new_x, new_w)); new_h = max(10, min(frame_h - new_y, new_h))
                 current_roi = (new_x, new_y, new_w, new_h)
                 if current_roi == best_roi: continue
                 variance = self._calculate_roi_signal_variance(current_roi, frames_np)
                 if variance > max_variance: max_variance = variance; best_roi = current_roi; print(f"  New best ROI: {best_roi} (Variance: {max_variance:.4f})")
         return best_roi

# --- Module: MotionDetector ---
class MotionDetector:
    """Calculates motion signal using ROI Average or Optical Flow."""
    def __init__(self, roi, config):
        if roi is None or len(roi) != 4: raise ValueError("Invalid ROI provided to MotionDetector")
        self.roi = roi; self.config = config; self.method = config.get('DEFAULT_MOTION_METHOD', 'ROI_Average'); self.prev_frame_gray = None; self.prev_points = None; self.tracked_points_count = 0
        # Safely access nested optical flow params with defaults
        of_params = config.get('OPTICAL_FLOW_PARAMS', {})
        default_feature_params = MINIMAL_DEFAULTS['OPTICAL_FLOW_PARAMS']['feature_params']
        default_lk_params = MINIMAL_DEFAULTS['OPTICAL_FLOW_PARAMS']['lk_params']
        self.feature_params = {**default_feature_params, **of_params.get('feature_params', {})}
        self.lk_params = {**default_lk_params, **of_params.get('lk_params', {})}

        # Ensure criteria is correctly formatted (handle list from JSON)
        criteria_val = self.lk_params.get('criteria')
        default_criteria_tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        if isinstance(criteria_val, list) and len(criteria_val) == 3:
             try: type_flags = int(criteria_val[0]); max_iter = int(criteria_val[1]); epsilon = float(criteria_val[2]); self.lk_params['criteria'] = (type_flags, max_iter, epsilon)
             except (ValueError, TypeError): print(f"Warning: Invalid format for lk_params 'criteria': {criteria_val}. Using default."); self.lk_params['criteria'] = default_criteria_tuple
        elif not isinstance(criteria_val, tuple): self.lk_params['criteria'] = default_criteria_tuple

        # Ensure winSize is a tuple
        if isinstance(self.lk_params.get('winSize'), list):
             try: self.lk_params['winSize'] = tuple(self.lk_params['winSize'])
             except TypeError: self.lk_params['winSize'] = (15, 15)
        elif not isinstance(self.lk_params.get('winSize'), tuple): self.lk_params['winSize'] = (15, 15)

        self.roi_prefilter = config.get('ROI_AVG_PREFILTER'); self.gaussian_kernel = tuple(config.get('ROI_AVG_GAUSSIAN_KERNEL', (3,3))); self.median_ksize = config.get('ROI_AVG_MEDIAN_KSIZE', 5)
        self.use_opencl = config.get('USE_OPENCL', False) and cv2.ocl.haveOpenCL(); print(f"MotionDetector initialized with method: {self.method}, ROI: {self.roi}")

    def set_method(self, method_name):
        """Sets the motion detection method."""
        if method_name in ['ROI_Average', 'OpticalFlow']:
            if self.method != method_name: print(f"Switching motion detection method to: {method_name}"); self.method = method_name; self.prev_frame_gray = None; self.prev_points = None; self.tracked_points_count = 0
        else: print(f"Warning: Invalid motion detection method '{method_name}'. Keeping '{self.method}'.")

    def get_tracked_points_count(self):
        """Returns the number of points successfully tracked in the last Optical Flow step."""
        return self.tracked_points_count if self.method == 'OpticalFlow' else 0

    def process_frame(self, frame):
        """Processes a single frame to extract a motion value."""
        x, y, w, h = self.roi; motion_value = 0.0
        if w <= 0 or h <= 0: return motion_value
        is_umat = isinstance(frame, cv2.UMat)
        try: # Wrap entire frame processing
            try: # ROI Slicing
                frame_h, frame_w = frame.shape[:2] if not is_umat else frame.size()[::-1]; y_start = max(0, y); y_end = min(frame_h, y + h); x_start = max(0, x); x_end = min(frame_w, x + w)
                if y_start >= y_end or x_start >= x_end: return motion_value
                roi_frame = frame[y_start:y_end, x_start:x_end]
            except (cv2.error, Exception) as e_slice: # Fallback slicing
                 print(f"Error slicing ROI: {e_slice}. Trying fallback.");
                 frame_np = frame.get() if is_umat else frame; frame_h, frame_w = frame_np.shape[:2]; y_start = max(0, y); y_end = min(frame_h, y + h); x_start = max(0, x); x_end = min(frame_w, x + w)
                 if y_start >= y_end or x_start >= x_end: return motion_value
                 roi_frame = frame_np[y_start:y_end, x_start:x_end]; is_umat = False # Ensure is_umat is updated on fallback

            if roi_frame is None or roi_frame.shape[0] == 0 or roi_frame.shape[1] == 0: return motion_value

            self.tracked_points_count = 0
            if self.method == 'ROI_Average': motion_value = self._detect_roi_average(roi_frame)
            elif self.method == 'OpticalFlow':
                try: # Grayscale conversion
                    if len(frame.shape) == 3 or (is_umat and len(frame.size()) == 3):
                         if is_umat:
                              try: frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                              except cv2.error: frame_gray = cv2.cvtColor(frame.get(), cv2.COLOR_BGR2GRAY) # Fallback
                         else: frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else: frame_gray = frame # Assume already grayscale
                except cv2.error as e_cvt: print(f"Error converting frame to grayscale: {e_cvt}"); return motion_value # Cannot proceed
                motion_value = self._detect_optical_flow(frame_gray); self.prev_frame_gray = frame_gray # Store full gray frame
            return float(motion_value) # Ensure float output
        except cv2.error as e_cv: print(f"OpenCV error during motion detection ({self.method}): {e_cv}"); return 0.0
        except Exception as e: print(f"Unexpected error during motion detection ({self.method}): {e}"); traceback.print_exc(); return 0.0

    def _detect_roi_average(self, roi_frame):
        """Calculates motion based on average intensity in ROI."""
        filtered_roi = roi_frame
        try: # Pre-processing
            if len(roi_frame.shape) == 3 or (isinstance(roi_frame, cv2.UMat) and len(roi_frame.size()) == 3): gray_roi_unfiltered = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else: gray_roi_unfiltered = roi_frame
            filtered_roi = gray_roi_unfiltered
            if self.roi_prefilter == 'Median': filtered_roi = cv2.medianBlur(gray_roi_unfiltered, self.median_ksize)
            elif self.roi_prefilter == 'Gaussian': filtered_roi = cv2.GaussianBlur(gray_roi_unfiltered, self.gaussian_kernel, 0)
        except cv2.error as e_filter: print(f"Warning: ROI pre-processing failed (OpenCV Error): {e_filter}."); filtered_roi = gray_roi_unfiltered
        except Exception as e_filter_other: print(f"Warning: ROI pre-processing failed (Other Error): {e_filter_other}."); filtered_roi = gray_roi_unfiltered
        try: # Mean calculation
            if isinstance(filtered_roi, cv2.UMat): mean_val = cv2.mean(filtered_roi.get())[0]
            else: mean_val = cv2.mean(filtered_roi)[0]
            return mean_val
        except cv2.error as e_mean_cv: print(f"Error calculating mean in ROI average (OpenCV Error): {e_mean_cv}"); return 0.0
        except Exception as e_mean: print(f"Error calculating mean in ROI average (Exception): {e_mean}"); return 0.0

    def _detect_optical_flow(self, frame_gray):
        """Calculates motion using Lucas-Kanade Optical Flow."""
        current_frame_gray_internal = frame_gray; motion_signal = 0.0
        is_umat = isinstance(current_frame_gray_internal, cv2.UMat); frame_h, frame_w = current_frame_gray_internal.shape[:2] if not is_umat else current_frame_gray_internal.size()[::-1]
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8); x, y, w, h = self.roi; y_start, y_end = max(0, y), min(frame_h, y + h); x_start, x_end = max(0, x), min(frame_w, x + w)
        if y_start < y_end and x_start < x_end: mask[y_start:y_end, x_start:x_end] = 255

        # --- Feature Detection/Update ---
        detect_features = False
        if self.prev_points is None or len(self.prev_points) == 0:
            detect_features = True
            detection_frame = self.prev_frame_gray if self.prev_frame_gray is not None else current_frame_gray_internal

        if detect_features:
            self.prev_points = None; self.tracked_points_count = 0
            if detection_frame is not None:
                try:
                    det_h, det_w = detection_frame.shape[:2] if not isinstance(detection_frame, cv2.UMat) else detection_frame.size()[::-1]
                    if mask.shape[0] != det_h or mask.shape[1] != det_w: mask = np.zeros((det_h, det_w), dtype=np.uint8); mask[y_start:y_end, x_start:x_end] = 255
                    self.prev_points = cv2.goodFeaturesToTrack(detection_frame, mask=mask, **self.feature_params); self.tracked_points_count = len(self.prev_points) if self.prev_points is not None else 0
                except cv2.error as e: print(f"Error in goodFeaturesToTrack detect (OpenCV Error): {e}")
                except Exception as e: print(f"Error in goodFeaturesToTrack detect (Exception): {e}")
            else: print("Warning: Cannot detect features, no valid previous or current frame.")

        # --- Feature Tracking ---
        tracking_successful = False
        if self.prev_frame_gray is not None and self.prev_points is not None and len(self.prev_points) > 0:
            try: next_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_frame_gray, current_frame_gray_internal, self.prev_points, None, **self.lk_params)
            except cv2.error as e: print(f"Error in calcOpticalFlowPyrLK (OpenCV Error): {e}"); next_points, status = None, None
            except Exception as e: print(f"Error in calcOpticalFlowPyrLK (Exception): {e}"); next_points, status = None, None

            if next_points is not None and status is not None:
                good_new = next_points[status.flatten() == 1]; good_old = self.prev_points[status.flatten() == 1]; self.tracked_points_count = len(good_new)
                if self.tracked_points_count >= 4: # Need min points for PCA and fallback
                    try: # PCA Calculation
                        displacements = good_new - good_old
                        # --- FIX: Reshape for PCACompute ---
                        # PCACompute expects samples per row, so shape (M, 2)
                        disp_np = np.float32(displacements.reshape(-1, 2))
                        if disp_np.shape[0] >= 2: # Ensure at least 2 samples for PCA
                            mean, eigenvectors = cv2.PCACompute(disp_np, mean=None, maxComponents=1)
                            mean_displacement = np.mean(disp_np, axis=0)
                            motion_signal = np.dot(mean_displacement, eigenvectors[0].T)
                            self.prev_points = good_new.reshape(-1, 1, 2) # Update points
                            tracking_successful = True # Mark as successful
                        else:
                             print("Warning: Not enough valid displacements for PCA after reshape.")
                             motion_signal = 0.0 # Not enough data for PCA
                             # Force re-detection by setting prev_points to None
                             self.prev_points = None
                             self.tracked_points_count = 0

                    except (cv2.error, Exception) as e_pca:
                        print(f"Error during PCA/Eigenvector signal extraction: {e_pca}")
                        # --- FIX: Reshape for Fallback ---
                        try:
                            # Reshape good_new/good_old to (M, 2) before accessing index 1
                            good_new_flat = good_new.reshape(-1, 2)
                            good_old_flat = good_old.reshape(-1, 2)
                            if good_new_flat.shape[1] == 2 and good_old_flat.shape[1] == 2: # Check shape after reshape
                                y_disp = good_new_flat[:, 1] - good_old_flat[:, 1]
                                motion_signal = np.mean(y_disp) if len(y_disp) > 0 else 0.0
                            else:
                                print("Warning: Fallback y_disp calculation failed due to unexpected shape.")
                                motion_signal = 0.0
                        except IndexError as e_index:
                             print(f"Fallback y_disp calculation failed (IndexError): {e_index}")
                             motion_signal = 0.0
                        # Force re-detection next time after PCA error
                        self.prev_points = None
                        self.tracked_points_count = 0
                # else: # Not enough points tracked, handled below
                #     pass
            # else: # Tracking failed (next_points or status is None), handled below
            #     pass

        # Reset points if tracking wasn't successful or not enough points
        if not tracking_successful:
             self.prev_points = None
             self.tracked_points_count = 0
             motion_signal = 0.0 # Ensure signal is zero if tracking fails

        # Always update the previous frame for the *next* iteration's tracking or detection
        self.prev_frame_gray = current_frame_gray_internal
        return motion_signal


# --- Module: SignalProcessor ---
class SignalProcessor:
    """Filters motion signal, detects peaks, and calculates BPM."""
    def __init__(self, sampling_rate, config):
        if sampling_rate <= 0: raise ValueError("Sampling rate must be positive.")
        self.config = config; self.buffer = collections.deque(maxlen=config.get('SIGNAL_BUFFER_SIZE', 300)); self.sampling_rate = sampling_rate; self.filter_cutoff = config.get('LOWPASS_FILTER_CUTOFF_HZ', 2.0); self.filter_order = config.get('LOWPASS_FILTER_ORDER', 4)
        self.use_gaussian_fit = config.get('USE_GAUSSIAN_FIT', False); self.min_buffer_fill = max(int(sampling_rate * 1.5), config.get('SIGNAL_BUFFER_SIZE', 300) // 3); self.last_bpms = collections.deque(maxlen=10); self.debug_prints = config.get('DEBUG_PRINT_VALUES', False)
        self.last_valid_filtered_signal = None; self.filter_sos = None
        if self.filter_cutoff > 0 and self.filter_cutoff < self.sampling_rate / 2.0:
            try: self.filter_sos = sp_signal.butter(self.filter_order, self.filter_cutoff, btype='low', fs=self.sampling_rate, output='sos'); print(f"SignalProcessor initialized. Sample Rate: {self.sampling_rate:.2f} Hz, Filter: Lowpass Butterworth Order {self.filter_order} Cutoff {self.filter_cutoff} Hz")
            except ValueError as e: print(f"Error creating Butterworth filter: {e}."); print("Warning: Filtering disabled.")
        else: print(f"Warning: Invalid filter cutoff {self.filter_cutoff} Hz for sample rate {self.sampling_rate} Hz. Filtering disabled.")

    def process_signal(self, motion_value):
        """Adds a new motion value to the buffer."""
        if np.isfinite(motion_value): self.buffer.append(motion_value)
        else: print(f"Warning: Received non-finite motion value ({motion_value}). Skipping.")

    def _fit_gaussian_to_peak(self, signal, peak_index, estimated_width):
        """Fits a Gaussian function to a window around the peak."""
        window_half_width = int(max(5, estimated_width * self.config.get('GAUSSIAN_FIT_WINDOW_FACTOR', 3) / 2)); start = max(0, peak_index - window_half_width); end = min(len(signal), peak_index + window_half_width + 1)
        window_x = np.arange(start, end); window_y = signal[start:end]
        if len(window_x) < 3: return None
        peak_val = signal[peak_index]; baseline = np.min(window_y); amp_guess = peak_val - baseline; mean_guess = peak_index
        if estimated_width > 0: stddev_guess = estimated_width / 2.355
        else: stddev_guess = 3.0
        if amp_guess <= 0: amp_guess = 1e-6
        stddev_guess = max(0.5, stddev_guess); initial_guess = [amp_guess, mean_guess, stddev_guess]; bounds = ([0, start - 1, 0.1], [np.inf, end + 1, np.inf])
        try:
            params, covariance = curve_fit(gaussian, window_x, window_y - baseline, p0=initial_guess, bounds=bounds, maxfev=5000)
            amplitude, mean, stddev = params
            min_std = self.config.get('GAUSSIAN_FIT_MIN_STDDEV', 1.0)
            max_std = self.config.get('GAUSSIAN_FIT_MAX_STDDEV', 50.0)
            if min_std <= abs(stddev) <= max_std: return mean
            else: return None
        except RuntimeError: return None
        except ValueError as e: print(f"ValueError during Gaussian fit: {e}"); return None
        except Exception as e: print(f"Unexpected error during Gaussian fit: {e}"); return None

    def analyze_buffer(self):
        """Analyzes the current signal buffer to find peaks and BPM."""
        bpm = 0.0; peaks = np.array([]); filtered_signal = None; normalized_signal_value = 0.0
        if len(self.buffer) < self.min_buffer_fill:
            if self.last_valid_filtered_signal is not None and len(self.last_valid_filtered_signal) > 0:
                 last_signal = self.last_valid_filtered_signal; sig_mean = np.mean(last_signal); sig_std = np.std(last_signal)
                 if sig_std > 1e-6: normalized_signal_value = np.clip((last_signal[-1] - sig_mean) / (3 * sig_std), -2, 2)
            return filtered_signal, bpm, peaks, normalized_signal_value
        signal_arr = np.array(self.buffer)
        if not np.all(np.isfinite(signal_arr)):
            print("Warning: Non-finite values in buffer PRE-filter."); finite_signal = signal_arr[np.isfinite(signal_arr)]
            if len(finite_signal) < self.min_buffer_fill: return None, 0.0, np.array([]), 0.0
            signal_arr = finite_signal
        max_abs_raw = np.max(np.abs(signal_arr)) if len(signal_arr) > 0 else 0
        if self.debug_prints and max_abs_raw > 1e6: print(f"DEBUG: Max abs raw signal: {max_abs_raw:.2e}")
        if self.filter_sos is not None:
            try:
                filtered_signal = sp_signal.sosfiltfilt(self.filter_sos, signal_arr)
                if not np.all(np.isfinite(filtered_signal)): print("Warning: Non-finite values POST-filter."); filtered_signal = signal_arr
                else:
                    max_abs_filtered = np.max(np.abs(filtered_signal))
                    if self.debug_prints and max_abs_filtered > 1e6: print(f"DEBUG: Max abs signal POST-filter: {max_abs_filtered:.2e}")
                    if max_abs_filtered > 1e10: print(f"Warning: Extremely large values POST-filter."); filtered_signal = signal_arr
            except Exception as e: print(f"Error applying filter: {e}."); filtered_signal = signal_arr
        else: filtered_signal = signal_arr
        if filtered_signal is None or not np.all(np.isfinite(filtered_signal)):
             print("Warning: Signal invalid after filter fallback.");
             if self.last_valid_filtered_signal is not None and len(self.last_valid_filtered_signal) > 0:
                 last_signal = self.last_valid_filtered_signal; sig_mean = np.mean(last_signal); sig_std = np.std(last_signal)
                 if sig_std > 1e-6: normalized_signal_value = np.clip((last_signal[-1] - sig_mean) / (3 * sig_std), -2, 2)
             return None, 0.0, np.array([]), normalized_signal_value
        self.last_valid_filtered_signal = filtered_signal.copy()
        sig_mean = np.mean(filtered_signal); sig_std = np.std(filtered_signal)
        if sig_std > 1e-6: normalized_signal_value = np.clip((filtered_signal[-1] - sig_mean) / (3 * sig_std), -2, 2)
        try:
            max_freq_hz = self.config.get('MAX_EXPECTED_BPM', 120) / 60.0; min_dist_samples = int((self.sampling_rate / max_freq_hz) * self.config.get('PEAK_FINDING_DISTANCE_FACTOR', 0.5)); min_dist_samples = max(1, min_dist_samples)
            signal_range = np.ptp(filtered_signal); signal_range = max(signal_range, 1e-9); prominence_val = max(1e-6, signal_range * self.config.get('PEAK_FINDING_PROMINENCE_FACTOR', 0.1))
            peaks, properties = sp_signal.find_peaks(filtered_signal, distance=min_dist_samples, prominence=prominence_val, width=(None, None))
        except ValueError as e_peaks: print(f"Error during peak finding (ValueError): {e_peaks}"); return filtered_signal, bpm, peaks, normalized_signal_value
        except Exception as e_peaks_other: print(f"Error during peak finding (Exception): {e_peaks_other}"); return filtered_signal, bpm, peaks, normalized_signal_value
        if len(peaks) < 2: return filtered_signal, bpm, peaks, normalized_signal_value
        if self.use_gaussian_fit:
            valid_peaks_indices = []; peak_widths = properties.get('widths')
            if peak_widths is None: avg_interval = np.mean(np.diff(peaks)) if len(peaks) >=2 else min_dist_samples; estimated_width = max(3.0, avg_interval / 4.0); peak_widths = [estimated_width] * len(peaks)
            else: peak_widths = [max(1.0, w) for w in peak_widths]
            for i, p in enumerate(peaks):
                width_for_peak = peak_widths[i]; refined_pos = self._fit_gaussian_to_peak(filtered_signal, p, width_for_peak)
                if refined_pos is not None: valid_peaks_indices.append(p)
            peaks = np.array(valid_peaks_indices)
            if len(peaks) < 2: return filtered_signal, bpm, peaks, normalized_signal_value
        if len(peaks) >= 2:
            avg_interval_samples = np.mean(np.diff(peaks))
            if avg_interval_samples > 0: bpm = (self.sampling_rate / avg_interval_samples) * 60; self.last_bpms.append(bpm)
        return filtered_signal, bpm, peaks, normalized_signal_value
    def get_bpm_stability(self):
        """Calculates the standard deviation of the last few BPM values."""
        if len(self.last_bpms) < 5: return 0.0
        return np.std(self.last_bpms)

# --- Module: AdaptiveController ---
class AdaptiveController:
    """Dynamically adjusts parameters based on performance and quality."""
    def __init__(self, config):
        self.config = config; self.frames_since_last_switch = 0; self.current_method = config.get('DEFAULT_MOTION_METHOD', 'ROI_Average'); self.hysteresis_frames = config.get('ADAPTIVE_HYSTERESIS_FRAMES', 60); print("AdaptiveController initialized.")
    def monitor(self, fps, bpm_stability, tracked_points):
        self.current_fps = fps; self.current_bpm_stability = bpm_stability; self.current_tracked_points = tracked_points; self.frames_since_last_switch += 1
    def adapt(self, motion_detector):
        if not self.config.get('USE_ADAPTIVE_CONTROL', False) or self.frames_since_last_switch < self.hysteresis_frames: return
        fps_threshold = self.config.get('ADAPTIVE_FPS_THRESHOLD', 15); stability_threshold = self.config.get('ADAPTIVE_BPM_STABILITY_THRESHOLD', 15); switched = False
        if self.current_method == 'OpticalFlow':
            if self.current_fps < fps_threshold: print(f"Adaptive Switch: FPS low. Switching to ROI_Average."); motion_detector.set_method('ROI_Average'); self.current_method = 'ROI_Average'; switched = True
        elif self.current_method == 'ROI_Average':
            if self.current_bpm_stability > 0 and self.current_bpm_stability > stability_threshold and self.current_fps > fps_threshold * 1.2: print(f"Adaptive Switch: BPM unstable, FPS OK. Switching to OpticalFlow."); motion_detector.set_method('OpticalFlow'); self.current_method = 'OpticalFlow'; switched = True
        if switched: self.frames_since_last_switch = 0

# --- Main Monitor Class ---
class RespirationMonitor:
    """Encapsulates the core respiration monitoring pipeline."""
    def __init__(self, config_filepath=None):
        """Initializes the monitor, loading config from the specified filepath."""
        if config_filepath: self.config = load_config(config_filepath)
        else: print("Warning: No config filepath provided. Using minimal defaults."); self.config = MINIMAL_DEFAULTS.copy()
        self.video_input = None; self.calibration = None; self.motion_detector = None; self.signal_processor = None; self.adaptive_controller = None
        self.roi = None; self.is_initialized = False; self.last_frame = None

    def initialize(self, manual_roi=None):
        """Initializes all components of the monitor. Skips calibration if manual_roi is provided."""
        print("Initializing Respiration Monitor...")
        self.video_input = VideoInput(self.config)
        if not self.video_input.initialize(): return False
        if manual_roi is not None and len(manual_roi) == 4:
            x, y, w, h = manual_roi
            if w > 0 and h > 0: print(f"Using manually provided ROI: {manual_roi}. Skipping automatic calibration."); self.roi = manual_roi
            else: print("Warning: Manual ROI has invalid dimensions. Falling back to calibration."); manual_roi = None
        else: manual_roi = None
        if self.roi is None:
            self.calibration = Calibration(self.video_input, self.config)
            print("Running automatic calibration..."); self.roi = self.calibration.run_calibration()
            if self.roi is None: print("Monitor initialization failed: Calibration unsuccessful."); self.video_input.release(); return False
        self.motion_detector = MotionDetector(self.roi, self.config)
        self.signal_processor = SignalProcessor(self.video_input.get_fps(), self.config)
        if self.config.get('USE_ADAPTIVE_CONTROL', False):
            self.adaptive_controller = AdaptiveController(self.config)
            self.adaptive_controller.current_method = self.motion_detector.method
        self.is_initialized = True; print("Respiration Monitor initialized successfully."); return True

    def run_cycle(self):
        """Runs one cycle of frame capture and processing."""
        if not self.is_initialized or self.video_input is None: return {'success': False, 'error': 'Not initialized'}
        success, frame = self.video_input.get_frame()
        if not success: return {'success': False, 'error': 'Video ended or failed'}
        try: self.last_frame = frame.get() if isinstance(frame, cv2.UMat) else frame
        except Exception as e: print(f"Warning: Could not get frame data for storage: {e}"); self.last_frame = None
        motion_value = self.motion_detector.process_frame(frame)
        self.signal_processor.process_signal(motion_value)
        filtered_signal, bpm, peaks, normalized_signal_value = self.signal_processor.analyze_buffer()
        if self.adaptive_controller:
             bpm_stability = self.signal_processor.get_bpm_stability()
             tracked_points = self.motion_detector.get_tracked_points_count()
             self.adaptive_controller.monitor(0, bpm_stability, tracked_points) # FPS set externally
        results = { 'success': True, 'bpm': bpm, 'normalized_signal': normalized_signal_value if normalized_signal_value is not None else 0.0, 'method': self.motion_detector.method, 'frame': self.last_frame, 'roi': self.roi, }
        return results

    def trigger_adaptation(self, fps):
         """Externally trigger the adaptive controller logic."""
         if self.adaptive_controller and self.motion_detector:
              self.adaptive_controller.current_fps = fps
              self.adaptive_controller.adapt(self.motion_detector)

    def get_current_method(self): return self.motion_detector.method if self.motion_detector else "N/A"
    def stop(self): 
        print("Stopping Respiration Monitor...")
        if self.video_input: 
            self.video_input.release()
            self.is_initialized = False

# (No test block needed here)
