# src/motion_detection.py
# Contains the MotionDetector class for ROI Average and Optical Flow methods.

import cv2
import numpy as np
import traceback # For detailed error printing

# Note: Uses config dictionary passed during initialization.

class MotionDetector:
    """Calculates motion signal using ROI Average or Optical Flow."""
    def __init__(self, roi, config):
        """Initializes MotionDetector.

        Args:
            roi (tuple): The (x, y, w, h) region of interest.
            config (dict): The configuration dictionary.
        """
        if roi is None or len(roi) != 4:
            raise ValueError("Invalid ROI provided to MotionDetector")
        self.roi = roi
        self.config = config
        self.method = config.get('DEFAULT_MOTION_METHOD', 'ROI_Average')
        self.prev_frame_gray = None
        self.prev_points = None # For Optical Flow
        self.tracked_points_count = 0 # For adaptive control metric

        # Safely access nested optical flow params with defaults from MINIMAL_DEFAULTS
        # (Assuming MINIMAL_DEFAULTS structure is known or passed implicitly via config merge)
        default_of_params = {
            'feature_params': {'maxCorners': 100, 'qualityLevel': 0.3, 'minDistance': 7, 'blockSize': 7},
            'lk_params': {'winSize': [15, 15], 'maxLevel': 2, 'criteria': [3, 10, 0.03]}
        }
        of_params = config.get('OPTICAL_FLOW_PARAMS', default_of_params)
        default_feature_params = default_of_params.get('feature_params', {})
        default_lk_params = default_of_params.get('lk_params', {})

        self.feature_params = {**default_feature_params, **of_params.get('feature_params', {})}
        self.lk_params = {**default_lk_params, **of_params.get('lk_params', {})}

        # Ensure criteria is correctly formatted (handle list from JSON)
        criteria_val = self.lk_params.get('criteria')
        # OpenCV criteria tuple elements: type, max_iter, epsilon
        # Type flags: cv2.TERM_CRITERIA_EPS=2, cv2.TERM_CRITERIA_COUNT=1
        # Default type is usually EPS | COUNT = 3
        default_criteria_tuple = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        if isinstance(criteria_val, list) and len(criteria_val) == 3:
             try:
                  # Combine flags correctly for OpenCV criteria tuple
                  type_flags = int(criteria_val[0]) # Should be 1, 2, or 3 from JSON
                  max_iter = int(criteria_val[1])
                  epsilon = float(criteria_val[2])
                  self.lk_params['criteria'] = (type_flags, max_iter, epsilon)
             except (ValueError, TypeError):
                  print(f"Warning: Invalid format for lk_params 'criteria': {criteria_val}. Using default.")
                  self.lk_params['criteria'] = default_criteria_tuple
        elif not isinstance(criteria_val, tuple): # If not list or tuple, use default
             # print(f"Warning: lk_params 'criteria' not a tuple or valid list: {criteria_val}. Using default.")
             self.lk_params['criteria'] = default_criteria_tuple

        # Ensure winSize is a tuple
        if isinstance(self.lk_params.get('winSize'), list):
             try: self.lk_params['winSize'] = tuple(self.lk_params['winSize'])
             except TypeError: self.lk_params['winSize'] = (15, 15) # Default on error
        elif not isinstance(self.lk_params.get('winSize'), tuple): # Use default if not list or tuple
             self.lk_params['winSize'] = (15, 15)

        # Get ROI Average specific parameters
        self.roi_prefilter = config.get('ROI_AVG_PREFILTER')
        self.gaussian_kernel = tuple(config.get('ROI_AVG_GAUSSIAN_KERNEL', (3,3)))
        self.median_ksize = config.get('ROI_AVG_MEDIAN_KSIZE', 5)
        # Get OpenCL status (should be set by VideoInput based on config)
        self.use_opencl = config.get('USE_OPENCL', False) and cv2.ocl.haveOpenCL()

        print(f"MotionDetector initialized with method: {self.method}, ROI: {self.roi}")

    def set_method(self, method_name):
        """Sets the motion detection method."""
        if method_name in ['ROI_Average', 'OpticalFlow']:
            if self.method != method_name:
                print(f"Switching motion detection method to: {method_name}")
                self.method = method_name
                # Reset state specific to optical flow when switching
                self.prev_frame_gray = None
                self.prev_points = None
                self.tracked_points_count = 0
        else:
            print(f"Warning: Invalid motion detection method '{method_name}'. Keeping '{self.method}'.")

    def get_tracked_points_count(self):
        """Returns the number of points successfully tracked in the last Optical Flow step."""
        return self.tracked_points_count if self.method == 'OpticalFlow' else 0

    def process_frame(self, frame):
        """Processes a single frame to extract a motion value."""
        x, y, w, h = self.roi
        motion_value = 0.0 # Default to float
        if w <= 0 or h <= 0:
            # print("Warning: Invalid ROI dimensions (<=0).") # Can be noisy
            return motion_value

        is_umat = isinstance(frame, cv2.UMat)
        try: # Wrap entire frame processing
            try: # ROI Slicing
                frame_h, frame_w = frame.shape[:2] if not is_umat else frame.size()[::-1] # UMat size() is (w,h)
                y_start = max(0, y); y_end = min(frame_h, y + h)
                x_start = max(0, x); x_end = min(frame_w, x + w)
                # Check if calculated slice is valid before slicing
                if y_start >= y_end or x_start >= x_end:
                    # print(f"Warning: Calculated ROI slice is empty or invalid. ROI: {self.roi}, Frame: {frame_w}x{frame_h}")
                    return motion_value
                roi_frame = frame[y_start:y_end, x_start:x_end]
            except (cv2.error, Exception) as e_slice: # Fallback slicing
                 print(f"Error slicing ROI: {e_slice}. Trying fallback.");
                 # Ensure frame is NumPy for fallback slicing
                 frame_np = frame.get() if is_umat else frame
                 if frame_np is None: return motion_value # Check if frame is valid
                 frame_h, frame_w = frame_np.shape[:2]
                 y_start = max(0, y); y_end = min(frame_h, y + h)
                 x_start = max(0, x); x_end = min(frame_w, x + w)
                 if y_start >= y_end or x_start >= x_end: return motion_value
                 roi_frame = frame_np[y_start:y_end, x_start:x_end]
                 is_umat = False # Ensure is_umat is updated on fallback

            # Check if roi_frame is valid after slicing
            if roi_frame is None or roi_frame.shape[0] == 0 or roi_frame.shape[1] == 0:
                # print("Warning: ROI frame is empty after slicing.")
                return motion_value

            # --- Method Dispatch ---
            self.tracked_points_count = 0 # Reset count for this frame
            if self.method == 'ROI_Average':
                motion_value = self._detect_roi_average(roi_frame)
            elif self.method == 'OpticalFlow':
                try: # Grayscale conversion
                    # Use the full frame for grayscale conversion, not just ROI
                    if len(frame.shape) == 3 or (is_umat and len(frame.size()) == 3):
                         if is_umat:
                              try: frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                              except cv2.error: frame_gray = cv2.cvtColor(frame.get(), cv2.COLOR_BGR2GRAY) # Fallback
                         else: frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else: frame_gray = frame # Assume already grayscale
                except cv2.error as e_cvt:
                     print(f"Error converting frame to grayscale: {e_cvt}")
                     return motion_value # Cannot proceed with OF if conversion fails

                motion_value = self._detect_optical_flow(frame_gray)
                # self.prev_frame_gray is updated inside _detect_optical_flow now
            return float(motion_value) # Ensure float output

        except cv2.error as e_cv: # Catch OpenCV errors during processing
             print(f"OpenCV error during motion detection ({self.method}): {e_cv}")
             return 0.0 # Return default on error
        except Exception as e: # Catch other unexpected errors
             print(f"Unexpected error during motion detection ({self.method}): {e}")
             traceback.print_exc() # Print stack trace for debugging
             return 0.0 # Return default on error

    def _detect_roi_average(self, roi_frame):
        """Calculates motion based on average intensity in ROI."""
        filtered_roi = roi_frame # Start with original ROI frame
        try: # Pre-processing (Filtering)
            # Convert to gray FIRST for efficiency if needed
            if len(roi_frame.shape) == 3 or (isinstance(roi_frame, cv2.UMat) and len(roi_frame.size()) == 3):
                 gray_roi_unfiltered = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            else:
                 gray_roi_unfiltered = roi_frame # Assume already grayscale

            filtered_roi = gray_roi_unfiltered # Use the grayscale version for filtering/mean
            if self.roi_prefilter == 'Median':
                filtered_roi = cv2.medianBlur(gray_roi_unfiltered, self.median_ksize)
            elif self.roi_prefilter == 'Gaussian':
                # Ensure kernel is tuple
                kernel = self.gaussian_kernel if isinstance(self.gaussian_kernel, tuple) else (3,3)
                filtered_roi = cv2.GaussianBlur(gray_roi_unfiltered, kernel, 0)
        except cv2.error as e_filter:
            print(f"Warning: ROI pre-processing failed (OpenCV Error): {e_filter}.")
            # Fallback: ensure filtered_roi is the grayscale version if filtering failed
            if len(roi_frame.shape) == 3 or (isinstance(roi_frame, cv2.UMat) and len(roi_frame.size()) == 3):
                 try: filtered_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                 except cv2.error: return 0.0 # Give up if grayscale fails too
            else: filtered_roi = roi_frame
        except Exception as e_filter_other:
            print(f"Warning: ROI pre-processing failed (Other Error): {e_filter_other}.")
            # Fallback as above
            if len(roi_frame.shape) == 3 or (isinstance(roi_frame, cv2.UMat) and len(roi_frame.size()) == 3):
                 try: filtered_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
                 except Exception: return 0.0
            else: filtered_roi = roi_frame

        try: # Mean calculation
            # Ensure input to mean is single channel
            if filtered_roi is None or len(filtered_roi.shape) > 2 :
                 print("Warning: Invalid input for cv2.mean in ROI average.")
                 return 0.0

            if isinstance(filtered_roi, cv2.UMat):
                mean_val = cv2.mean(filtered_roi.get())[0] # Safer fallback
            else:
                mean_val = cv2.mean(filtered_roi)[0]
            return mean_val
        except cv2.error as e_mean_cv: print(f"Error calculating mean in ROI average (OpenCV Error): {e_mean_cv}"); return 0.0
        except Exception as e_mean: print(f"Error calculating mean in ROI average (Exception): {e_mean}"); return 0.0

    def _detect_optical_flow(self, frame_gray):
        """Calculates motion using Lucas-Kanade Optical Flow."""
        current_frame_gray_internal = frame_gray; motion_signal = 0.0
        is_umat = isinstance(current_frame_gray_internal, cv2.UMat);
        # Handle potential error if frame_gray is None
        if current_frame_gray_internal is None:
             print("Error: _detect_optical_flow received None frame.")
             return motion_signal
        frame_h, frame_w = current_frame_gray_internal.shape[:2] if not is_umat else current_frame_gray_internal.size()[::-1]

        # Create mask for ROI
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8); x, y, w, h = self.roi; y_start, y_end = max(0, y), min(frame_h, y + h); x_start, x_end = max(0, x), min(frame_w, x + w)
        if y_start < y_end and x_start < x_end: mask[y_start:y_end, x_start:x_end] = 255

        # --- Feature Detection/Update ---
        detect_features = False
        if self.prev_points is None or len(self.prev_points) == 0:
            detect_features = True
            # Use previous frame for detection if available, otherwise current
            detection_frame = self.prev_frame_gray if self.prev_frame_gray is not None else current_frame_gray_internal

        if detect_features:
            self.prev_points = None; self.tracked_points_count = 0 # Reset state
            if detection_frame is not None:
                try:
                    det_h, det_w = detection_frame.shape[:2] if not isinstance(detection_frame, cv2.UMat) else detection_frame.size()[::-1]
                    # Recreate mask if dimensions changed (e.g., first frame vs subsequent)
                    if mask.shape[0] != det_h or mask.shape[1] != det_w:
                        mask = np.zeros((det_h, det_w), dtype=np.uint8)
                        if y_start < y_end and x_start < x_end: mask[y_start:y_end, x_start:x_end] = 255
                    # Detect features
                    self.prev_points = cv2.goodFeaturesToTrack(detection_frame, mask=mask, **self.feature_params)
                    self.tracked_points_count = len(self.prev_points) if self.prev_points is not None else 0
                except cv2.error as e: print(f"Error in goodFeaturesToTrack detect (OpenCV Error): {e}")
                except Exception as e: print(f"Error in goodFeaturesToTrack detect (Exception): {e}")
            else: print("Warning: Cannot detect features, no valid frame available.")

        # --- Feature Tracking ---
        tracking_successful = False
        # Proceed only if we have a previous frame and features to track
        if self.prev_frame_gray is not None and self.prev_points is not None and len(self.prev_points) > 0:
            try:
                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_frame_gray, current_frame_gray_internal, self.prev_points, None, **self.lk_params
                )
            except cv2.error as e: print(f"Error in calcOpticalFlowPyrLK (OpenCV Error): {e}"); next_points, status = None, None
            except Exception as e: print(f"Error in calcOpticalFlowPyrLK (Exception): {e}"); next_points, status = None, None

            if next_points is not None and status is not None:
                # Filter points based on status
                good_new = next_points[status.flatten() == 1]
                good_old = self.prev_points[status.flatten() == 1]
                self.tracked_points_count = len(good_new) # Update tracked count

                if self.tracked_points_count >= 4: # Need minimum points for PCA/fallback
                    try: # PCA Calculation
                        displacements = good_new - good_old
                        # Reshape for PCACompute: samples per row -> (M, 2)
                        disp_np = np.float32(displacements.reshape(-1, 2))
                        if disp_np.shape[0] >= 2: # Ensure at least 2 samples for PCA
                            mean, eigenvectors = cv2.PCACompute(disp_np, mean=None, maxComponents=1)
                            mean_displacement = np.mean(disp_np, axis=0)
                            motion_signal = np.dot(mean_displacement, eigenvectors[0].T) # Project mean onto first eigenvector
                            # Update points for next iteration ONLY if PCA was successful
                            self.prev_points = good_new.reshape(-1, 1, 2)
                            tracking_successful = True # Mark tracking as successful
                        else:
                             print("Warning: Not enough valid displacements for PCA after reshape.")
                             motion_signal = 0.0 # Not enough data for PCA
                    except (cv2.error, Exception) as e_pca:
                        print(f"Error during PCA/Eigenvector signal extraction: {e_pca}")
                        # Fallback calculation
                        try:
                            # Reshape good_new/good_old to (M, 2) before accessing index 1
                            good_new_flat = good_new.reshape(-1, 2)
                            good_old_flat = good_old.reshape(-1, 2)
                            # Check shape after reshape just in case
                            if good_new_flat.shape[1] == 2 and good_old_flat.shape[1] == 2:
                                y_disp = good_new_flat[:, 1] - good_old_flat[:, 1]
                                motion_signal = np.mean(y_disp) if len(y_disp) > 0 else 0.0
                            else:
                                print("Warning: Fallback y_disp calculation failed due to unexpected shape.")
                                motion_signal = 0.0
                        except IndexError as e_index:
                             print(f"Fallback y_disp calculation failed (IndexError): {e_index}")
                             motion_signal = 0.0
                        # Don't mark tracking as successful if PCA failed, force re-detection
                        tracking_successful = False # Ensure re-detection
                # else: # Not enough points tracked, handled below
                #     pass
            # else: # Tracking failed (next_points or status is None), handled below
            #     pass

        # Reset points if tracking wasn't successful or not enough points
        if not tracking_successful:
             self.prev_points = None
             self.tracked_points_count = 0
             # motion_signal remains 0.0 if tracking/PCA fails

        # Always update the previous frame for the *next* iteration's tracking or detection
        self.prev_frame_gray = current_frame_gray_internal
        return motion_signal

