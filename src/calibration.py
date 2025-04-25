# src/calibration.py
# Contains the Calibration class for automatic ROI detection.

import cv2
import numpy as np
from scipy import signal as sp_signal
import time
import traceback # For error printing
import os

# Note: Assumes VideoInput class provides necessary methods like get_frame(), get_fps(), get_frame_size()
# Note: Uses config dictionary passed during initialization.

class Calibration:
    """Handles ROI selection based on motion analysis."""
    def __init__(self, video_input, config):
        """Initializes Calibration.

        Args:
            video_input (VideoInput): An initialized VideoInput object.
            config (dict): The configuration dictionary.
        """
        self.video_input = video_input
        self.config = config
        # Determine OpenCL usage based on video_input's status after initialization
        self.use_opencl = getattr(video_input, 'use_opencl', False)
        self.debug_viz = config.get('DEBUG_CALIBRATION_VIZ', False) # Get debug flag

    def run_calibration(self):
        """Runs the calibration process to find the ROI."""
        method = self.config.get('CALIBRATION_METHOD', 'MotionVariance')
        print(f"Starting calibration using '{method}' method...")
        frames_for_calib_gray = []
        frames_for_calib_color = []
        start_time = time.time()
        calibration_duration = self.config.get('CALIBRATION_DURATION_FRAMES', 150)
        try: # Wrap frame collection loop
            for i in range(calibration_duration):
                success, frame = self.video_input.get_frame()
                if not success:
                    print("Warning: Video source ended during calibration.")
                    break
                # Store both color (NumPy) and gray (UMat or NumPy)
                frame_np = frame.get() if isinstance(frame, cv2.UMat) else frame
                frames_for_calib_color.append(frame_np)
                if isinstance(frame, cv2.UMat):
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames_for_calib_gray.append(gray_frame)

                # Optional visualization during calibration
                if self.debug_viz and i % 15 == 0:
                     viz_frame = frame_np.copy()
                     cv2.putText(viz_frame, f"Calibrating... Frame {i+1}/{calibration_duration}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                     cv2.imshow("Calibration Progress", viz_frame)
                     cv2.waitKey(1)
        except Exception as e:
             print(f"Error during frame collection for calibration: {e}")
             if self.debug_viz: cv2.destroyWindow("Calibration Progress") # Ensure cleanup on error
             return None # Cannot proceed if frame collection fails
        finally:
             # Ensure calibration progress window is closed
             if self.debug_viz:
                 cv2.destroyWindow("Calibration Progress")
                 cv2.waitKey(1) # Give time for window to close

        # Check if enough frames were collected
        if len(frames_for_calib_gray) < 20: # Need a reasonable number for analysis
            print("Error: Not enough frames collected for calibration.")
            return None
        print(f"Collected {len(frames_for_calib_gray)} frames for calibration in {time.time() - start_time:.2f} seconds.")

        try: # Wrap the main calibration algorithm execution
            # Select and run the chosen calibration method
            if method == 'SimplifiedEVM':
                roi = self._perform_simplified_evm(frames_for_calib_gray)
            elif method == 'MotionVariance':
                roi = self._find_roi_by_motion_variance(frames_for_calib_gray)
            else:
                print(f"Error: Unknown calibration method '{method}'.")
                return None

            # Optional ROI refinement step
            if roi is not None and self.config.get('ROI_REFINE_SIZE', False):
                 print("Performing ROI Size Refinement...")
                 refine_frames_count = min(len(frames_for_calib_color), int(self.video_input.get_fps() * 3))
                 if refine_frames_count > 10:
                     refined_roi = self._refine_roi_size(roi, frames_for_calib_color[:refine_frames_count])
                     # Only update if refinement was successful and returned a valid ROI
                     if refined_roi:
                          print(f"ROI refined from {roi} to {refined_roi}")
                          roi = refined_roi
                     else: print("ROI refinement failed or returned invalid ROI, keeping original.")
                 else: print("Not enough frames for ROI refinement.")

            # Final check and return
            if roi is None: print("Calibration failed to find ROI.")
            else: print(f"Calibration complete. Selected ROI: {roi}")
            return roi
        except cv2.error as e_cv: # Catch OpenCV errors during calibration processing
             print(f"OpenCV error during calibration ({method}): {e_cv}")
             return None
        except Exception as e: # Catch other unexpected errors
             print(f"Unexpected error during calibration ({method}): {e}")
             traceback.print_exc() # Print full traceback for debugging
             return None
        finally:
            # Clean up any debug visualization windows
            if self.debug_viz:
                 cv2.destroyWindow("Variance Map (Blurred - EVM)")
                 cv2.destroyWindow("Variance Map (Raw - EVM)")
                 cv2.destroyWindow("Variance Map (Blurred - MotionVar)")
                 cv2.destroyWindow("Variance Map (Raw - MotionVar)")
                 cv2.waitKey(1)

    def _build_gaussian_pyramid(self, frame, levels):
        """Builds a Gaussian pyramid using OpenCV."""
        pyramid = [frame]
        current_level = frame
        for i in range(levels): # Build 'levels' down pyramid steps
            try:
                if isinstance(current_level, cv2.UMat):
                     try: next_level = cv2.pyrDown(current_level)
                     except cv2.error: next_level = cv2.pyrDown(current_level.get()) # Fallback
                else: next_level = cv2.pyrDown(current_level)
                pyramid.append(next_level)
                current_level = next_level
            except cv2.error as e:
                 print(f"Error in pyrDown at level {i}: {e}")
                 raise # Re-raise error to be caught by caller
        return pyramid

    def _temporal_filter(self, data_sequence, fs, low_hz, high_hz):
        """Applies a temporal bandpass filter using SciPy."""
        if data_sequence.shape[0] < 10: return None # Need minimum sequence length
        try:
            # Design 4th order Butterworth bandpass filter
            filter_order = 4 # Could make this configurable
            sos = sp_signal.butter(filter_order, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos')
            # Apply zero-phase filter along time axis (axis=0)
            filtered_sequence = sp_signal.sosfiltfilt(sos, data_sequence, axis=0)
            return filtered_sequence
        except ValueError as e: # Catch specific errors like invalid frequency ranges
            print(f"Error designing/applying temporal filter (ValueError): {e}.")
            return None
        except Exception as e: # Catch other filtering errors
            print(f"Error during temporal filtering (Exception): {e}")
            return None

    def _visualize_variance_map(self, variance_map, window_title="Variance Map"):
         """Helper function to display variance maps for debugging."""
         if variance_map is None or variance_map.size == 0: return
         try:
             # Normalize map to 0-255 for display
             norm_map = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
             # Apply a colormap for better contrast
             color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
             cv2.imshow(window_title, color_map)
         except cv2.error as e:
              print(f"Error visualizing variance map '{window_title}': {e}")
         except Exception as e:
              print(f"Unexpected error visualizing variance map '{window_title}': {e}")

    def _perform_simplified_evm(self, gray_frames):
        """Simplified EVM approach: Temporal filtering of a specific pyramid level."""
        target_level = self.config.get('EVM_PYRAMID_LEVEL', 2)
        fs = self.video_input.get_fps()
        low_hz = self.config.get('EVM_LOW_FREQ_HZ', 0.4)
        high_hz = self.config.get('EVM_HIGH_FREQ_HZ', 2.0)

        # Validate frequency parameters against Nyquist limit
        if fs <= 0 or high_hz >= fs / 2.0:
            print(f"Error: Invalid frequencies for temporal filter. Fs={fs}, HighHz={high_hz}. Nyquist limit={fs/2.0}. Falling back.")
            return self._find_roi_by_motion_variance(gray_frames) # Fallback to simpler method

        # Build pyramids and extract target level
        pyramids = []
        for frame in gray_frames:
            pyramid = self._build_gaussian_pyramid(frame, target_level) # Can raise cv2.error
            if len(pyramid) > target_level: pyramids.append(pyramid[target_level])
            else: print(f"Warning: Could not build pyramid to level {target_level}. Falling back."); return self._find_roi_by_motion_variance(gray_frames)
        if not pyramids: return None

        # Ensure consistent shapes and convert to NumPy array
        target_shape = pyramids[0].shape
        pyramid_data_list = [p.get() if isinstance(p, cv2.UMat) else p for p in pyramids if p.shape == target_shape]
        if len(pyramid_data_list) < 10: print("Error: Not enough consistent pyramid levels."); return self._find_roi_by_motion_variance(gray_frames)
        pyramid_sequence = np.stack(pyramid_data_list, axis=0)

        # Apply temporal filter
        filtered_sequence = self._temporal_filter(pyramid_sequence, fs, low_hz, high_hz)
        if filtered_sequence is None: print("Temporal filtering failed. Falling back."); return self._find_roi_by_motion_variance(gray_frames)

        # Analyze variance and find max location
        variance_map = np.var(filtered_sequence, axis=0)
        variance_map_blurred = cv2.GaussianBlur(variance_map.astype(np.float32), (15, 15), 0) # GaussianBlur can raise cv2.error
        if self.debug_viz: self._visualize_variance_map(variance_map, "Variance Map (Raw - EVM)"); self._visualize_variance_map(variance_map_blurred, "Variance Map (Blurred - EVM)"); cv2.waitKey(1)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(variance_map_blurred) # minMaxLoc can raise cv2.error
        print(f"Simplified EVM - Max Variance: {maxVal:.4f} at {maxLoc} (Level {target_level})")
        if maxVal <= 1e-6: print("Warning: No significant variance found in filtered signal. Falling back."); return self._find_roi_by_motion_variance(gray_frames)

        # Map location back to original frame coordinates
        level_h, level_w = target_shape[:2]; orig_w, orig_h = self.video_input.get_frame_size(); scale_factor = 2**target_level
        roi_w_orig = max(20, orig_w // 4); roi_h_orig = max(20, orig_h // 4)
        center_x_orig = maxLoc[0] * scale_factor; center_y_orig = maxLoc[1] * scale_factor
        roi_x_orig = max(0, min(orig_w - roi_w_orig, center_x_orig - roi_w_orig // 2)); roi_y_orig = max(0, min(orig_h - roi_h_orig, center_y_orig - roi_h_orig // 2))
        return (int(roi_x_orig), int(roi_y_orig), int(roi_w_orig), int(roi_h_orig))

    def _find_roi_by_motion_variance(self, frames):
        """Finds ROI based on variance of frame differences."""
        if len(frames) < 5: return None
        frames_np = [(f.get() if isinstance(f, cv2.UMat) else f) for f in frames]; frame_h, frame_w = frames_np[0].shape[:2]
        diffs = [cv2.absdiff(frames_np[i], frames_np[i+1]) for i in range(len(frames_np)-1)] # absdiff can raise cv2.error
        if not diffs: return None
        diff_stack = np.stack(diffs, axis=0); variance_map = np.var(diff_stack, axis=0); variance_map_blurred = cv2.GaussianBlur(variance_map.astype(np.float32), (15, 15), 0) # GaussianBlur can raise cv2.error
        if self.debug_viz: self._visualize_variance_map(variance_map, "Variance Map (Raw - MotionVar)"); self._visualize_variance_map(variance_map_blurred, "Variance Map (Blurred - MotionVar)"); cv2.waitKey(1)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(variance_map_blurred) # minMaxLoc can raise cv2.error
        print(f"Motion Variance - Max Variance: {maxVal:.4f} at {maxLoc}")
        if maxVal <= 1e-6: print("Warning: No significant motion variance found."); roi_w = max(20, frame_w // 4); roi_h = max(20, frame_h // 4); roi_x = max(0, frame_w // 2 - roi_w // 2); roi_y = max(0, frame_h // 2 - roi_h // 2); return (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
        roi_w = max(20, frame_w // 4); roi_h = max(20, frame_h // 4); roi_x = max(0, min(frame_w - roi_w, maxLoc[0] - roi_w // 2)); roi_y = max(0, min(frame_h - roi_h, maxLoc[1] - roi_h // 2))
        return (int(roi_x), int(roi_y), int(roi_w), int(roi_h))

    def _calculate_roi_signal_variance(self, roi, frames_np):
        """Helper to calculate variance of ROI average signal for refinement."""
        x, y, w, h = roi; motion_signal = []; prefilter = self.config.get('ROI_AVG_PREFILTER'); median_ksize = self.config.get('ROI_AVG_MEDIAN_KSIZE', 5); gaussian_kernel = self.config.get('ROI_AVG_GAUSSIAN_KERNEL', (3,3))
        if w <= 0 or h <= 0: return 0
        for frame in frames_np:
            try: # Wrap per-frame processing
                if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0: continue
                actual_y_end = min(frame.shape[0], y + h); actual_x_end = min(frame.shape[1], x + w); actual_y = max(0, y); actual_x = max(0, x)
                if actual_y_end <= actual_y or actual_x_end <= actual_x: continue
                roi_frame = frame[actual_y:actual_y_end, actual_x:actual_x_end]
                if roi_frame.shape[0] == 0 or roi_frame.shape[1] == 0: continue
                filtered_roi = roi_frame
                try: # Inner try for filtering/mean
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

# Example usage (for testing this module directly)
if __name__ == '__main__':
    print("Testing Calibration module...")
    # Need mock VideoInput and config for testing
    class MockVideoInput:
        def __init__(self, fps=30, width=640, height=480):
            self._fps = fps; self._w = width; self._h = height; self.frame_count = 0; self.use_opencl = False
        def initialize(self): return True
        def get_frame(self):
            # Generate dummy noise frames
            frame = np.random.randint(0, 256, (self._h, self._w, 3), dtype=np.uint8)
            # Add a moving rectangle to simulate motion variance
            x = int(100 + 50 * np.sin(self.frame_count * 0.1))
            cv2.rectangle(frame, (x, 100), (x+50, 150), (255,255,255), -1)
            self.frame_count += 1
            return True, frame
        def get_fps(self): return self._fps
        def get_frame_size(self): return self._w, self._h
        def release(self): pass

    try:
        from config_loader import load_config
        script_dir = os.path.dirname(__file__)
        config_dir = os.path.join(script_dir, '..', 'configs')
        test_config_path = os.path.join(config_dir, 'config_default.json')
        test_config = load_config(test_config_path)
        test_config['DEBUG_CALIBRATION_VIZ'] = True # Enable viz for test
        test_config['CALIBRATION_DURATION_FRAMES'] = 50 # Shorter duration for test
    except ImportError:
        print("Could not import config_loader, using basic test config.")
        test_config = {'CALIBRATION_METHOD': 'MotionVariance', 'CALIBRATION_DURATION_FRAMES': 50, 'DEBUG_CALIBRATION_VIZ': True}

    mock_video = MockVideoInput()
    if mock_video.initialize():
        calibrator = Calibration(mock_video, test_config)
        selected_roi = calibrator.run_calibration()
        print(f"Test calibration finished. Selected ROI: {selected_roi}")
        # Keep viz windows open briefly
        if test_config['DEBUG_CALIBRATION_VIZ']:
             print("Close debug windows by pressing any key...")
             cv2.waitKey(0)
             cv2.destroyAllWindows()

