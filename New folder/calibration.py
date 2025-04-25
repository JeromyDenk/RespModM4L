# src/calibration.py
# Contains the Calibration class for automatic ROI detection.
# MODIFIED: To return variance map for weighted averaging.
# MODIFIED: Added detailed debug prints for troubleshooting calibration failures.

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
        print("[Calibration Init] Debug Viz Enabled:", self.debug_viz) # DEBUG

    def run_calibration(self):
        """
        Runs the calibration process to find the ROI and calculate a variance map
        for potential weighted averaging.

        Returns:
            tuple: (roi, variance_map_for_weighting)
                   roi (tuple): (x, y, w, h) of the selected ROI, or None if failed.
                   variance_map_for_weighting (np.ndarray or UMat): Full-resolution variance map
                                                                   based on frame differencing,
                                                                   or None if failed.
        """
        method = self.config.get('CALIBRATION_METHOD', 'MotionVariance')
        print(f"[Calibration] Starting calibration using '{method}' method...")
        frames_for_calib_gray = []
        frames_for_calib_color = []
        start_time = time.time()
        calibration_duration = self.config.get('CALIBRATION_DURATION_FRAMES', 150)
        print(f"[Calibration] Target duration: {calibration_duration} frames.") # DEBUG
        try: # Wrap frame collection loop
            for i in range(calibration_duration):
                success, frame = self.video_input.get_frame()
                if not success:
                    print("[Calibration] Warning: Video source ended during calibration.")
                    break
                # Store both color (NumPy) and gray (UMat or NumPy)
                frame_np = frame.get() if isinstance(frame, cv2.UMat) else frame
                if frame_np is None: # DEBUG Check
                     print(f"[Calibration] Warning: Frame {i} is None after get(). Skipping.")
                     continue
                frames_for_calib_color.append(frame_np)

                # Grayscale Conversion
                try:
                    if isinstance(frame, cv2.UMat):
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        # Ensure consistent type (NumPy array) for processing
                        gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
                    frames_for_calib_gray.append(gray_frame)
                except cv2.error as e_cvt: # DEBUG Check
                     print(f"[Calibration] Warning: cvtColor failed for frame {i}: {e_cvt}. Skipping frame.")
                     continue
                except Exception as e_cvt_other: # DEBUG Check
                     print(f"[Calibration] Warning: Error converting frame {i} to gray: {e_cvt_other}. Skipping frame.")
                     continue


                # Optional visualization during calibration
                if self.debug_viz and i % 15 == 0:
                    viz_frame = frame_np.copy()
                    cv2.putText(viz_frame, f"Calibrating... Frame {i+1}/{calibration_duration}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    cv2.imshow("Calibration Progress", viz_frame)
                    cv2.waitKey(1)
        except Exception as e:
            print(f"[Calibration] Error during frame collection for calibration: {e}")
            traceback.print_exc() # DEBUG
            if self.debug_viz: cv2.destroyWindow("Calibration Progress") # Ensure cleanup on error
            return None, None # Cannot proceed if frame collection fails
        finally:
            # Ensure calibration progress window is closed
            if self.debug_viz:
                # Use try-except for destroyWindow as it might fail if window never opened
                try: cv2.destroyWindow("Calibration Progress")
                except cv2.error: pass
                cv2.waitKey(1) # Give time for window to close

        # Check if enough frames were collected
        print(f"[Calibration] Collected {len(frames_for_calib_gray)} gray frames.") # DEBUG
        if len(frames_for_calib_gray) < 20: # Need a reasonable number for analysis
            print("[Calibration] Error: Not enough valid frames collected for calibration.")
            return None, None

        print(f"[Calibration] Frame collection took {time.time() - start_time:.2f} seconds.")

        # --- Calculate Motion Variance Map First ---
        variance_map_for_weighting = None
        variance_map_raw = None # DEBUG
        try:
            print("[Calibration] Calculating full-resolution motion variance map...")
            # Ensure frames are NumPy arrays for stacking
            frames_np = [(f.get() if isinstance(f, cv2.UMat) else f) for f in frames_for_calib_gray if f is not None] # Filter out potential Nones
            if len(frames_np) >= 5:
                frame_h, frame_w = frames_np[0].shape[:2]
                print(f"[Calibration] Variance map frame size: {frame_w}x{frame_h}") # DEBUG
                diffs = [cv2.absdiff(frames_np[i], frames_np[i+1]) for i in range(len(frames_np)-1)]
                if diffs:
                    print(f"[Calibration] Calculated {len(diffs)} frame differences.") # DEBUG
                    diff_stack = np.stack(diffs, axis=0)
                    variance_map_raw = np.var(diff_stack, axis=0)
                    # Use float32 for GaussianBlur
                    variance_map_for_weighting = cv2.GaussianBlur(variance_map_raw.astype(np.float32), (15, 15), 0)
                    print("[Calibration] Motion variance map calculated and blurred.")
                    # DEBUG Print min/max of raw variance
                    raw_min, raw_max, _, _ = cv2.minMaxLoc(variance_map_raw)
                    print(f"[Calibration] Raw Variance Map Min: {raw_min:.4f}, Max: {raw_max:.4f}")
                    if self.debug_viz:
                         self._visualize_variance_map(variance_map_raw, "Variance Map (Raw - MotionVar)")
                         self._visualize_variance_map(variance_map_for_weighting, "Variance Map (Blurred - MotionVar)")
                         cv2.waitKey(1)
                else:
                    print("[Calibration] Warning: Could not compute frame differences for variance map (diffs list empty).")
            else:
                print(f"[Calibration] Warning: Not enough valid numpy frames ({len(frames_np)}) to compute motion variance map.")

        except cv2.error as e_cv:
             print(f"[Calibration] OpenCV error during motion variance map calculation: {e_cv}")
             # Continue, but variance_map_for_weighting will be None
        except Exception as e:
             print(f"[Calibration] Unexpected error during motion variance map calculation: {e}")
             traceback.print_exc()
             # Continue, but variance_map_for_weighting will be None

        # --- Select ROI based on chosen method ---
        roi = None # Initialize ROI to None
        try: # Wrap the main calibration algorithm execution
            print(f"[Calibration] Selecting ROI using method: {method}") # DEBUG
            if method == 'SimplifiedEVM':
                roi = self._perform_simplified_evm(frames_for_calib_gray)
            elif method == 'MotionVariance':
                 roi = self._find_roi_by_motion_variance(frames_for_calib_gray, precomputed_variance_map=variance_map_for_weighting)
            else:
                print(f"[Calibration] Error: Unknown calibration method '{method}'.")
                # Return map even if ROI selection fails, maybe it's useful?
                return None, variance_map_for_weighting

            # --- ROI Refinement (Optional) ---
            if roi is not None and self.config.get('ROI_REFINE_SIZE', False):
                print("[Calibration] Performing ROI Size Refinement...")
                refine_frames_count = min(len(frames_for_calib_color), int(self.video_input.get_fps() * 3))
                if refine_frames_count > 10:
                    refined_roi = self._refine_roi_size(roi, frames_for_calib_color[:refine_frames_count])
                    if refined_roi:
                        print(f"[Calibration] ROI refined from {roi} to {refined_roi}")
                        roi = refined_roi
                    else: print("[Calibration] ROI refinement failed or returned invalid ROI, keeping original.")
                else: print("[Calibration] Not enough frames for ROI refinement.")
            elif roi is None:
                 print("[Calibration] Skipping ROI refinement because ROI selection failed.") # DEBUG

            # --- Final Check and Return ---
            if roi is None:
                print("[Calibration] Calibration process finished, but FAILED to find a valid ROI.") # DEBUG
            else:
                print(f"[Calibration] Calibration complete. Selected ROI: {roi}")

            # Return both ROI and the variance map
            return roi, variance_map_for_weighting

        except cv2.error as e_cv: # Catch OpenCV errors during calibration processing
            print(f"[Calibration] OpenCV error during ROI selection ({method}): {e_cv}")
            return None, variance_map_for_weighting # Return map even on error
        except Exception as e: # Catch other unexpected errors
            print(f"[Calibration] Unexpected error during ROI selection ({method}): {e}")
            traceback.print_exc() # Print full traceback for debugging
            return None, variance_map_for_weighting # Return map even on error
        finally:
            # Clean up any debug visualization windows
            if self.debug_viz:
                # Use try-except as windows might not exist if errors occurred earlier
                try: cv2.destroyWindow("Variance Map (Blurred - EVM)")
                except cv2.error: pass
                try: cv2.destroyWindow("Variance Map (Raw - EVM)")
                except cv2.error: pass
                try: cv2.destroyWindow("Variance Map (Blurred - MotionVar)")
                except cv2.error: pass
                try: cv2.destroyWindow("Variance Map (Raw - MotionVar)")
                except cv2.error: pass
                cv2.waitKey(1)

    def _build_gaussian_pyramid(self, frame, levels):
        """Builds a Gaussian pyramid using OpenCV."""
        # print(f"[Calibration] Building pyramid, levels={levels}") # DEBUG - Can be very noisy
        pyramid = [frame]
        current_level = frame
        for i in range(levels): # Build 'levels' down pyramid steps
            try:
                input_frame = current_level.get() if isinstance(current_level, cv2.UMat) else current_level
                if input_frame is None or input_frame.shape[0] < 2 or input_frame.shape[1] < 2:
                    print(f"[Calibration] Warning: Frame too small for pyrDown at level {i+1}. Shape: {input_frame.shape if input_frame is not None else 'None'}")
                    break
                h, w = input_frame.shape[:2]
                if w < 2 or h < 2:
                     print(f"[Calibration] Warning: Frame too small for pyrDown at level {i+1}. Shape: {(h, w)}")
                     break
                next_level = cv2.pyrDown(input_frame)
                pyramid.append(next_level)
                current_level = next_level
            except cv2.error as e:
                print(f"[Calibration] Error in pyrDown at level {i+1}: {e}. Input shape: {input_frame.shape if input_frame is not None else 'None'}")
                break # Stop building if pyrDown fails
        return pyramid

    def _temporal_filter(self, data_sequence, fs, low_hz, high_hz):
        """Applies a temporal bandpass filter using SciPy."""
        if data_sequence is None or data_sequence.shape[0] < 10:
             print("[Calibration] Warning: Not enough data for temporal filtering.")
             return None
        try:
            filter_order = 4
            nyquist = fs / 2.0
            if low_hz <= 0 or high_hz >= nyquist or low_hz >= high_hz:
                 print(f"[Calibration] Error: Invalid frequency range for filter. Fs={fs}, Low={low_hz}, High={high_hz}, Nyquist={nyquist}")
                 return None
            sos = sp_signal.butter(filter_order, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos')
            filtered_sequence = sp_signal.sosfiltfilt(sos, data_sequence, axis=0)
            return filtered_sequence
        except ValueError as e:
            print(f"[Calibration] Error designing/applying temporal filter (ValueError): {e}.")
            return None
        except Exception as e:
            print(f"[Calibration] Error during temporal filtering (Exception): {e}")
            traceback.print_exc()
            return None

    def _visualize_variance_map(self, variance_map, window_title="Variance Map"):
        """Helper function to display variance maps for debugging."""
        if not self.debug_viz: return # Skip if debug viz is off
        if variance_map is None or variance_map.size == 0:
             print(f"[Calibration] Cannot visualize '{window_title}', map is None or empty.") # DEBUG
             return
        try:
            map_to_show = variance_map.get() if isinstance(variance_map, cv2.UMat) else variance_map
            if map_to_show is None or map_to_show.size == 0: return
            if map_to_show.dtype != np.float32 and map_to_show.dtype != np.float64:
                map_to_show = map_to_show.astype(np.float32)
            norm_map = cv2.normalize(map_to_show, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            color_map = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
            cv2.imshow(window_title, color_map)
        except cv2.error as e:
            print(f"[Calibration] Error visualizing variance map '{window_title}': {e}")
        except Exception as e:
            print(f"[Calibration] Unexpected error visualizing variance map '{window_title}': {e}")

    def _perform_simplified_evm(self, gray_frames):
        """ Simplified EVM approach. Returns ROI tuple or None. """
        print("[Calibration] Attempting SimplifiedEVM method...") # DEBUG
        target_level = self.config.get('EVM_PYRAMID_LEVEL', 2)
        fs = self.video_input.get_fps()
        low_hz = self.config.get('EVM_LOW_FREQ_HZ', 0.4)
        high_hz = self.config.get('EVM_HIGH_FREQ_HZ', 2.0)
        orig_w, orig_h = self.video_input.get_frame_size()

        if fs <= 0 or high_hz >= fs / 2.0 or low_hz <= 0 or low_hz >= high_hz:
            print(f"[Calibration] Error (EVM): Invalid frequencies for temporal filter. Fs={fs}, Low={low_hz}, High={high_hz}. Nyquist limit={fs/2.0}. Falling back.")
            return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)

        pyramids = []
        level_shapes = set()
        print("[Calibration] Building pyramids for EVM...") # DEBUG
        for i, frame in enumerate(gray_frames): # DEBUG add index
             if frame is None: continue # Skip None frames
             try:
                 pyramid = self._build_gaussian_pyramid(frame, target_level)
                 if len(pyramid) > target_level:
                     level_frame = pyramid[target_level]
                     level_frame_np = level_frame.get() if isinstance(level_frame, cv2.UMat) else level_frame
                     if level_frame_np is not None:
                          pyramids.append(level_frame_np)
                          level_shapes.add(level_frame_np.shape)
                 # else: # DEBUG - Too noisy
                 #     print(f"[Calibration] Warning (EVM): Could not build pyramid to level {target_level} for frame {i}.")
             except cv2.error as e: print(f"[Calibration] Warning (EVM): OpenCV error building pyramid for frame {i}: {e}.")
             except Exception as e: print(f"[Calibration] Warning (EVM): Unexpected error building pyramid for frame {i}: {e}.")

        print(f"[Calibration] Collected {len(pyramids)} pyramid levels for EVM.") # DEBUG
        if not pyramids:
             print("[Calibration] Error (EVM): No valid pyramid levels collected. Falling back.")
             return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)

        if len(level_shapes) > 1:
            print(f"[Calibration] Warning (EVM): Inconsistent shapes found at pyramid level {target_level}: {level_shapes}. Using most common.")
            from collections import Counter
            shape_counts = Counter(p.shape for p in pyramids)
            most_common_shape = shape_counts.most_common(1)[0][0]
            pyramid_data_list = [p for p in pyramids if p.shape == most_common_shape]
            print(f"[Calibration] Using shape {most_common_shape} for EVM analysis ({len(pyramid_data_list)} frames).")
        else:
            pyramid_data_list = pyramids

        if len(pyramid_data_list) < 10:
            print(f"[Calibration] Error (EVM): Not enough consistent pyramid levels ({len(pyramid_data_list)} < 10). Falling back.")
            return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)

        pyramid_sequence = np.stack(pyramid_data_list, axis=0)
        target_shape = pyramid_sequence.shape[1:3]

        print("[Calibration] Applying temporal filter for EVM...") # DEBUG
        filtered_sequence = self._temporal_filter(pyramid_sequence, fs, low_hz, high_hz)
        if filtered_sequence is None:
            print("[Calibration] Error (EVM): Temporal filtering failed. Falling back.")
            return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)

        print("[Calibration] Analyzing variance for EVM...") # DEBUG
        try:
            variance_map = np.var(filtered_sequence, axis=0)
            variance_map_blurred = cv2.GaussianBlur(variance_map.astype(np.float32), (15, 15), 0)
            if self.debug_viz:
                self._visualize_variance_map(variance_map, "Variance Map (Raw - EVM)")
                self._visualize_variance_map(variance_map_blurred, "Variance Map (Blurred - EVM)")
                cv2.waitKey(1)

            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(variance_map_blurred)
            print(f"[Calibration] Simplified EVM - Max Variance: {maxVal:.4f} at {maxLoc} (Level {target_level})") # DEBUG
            if maxVal <= 1e-6: # DEBUG Threshold Check
                print("[Calibration] Warning (EVM): No significant variance found in filtered signal (maxVal <= 1e-6). Falling back.")
                return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)

            scale_factor = 2**target_level
            roi_w_orig = max(20, orig_w // 4)
            roi_h_orig = max(20, orig_h // 4)
            center_x_orig = maxLoc[0] * scale_factor + scale_factor / 2
            center_y_orig = maxLoc[1] * scale_factor + scale_factor / 2
            roi_x_orig = max(0, min(orig_w - roi_w_orig, int(center_x_orig - roi_w_orig / 2)))
            roi_y_orig = max(0, min(orig_h - roi_h_orig, int(center_y_orig - roi_h_orig / 2)))
            roi_w_orig = min(roi_w_orig, orig_w - roi_x_orig)
            roi_h_orig = min(roi_h_orig, orig_h - roi_y_orig)
            print(f"[Calibration] EVM selected ROI: {(int(roi_x_orig), int(roi_y_orig), int(roi_w_orig), int(roi_h_orig))}") # DEBUG
            return (int(roi_x_orig), int(roi_y_orig), int(roi_w_orig), int(roi_h_orig))

        except cv2.error as e_cv:
            print(f"[Calibration] OpenCV error during EVM variance analysis: {e_cv}. Falling back.")
            return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)
        except Exception as e:
            print(f"[Calibration] Unexpected error during EVM variance analysis: {e}. Falling back.")
            traceback.print_exc()
            return self._find_roi_by_motion_variance(gray_frames, precomputed_variance_map=None)

    def _find_roi_by_motion_variance(self, frames, precomputed_variance_map=None):
        """ Finds ROI based on variance of frame differences. Returns ROI tuple or None. """
        print("[Calibration] Attempting MotionVariance method...") # DEBUG
        if not frames and precomputed_variance_map is None: # Need one or the other
             print("[Calibration] Error (MotionVar): No frames and no precomputed map.")
             return None

        # Get frame dimensions from first valid frame if needed
        frame_h, frame_w = -1, -1
        if frames:
             frames_np = [(f.get() if isinstance(f, cv2.UMat) else f) for f in frames if f is not None]
             if frames_np:
                  frame_h, frame_w = frames_np[0].shape[:2]
             else:
                  print("[Calibration] Error (MotionVar): No valid frames provided.")
                  return None
        elif precomputed_variance_map is not None:
             # Get dimensions from map if frames aren't available
             map_h, map_w = precomputed_variance_map.shape[:2]
             frame_h, frame_w = map_h, map_w # Assume map has original frame dims
        else:
             print("[Calibration] Error (MotionVar): Cannot determine frame dimensions.")
             return None

        variance_map_blurred = None
        if precomputed_variance_map is not None:
            print("[Calibration] Using precomputed motion variance map.")
            variance_map_blurred = precomputed_variance_map # Assumes it's already blurred float32
        elif len(frames_np) >= 5: # Calculate map if not precomputed and enough frames
            print("[Calibration] Calculating motion variance map inside _find_roi_by_motion_variance...")
            try:
                diffs = [cv2.absdiff(frames_np[i], frames_np[i+1]) for i in range(len(frames_np)-1)]
                if not diffs: print("[Calibration] Error (MotionVar): Failed to calculate diffs."); return None
                diff_stack = np.stack(diffs, axis=0)
                variance_map = np.var(diff_stack, axis=0)
                variance_map_blurred = cv2.GaussianBlur(variance_map.astype(np.float32), (15, 15), 0)
                print("[Calibration] Motion variance map calculated internally.") # DEBUG
                raw_min, raw_max, _, _ = cv2.minMaxLoc(variance_map) # DEBUG
                print(f"[Calibration] Internally Calculated Raw Var Map Min: {raw_min:.4f}, Max: {raw_max:.4f}") # DEBUG
            except cv2.error as e_cv: print(f"[Calibration] OpenCV error calculating motion variance internally: {e_cv}"); return None
            except Exception as e: print(f"[Calibration] Error calculating motion variance internally: {e}"); return None
        else:
             print("[Calibration] Warning (MotionVar): Not enough frames to calculate motion variance.")
             return None

        if variance_map_blurred is None or variance_map_blurred.size == 0:
             print("[Calibration] Error (MotionVar): Failed to obtain blurred variance map.")
             return None

        try:
            map_for_loc = variance_map_blurred.get() if isinstance(variance_map_blurred, cv2.UMat) else variance_map_blurred
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(map_for_loc)
            print(f"[Calibration] Motion Variance - Max Blurred Variance: {maxVal:.4f} at {maxLoc}") # DEBUG

            roi_w = max(20, frame_w // 4)
            roi_h = max(20, frame_h // 4)

            if maxVal <= 1e-6: # DEBUG Threshold Check
                print("[Calibration] Warning (MotionVar): No significant motion variance found (maxVal <= 1e-6). Centering ROI.")
                roi_x = max(0, frame_w // 2 - roi_w // 2)
                roi_y = max(0, frame_h // 2 - roi_h // 2)
            else:
                roi_x = max(0, min(frame_w - roi_w, maxLoc[0] - roi_w // 2))
                roi_y = max(0, min(frame_h - roi_h, maxLoc[1] - roi_h // 2))

            roi_w = min(roi_w, frame_w - roi_x)
            roi_h = min(roi_h, frame_h - roi_y)
            final_roi = (int(roi_x), int(roi_y), int(roi_w), int(roi_h))
            print(f"[Calibration] MotionVar selected ROI: {final_roi}") # DEBUG
            return final_roi

        except cv2.error as e_cv: print(f"[Calibration] OpenCV error during minMaxLoc for motion variance: {e_cv}"); return None
        except Exception as e: print(f"[Calibration] Unexpected error finding max variance location: {e}"); traceback.print_exc(); return None

    def _calculate_roi_signal_variance(self, roi, frames_np):
        """Helper to calculate variance of ROI average signal for refinement."""
        # print("[Calibration] Calculating ROI signal variance for refinement...") # DEBUG - Noisy
        if roi is None: return 0
        x, y, w, h = roi
        motion_signal = []
        prefilter = self.config.get('ROI_AVG_PREFILTER', None)
        median_ksize = self.config.get('ROI_AVG_MEDIAN_KSIZE', 5)
        gaussian_kernel = tuple(self.config.get('ROI_AVG_GAUSSIAN_KERNEL', (3,3)))

        if w <= 0 or h <= 0: return 0

        for frame in frames_np:
            try:
                if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0: continue
                actual_y = max(0, y); actual_x = max(0, x)
                actual_y_end = min(frame.shape[0], y + h); actual_x_end = min(frame.shape[1], x + w)
                if actual_y_end <= actual_y or actual_x_end <= actual_x: continue
                roi_frame = frame[actual_y:actual_y_end, actual_x:actual_x_end]
                if roi_frame.shape[0] == 0 or roi_frame.shape[1] == 0: continue

                filtered_roi = roi_frame
                try:
                    if prefilter == 'Median':
                        if median_ksize > 1 and median_ksize % 2 == 1: filtered_roi = cv2.medianBlur(roi_frame, median_ksize)
                        # else: print(f"Warning: Invalid median_ksize ({median_ksize}), skipping median filter.")
                    elif prefilter == 'Gaussian':
                         if len(gaussian_kernel) == 2 and gaussian_kernel[0] > 0 and gaussian_kernel[1] > 0 and gaussian_kernel[0] % 2 == 1 and gaussian_kernel[1] % 2 == 1: filtered_roi = cv2.GaussianBlur(roi_frame, gaussian_kernel, 0)
                         # else: print(f"Warning: Invalid gaussian_kernel ({gaussian_kernel}), skipping Gaussian filter.")

                    if len(filtered_roi.shape) == 3: gray_roi = cv2.cvtColor(filtered_roi, cv2.COLOR_BGR2GRAY)
                    else: gray_roi = filtered_roi

                    mean_val = cv2.mean(gray_roi)[0]
                    motion_signal.append(mean_val)
                except (cv2.error, Exception): continue
            except Exception as e_outer: print(f"Warning: Error processing frame for ROI signal variance: {e_outer}"); continue

        if len(motion_signal) < 5: return 0
        return np.var(motion_signal)

    def _refine_roi_size(self, initial_roi, frames_np):
        """Tests ROI size variations and selects based on signal variance."""
        # print("[Calibration] Refining ROI size...") # DEBUG - Noisy
        if initial_roi is None: return None
        orig_x, orig_y, orig_w, orig_h = initial_roi
        if not frames_np or len(frames_np) == 0: return initial_roi
        frame_h, frame_w = frames_np[0].shape[:2]
        best_roi = initial_roi
        max_variance = self._calculate_roi_signal_variance(initial_roi, frames_np)
        if max_variance is None: return initial_roi
        # print(f"  Initial ROI variance: {max_variance:.4f}") # DEBUG - Noisy
        scale_factors = [0.8, 0.9, 1.1, 1.2]
        for scale_w in scale_factors:
            for scale_h in scale_factors:
                if scale_w == 1.0 and scale_h == 1.0: continue
                new_w = int(orig_w * scale_w); new_h = int(orig_h * scale_h)
                new_x = orig_x + (orig_w - new_w) // 2; new_y = orig_y + (orig_h - new_h) // 2
                new_x = max(0, new_x); new_y = max(0, new_y)
                new_w = max(10, new_w); new_h = max(10, new_h)
                if new_x + new_w > frame_w: new_w = frame_w - new_x
                if new_y + new_h > frame_h: new_h = frame_h - new_y
                if new_w <= 0 or new_h <= 0: continue
                current_roi = (new_x, new_y, new_w, new_h)
                if current_roi == best_roi: continue
                variance = self._calculate_roi_signal_variance(current_roi, frames_np)
                if variance is None: continue
                if variance > max_variance:
                    max_variance = variance; best_roi = current_roi
                    # print(f"  New best ROI: {best_roi} (Variance: {max_variance:.4f})") # DEBUG - Noisy
        return best_roi

# Example usage (keep as is)
if __name__ == '__main__':
    print("Testing Calibration module...")
    # ... (rest of the test code remains the same) ...
    class MockVideoInput:
        def __init__(self, fps=30, width=640, height=480):
            self._fps = fps; self._w = width; self._h = height; self.frame_count = 0; self.use_opencl = False
        def initialize(self): return True
        def get_frame(self):
            frame = np.random.randint(0, 256, (self._h, self._w, 3), dtype=np.uint8)
            offset = int(self._w * 0.3)
            rect_size = int(self._w * 0.1)
            x = int(self._w/2 - offset/2 + offset * np.sin(self.frame_count * 0.1))
            y = int(self._h/2 - rect_size/2)
            cv2.rectangle(frame, (x, y), (x+rect_size, y+rect_size), (255,255,255), -1)
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
        if os.path.exists(test_config_path):
             test_config = load_config(test_config_path)
             print(f"Loaded config from: {test_config_path}")
        else:
             print(f"Warning: Default config not found at {test_config_path}. Using basic test config.")
             raise ImportError
        test_config['DEBUG_CALIBRATION_VIZ'] = True
        test_config['CALIBRATION_DURATION_FRAMES'] = 50
        test_config['CALIBRATION_METHOD'] = 'MotionVariance'
        test_config['ROI_REFINE_SIZE'] = True
    except ImportError:
        print("Could not import config_loader or find config file, using basic test config.")
        test_config = {
            'CALIBRATION_METHOD': 'MotionVariance', 'CALIBRATION_DURATION_FRAMES': 50,
            'DEBUG_CALIBRATION_VIZ': True, 'ROI_REFINE_SIZE': True,
            'EVM_PYRAMID_LEVEL': 2, 'EVM_LOW_FREQ_HZ': 0.4, 'EVM_HIGH_FREQ_HZ': 2.0,
            'ROI_AVG_PREFILTER': None, 'ROI_AVG_MEDIAN_KSIZE': 5, 'ROI_AVG_GAUSSIAN_KERNEL': [3,3]
        }

    mock_video = MockVideoInput()
    if mock_video.initialize():
        calibrator = Calibration(mock_video, test_config)
        selected_roi, variance_map = calibrator.run_calibration()
        print(f"Test calibration finished.")
        if selected_roi: print(f"  Selected ROI: {selected_roi}")
        else: print(f"  Selected ROI: None")
        if variance_map is not None: print(f"  Returned Variance Map Shape: {variance_map.shape}, Type: {variance_map.dtype}")
        else: print(f"  Returned Variance Map: None")
        if test_config['DEBUG_CALIBRATION_VIZ']:
            if variance_map is not None: calibrator._visualize_variance_map(variance_map, "Final Variance Map for Weighting")
            print("Close debug windows by pressing any key...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

