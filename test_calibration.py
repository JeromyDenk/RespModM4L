# test_calibration.py
# Unit test script for src/calibration.py

import unittest
import os
import sys
import cv2
import numpy as np
import time

# --- Add src directory to Python path ---
# This allows importing modules from the 'src' directory
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Import necessary modules from src ---
try:
    from calibration import Calibration
    from config_loader import load_config
    # We need a mock VideoInput, let's define one here based on the one in calibration.py
except ImportError as e:
    print(f"Error importing modules from 'src': {e}")
    print("Please ensure this script is run from the project root directory")
    print("and all necessary files exist in the 'src' folder.")
    sys.exit(1)

# --- Configuration for Tests ---
# Assumes 'profiles' folder is in the same directory as this script (project root)
CONFIG_FOLDER = "profiles"
# --- !!! ADJUST THIS FILENAME if needed !!! ---
TEST_CONFIG_FILENAME = "profile_1.json"
# --- !!! ADJUST THIS FILENAME if needed !!! ---
TEST_CONFIG_FILEPATH = os.path.join(script_dir, CONFIG_FOLDER, TEST_CONFIG_FILENAME)

# --- Mock Video Input Class ---
class MockVideoInput:
    """A mock VideoInput class for testing Calibration."""
    def __init__(self, fps=30, width=640, height=480, motion_type='sine', use_opencl=False):
        """
        Initializes the mock video input.

        Args:
            fps (int): Frames per second.
            width (int): Frame width.
            height (int): Frame height.
            motion_type (str): Type of motion to simulate ('sine', 'static', 'noise').
            use_opencl (bool): Simulate OpenCL usage status.
        """
        self._fps = float(fps)
        self._w = int(width)
        self._h = int(height)
        self.frame_count = 0
        self.motion_type = motion_type
        self.use_opencl = use_opencl # Store OpenCL status
        self.initialized = False
        print(f"[MockVideoInput] Initialized: {self._w}x{self._h} @ {self._fps:.1f} FPS, Motion: {self.motion_type}, OpenCL: {self.use_opencl}")

    def initialize(self):
        """Simulates successful initialization."""
        self.initialized = True
        print("[MockVideoInput] Initialized successfully.")
        return True

    def get_frame(self):
        """Generates a dummy frame with simulated motion."""
        if not self.initialized:
            return False, None

        # Generate base noise frame
        frame = np.random.randint(50, 150, (self._h, self._w, 3), dtype=np.uint8)

        # Add simulated motion based on type
        if self.motion_type == 'sine':
            # Add a moving rectangle to simulate motion variance
            rect_size = max(20, int(self._w * 0.1))
            offset_amplitude = int(self._w * 0.25)
            center_x = self._w // 2
            center_y = self._h // 2
            # Calculate position based on sine wave
            x_pos = int(center_x - offset_amplitude/2 + offset_amplitude * np.sin(self.frame_count * 2 * np.pi / (self._fps * 5))) # 5-second cycle
            y_pos = center_y - rect_size // 2
            # Ensure coordinates are within bounds
            x1 = max(0, x_pos)
            y1 = max(0, y_pos)
            x2 = min(self._w, x_pos + rect_size)
            y2 = min(self._h, y_pos + rect_size)
            if x1 < x2 and y1 < y2: # Draw only if valid
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
        elif self.motion_type == 'static':
            # No additional motion, just the noise frame
            pass
        elif self.motion_type == 'noise':
            # Frame is already random noise
             pass

        self.frame_count += 1
        # Simulate potential OpenCL UMat return
        if self.use_opencl:
             try:
                  return True, cv2.UMat(frame)
             except cv2.error: # Fallback if UMat conversion fails (e.g., no OpenCL support)
                  return True, frame
        else:
             return True, frame

    def get_fps(self):
        """Returns the simulated FPS."""
        return self._fps

    def get_frame_size(self):
        """Returns the simulated frame size."""
        return self._w, self._h

    def release(self):
        """Simulates releasing the resource."""
        print("[MockVideoInput] Released.")
        self.initialized = False

# --- Test Class ---
class TestCalibration(unittest.TestCase):
    """Test suite for the Calibration class."""

    test_config = None # Class variable to store loaded config

    @classmethod
    def setUpClass(cls):
        """Load configuration once for all tests."""
        print(f"\n--- Loading Test Configuration ({TEST_CONFIG_FILEPATH}) ---")
        if not os.path.exists(TEST_CONFIG_FILEPATH):
            print(f"FATAL: Test config file not found: {TEST_CONFIG_FILEPATH}")
            # Use minimal defaults as fallback? Or raise error?
            # For now, let's raise an error to make it clear config is missing.
            raise FileNotFoundError(f"Test config file not found: {TEST_CONFIG_FILEPATH}")
        cls.test_config = load_config(TEST_CONFIG_FILEPATH)
        print("--- Configuration Loaded ---")

    def setUp(self):
        """Set up common resources for each test."""
        # Reset visualization setting for each test based on loaded config
        self.debug_viz = self.test_config.get('DEBUG_CALIBRATION_VIZ', False)
        # Suppress default viz window popping up unless specifically tested
        self.test_config['DEBUG_CALIBRATION_VIZ'] = False
        print(f"\n--- Running Test: {self._testMethodName} ---")

    def tearDown(self):
        """Clean up after each test."""
        # Ensure any OpenCV windows opened during a test are closed
        cv2.destroyAllWindows()
        cv2.waitKey(1) # Give time for windows to close
        print(f"--- Finished Test: {self._testMethodName} ---")

    def run_single_calibration(self, motion_type='sine', method='MotionVariance', refine=False, viz=False, duration=50):
        """Helper function to run one calibration test."""
        print(f"  Configuring Test: Motion='{motion_type}', Method='{method}', Refine={refine}, Viz={viz}, Duration={duration}")

        # Create a copy of the base config and modify it for this test run
        current_config = self.test_config.copy()
        current_config['CALIBRATION_METHOD'] = method
        current_config['ROI_REFINE_SIZE'] = refine
        current_config['DEBUG_CALIBRATION_VIZ'] = viz
        current_config['CALIBRATION_DURATION_FRAMES'] = duration

        # Create mock video input with specified motion
        mock_video = MockVideoInput(motion_type=motion_type, use_opencl=current_config.get('USE_OPENCL', False))
        self.assertTrue(mock_video.initialize(), "MockVideoInput failed to initialize")

        # Create and run calibration
        calibrator = Calibration(mock_video, current_config)
        start_t = time.time()
        selected_roi, variance_map = calibrator.run_calibration()
        end_t = time.time()
        print(f"  Calibration Run Time: {end_t - start_t:.2f} seconds")

        # Print results
        print(f"  Selected ROI: {selected_roi}")
        if variance_map is not None:
            map_shape = variance_map.shape
            map_dtype = variance_map.dtype
            print(f"  Variance Map: Shape={map_shape}, Dtype={map_dtype}")
        else:
            print(f"  Variance Map: None")

        # Basic Assertions
        self.assertIsNotNone(selected_roi, f"Calibration failed to return an ROI (Motion='{motion_type}', Method='{method}')")
        self.assertIsInstance(selected_roi, tuple, "Selected ROI should be a tuple")
        self.assertEqual(len(selected_roi), 4, "Selected ROI should have 4 elements (x, y, w, h)")
        self.assertTrue(all(isinstance(n, int) for n in selected_roi), "ROI elements should be integers")
        self.assertTrue(selected_roi[2] > 0, "ROI width should be positive")
        self.assertTrue(selected_roi[3] > 0, "ROI height should be positive")

        self.assertIsNotNone(variance_map, "Calibration failed to return a variance map")
        self.assertIsInstance(variance_map, np.ndarray, "Variance map should be a NumPy array")
        # Check if map shape matches video input size
        vid_w, vid_h = mock_video.get_frame_size()
        self.assertEqual(variance_map.shape, (vid_h, vid_w), "Variance map shape should match video frame size")
        self.assertEqual(variance_map.dtype, np.float32, "Variance map dtype should be float32")

        # Handle visualization window if enabled for this test
        if viz:
            print("  Debug Visualization was enabled. Press any key in an OpenCV window to continue...")
            cv2.waitKey(0) # Wait indefinitely until user presses a key

        mock_video.release()
        cv2.destroyAllWindows() # Clean up any windows opened by this specific test run

    # --- Test Cases ---

    def test_01_motion_variance_basic(self):
        """Test MotionVariance method with sine motion."""
        self.run_single_calibration(motion_type='sine', method='MotionVariance')

    def test_02_motion_variance_static(self):
        """Test MotionVariance method with static input (should still return centered ROI)."""
        self.run_single_calibration(motion_type='static', method='MotionVariance')

    def test_03_motion_variance_refine(self):
        """Test MotionVariance method with refinement enabled."""
        self.run_single_calibration(motion_type='sine', method='MotionVariance', refine=True)

    # def test_04_simplified_evm_basic(self): # EVM can be slow, optional
    #     """Test SimplifiedEVM method with sine motion."""
    #     self.run_single_calibration(motion_type='sine', method='SimplifiedEVM')

    # def test_05_simplified_evm_static(self): # EVM can be slow, optional
    #     """Test SimplifiedEVM method with static input."""
    #     self.run_single_calibration(motion_type='static', method='SimplifiedEVM')

    def test_06_motion_variance_viz(self):
        """Test MotionVariance method with visualization enabled."""
        print("\n  --- This test will open OpenCV windows. Press any key in a window to proceed. ---")
        # Temporarily override the setUp suppression for this test
        self.test_config['DEBUG_CALIBRATION_VIZ'] = True
        self.run_single_calibration(motion_type='sine', method='MotionVariance', viz=True)
        self.test_config['DEBUG_CALIBRATION_VIZ'] = False # Reset for other tests


# --- Run Tests ---
if __name__ == '__main__':
    print("=======================================")
    print("     Running Calibration Test Suite    ")
    print("=======================================")
    unittest.main()
