# src/video_input.py
# Contains the VideoInput class for handling camera/video sources.

import cv2
import time # Although not directly used now, might be useful later
import warnings
import os
import numpy as np

class VideoInput:
    """Handles video capture from webcam or file."""
    def __init__(self, config):
        """Initializes VideoInput with configuration.

        Args:
            config (dict): Configuration dictionary containing VIDEO_SOURCE, USE_OPENCL, etc.
        """
        self.config = config
        self.capture = None
        self.fps = 30 # Default FPS
        self.use_opencl = False
        self.frame_width = 0
        self.frame_height = 0

    def initialize(self):
        """Initializes the video capture source based on config."""
        try:
            source = self.config.get('VIDEO_SOURCE', 0) # Default to 0 if not specified
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise IOError(f"Cannot open video source: {source}") # Specific error

            # Get properties from the capture device
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Handle cases where FPS is not reported correctly
            if self.fps <= 0 or not np.isfinite(self.fps): # Added check for non-finite FPS
                self.fps = 30 # Fallback FPS
                print(f"Warning: Could not get valid FPS from video source ({self.fps}), defaulting to 30.")

            # Check config and OpenCV availability for OpenCL
            self.use_opencl = self.config.get('USE_OPENCL', False) and cv2.ocl.haveOpenCL()
            cv2.ocl.setUseOpenCL(self.use_opencl) # Enable/disable OpenCL globally

            print(f"Video source initialized. Res: {self.frame_width}x{self.frame_height}, FPS: {self.fps:.2f}, OpenCL: {self.use_opencl}")
            return True
        except IOError as e: # Catch file/device access errors
             print(f"Error initializing video input (IOError): {e}")
             self.capture = None; return False
        except cv2.error as e: # Catch OpenCV specific errors
             print(f"Error initializing video input (OpenCV Error): {e}")
             self.capture = None; return False
        except Exception as e: # Catch any other unexpected errors
             print(f"Unexpected error initializing video input: {e}")
             self.capture = None; return False

    def get_frame(self):
        """Retrieves the next frame from the source.

        Returns:
            tuple: (bool success, frame) where frame is NumPy array or UMat.
                   Returns (False, None) if capture fails or ends.
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None
        try:
            success, frame = self.capture.read()
            if not success:
                return False, None # End of video or capture error

            # If OpenCL is enabled, try converting to UMat
            if self.use_opencl:
                try:
                    return success, cv2.UMat(frame) # Return UMat
                except (cv2.error, Exception) as e: # Catch potential conversion errors
                    print(f"Warning: UMat conversion failed ({e}), disabling OpenCL and using NumPy.")
                    self.use_opencl = False; cv2.ocl.setUseOpenCL(False); # Disable OpenCL for future frames
                    return success, frame # Fallback to NumPy array
            else:
                return success, frame # Return NumPy array directly
        except cv2.error as e: # Catch errors during frame reading
             print(f"Error reading frame (OpenCV Error): {e}"); return False, None
        except Exception as e: # Catch other unexpected errors
             print(f"Unexpected error reading frame: {e}"); return False, None

    def get_fps(self):
        """Returns the frames per second of the video source."""
        return self.fps

    def get_frame_size(self):
        """Returns the frame width and height."""
        return self.frame_width, self.frame_height

    def release(self):
        """Releases the video capture object."""
        if self.capture:
            try:
                self.capture.release()
                print("Video source released.")
            except Exception as e: # Catch potential errors during release
                 print(f"Error releasing video source: {e}")

# Example usage (for testing this module directly)
if __name__ == '__main__':
    # Assumes config_loader is in the same directory (src)
    # If running directly, need to handle path differently or use default dict
    try:
        from config_loader import load_config # Try importing sibling module
        # Assume configs folder is one level up
        script_dir = os.path.dirname(__file__)
        config_dir = os.path.join(script_dir, '..', 'configs')
        test_config_path = os.path.join(config_dir, 'config_default.json')
        test_config = load_config(test_config_path)
    except ImportError:
        print("Warning: Could not import config_loader. Using basic default config for test.")
        test_config = {'VIDEO_SOURCE': 0, 'USE_OPENCL': False}


    print("Testing VideoInput module...")
    video_in = VideoInput(test_config)

    if video_in.initialize():
        print(f"FPS: {video_in.get_fps()}")
        print(f"Size: {video_in.get_frame_size()}")

        count = 0
        max_frames_to_show = 50
        start_time = time.time()

        while count < max_frames_to_show:
            success, frame = video_in.get_frame()
            if not success:
                print("Failed to get frame or video ended.")
                break

            # Convert UMat back to NumPy for display if needed
            frame_display = frame.get() if isinstance(frame, cv2.UMat) else frame

            cv2.imshow("VideoInput Test", frame_display)
            count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        print(f"Displayed {count} frames in {end_time - start_time:.2f} seconds.")
        video_in.release()
        cv2.destroyAllWindows()
    else:
        print("VideoInput initialization failed.")

