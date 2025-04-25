# main.py
# Runs the RespirationMonitor, displays UI, and sends data via OSC.
# v2: Reintegrated UI Manager

import time
import numpy as np
import cv2 # Needed for UI display frame conversion
import matplotlib.pyplot as plt # Needed for UI
import matplotlib.patches as patches # Needed for UI

# --- OSC Library Import ---
# Ensure you have installed it: pip install python-osc
try:
    from pythonosc import udp_client
    from pythonosc import osc_message_builder
    osc_available = True
except ImportError:
    print("Warning: 'python-osc' library not found. OSC output will be disabled.")
    print("Install it using: pip install python-osc")
    osc_available = False
    # Define dummy classes/functions if library is missing to avoid NameErrors later
    class DummyOSCClient:
        def send_message(self, address, value): pass
    udp_client = DummyOSCClient() # Assign dummy client

# Import the core monitor class
try:
    # Assuming respiration_monitor.py is in the same directory
    from respiration_monitor import RespirationMonitor, DEFAULT_CONFIG
except ImportError:
    print("ERROR: Could not import RespirationMonitor from respiration_monitor.py.")
    print("Ensure respiration_monitor.py is in the same directory as main.py.")
    exit() # Exit if the core module cannot be imported

# --- OSC Configuration ---
OSC_CONFIG = {
    'ENABLE_OSC': True,
    'OSC_IP_ADDRESS': "127.0.0.1", # Target IP (Ableton/M4L machine)
    'OSC_PORT': 9001,             # Target Port
    'OSC_ADDRESS_SIGNAL': "/respmon/signal", # Address for normalized signal
    'OSC_ADDRESS_BPM': "/respmon/bpm",       # Address for BPM
    'OSC_ADDRESS_STATUS': "/respmon/status" # Address for status messages
}

# --- UI Configuration ---
UI_CONFIG = {
    'ENABLE_UI': True # Set to False to disable the Matplotlib window
}

# --- Module: UIManager (Copied back from previous version) ---
class UIManager:
    """Handles basic UI display using Matplotlib."""
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6))
        self.fig.canvas.manager.set_window_title('Respmon Enhanced v4 (OSC + UI)')

        # Video display subplot
        self.ax_video = self.axs[0]
        self.video_im = self.ax_video.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        self.ax_video.set_title("Video Feed")
        self.ax_video.axis('off')
        self.roi_rect_patch = None
        self.status_text = self.ax_video.text(0.02, 0.95, 'Status: Initializing', color='white',
                                          fontsize=10, va='top', ha='left',
                                          bbox=dict(facecolor='black', alpha=0.5, pad=0.2))

        # Signal plot subplot
        self.ax_plot = self.axs[1]
        self.line, = self.ax_plot.plot([], [], 'b-') # Signal line
        self.peaks_scatter = self.ax_plot.scatter([], [], c='r', marker='x', s=50) # Peak markers
        self.ax_plot.set_title("Motion Signal")
        self.ax_plot.set_xlabel("Sample Index")
        self.ax_plot.set_ylabel("Motion Value (Normalized)") # Label change

        self.fig.tight_layout(pad=1.5)
        plt.ion() # Interactive mode ON
        plt.show()
        self._is_figure_open = True
        self.fig.canvas.mpl_connect('close_event', self._handle_close)
        self.last_plot_signal = None # Store the data used for the last plot


    def _handle_close(self, evt):
        """Callback function for when the plot window is closed."""
        print("UI window closed by user.")
        self._is_figure_open = False

    def is_open(self):
        """Check if the UI window is still open."""
        # Also check if the underlying figure exists
        return self._is_figure_open and plt.fignum_exists(self.fig.number)


    def display_frame(self, frame, roi=None):
        """Displays the video frame and ROI."""
        if not self.is_open() or frame is None: return

        # Ensure frame is NumPy BGR for display
        frame_np = frame # Assume it's already NumPy from monitor results
        if frame_np is None: return

        try:
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            self.video_im.set_data(frame_rgb)
            h, w = frame_rgb.shape[:2]
            self.video_im.set_extent([0, w, h, 0])

            if roi:
                x, y, w_roi, h_roi = roi
                if self.roi_rect_patch:
                    self.roi_rect_patch.set_bounds(x, y, w_roi, h_roi)
                else:
                    self.roi_rect_patch = patches.Rectangle((x, y), w_roi, h_roi, linewidth=1, edgecolor='lime', facecolor='none')
                    self.ax_video.add_patch(self.roi_rect_patch)
            elif self.roi_rect_patch:
                 self.roi_rect_patch.remove()
                 self.roi_rect_patch = None

            self.ax_video.set_xlim(0, w)
            self.ax_video.set_ylim(h, 0)
        except cv2.error as e:
             print(f"Warning: OpenCV error during UI frame display: {e}")
        except Exception as e:
             print(f"Warning: Error during UI frame display: {e}")


    def display_plot(self, signal_data, peaks=None):
        """Updates the motion signal plot after normalizing the data.

        Args:
            signal_data (np.ndarray or None): The raw filtered signal buffer.
            peaks (np.ndarray or None): Indices of detected peaks.
        """
        if not self.is_open():
             self.last_plot_signal = None
             return

        if signal_data is None or len(signal_data) == 0:
            # Clear plot if no valid data
            self.line.set_data([], [])
            self.peaks_scatter.set_offsets(np.empty((0, 2)))
            # Keep previous axes limits if clearing, or reset? Resetting might be jumpy.
            # self.ax_plot.relim()
            # self.ax_plot.autoscale_view()
            self.last_plot_signal = None
            return

        # --- Normalization for Plotting ---
        try:
            signal_mean = np.mean(signal_data)
            signal_std = np.std(signal_data)
            if signal_std > 1e-6:
                plot_signal = (signal_data - signal_mean) / (3 * signal_std)
                plot_signal = np.clip(plot_signal, -2, 2)
            else:
                plot_signal = np.zeros_like(signal_data)

            # Store the data being plotted
            self.last_plot_signal = plot_signal

            t = np.arange(len(plot_signal))
            self.line.set_data(t, plot_signal)

            if peaks is not None and len(peaks) > 0:
                valid_peaks = peaks[peaks < len(plot_signal)]
                if len(valid_peaks) > 0:
                     peak_values_normalized = plot_signal[valid_peaks]
                     finite_peak_indices = valid_peaks[np.isfinite(peak_values_normalized)]
                     finite_peak_values = peak_values_normalized[np.isfinite(peak_values_normalized)]
                     if len(finite_peak_indices) > 0:
                         self.peaks_scatter.set_offsets(np.c_[finite_peak_indices, finite_peak_values])
                     else:
                         self.peaks_scatter.set_offsets(np.empty((0, 2)))
                else:
                    self.peaks_scatter.set_offsets(np.empty((0, 2)))
            else:
                self.peaks_scatter.set_offsets(np.empty((0, 2)))

            # Autoscale view based on the normalized data
            self.ax_plot.relim()
            self.ax_plot.autoscale_view()
            self.ax_plot.set_ylim(-2.5, 2.5) # Fixed y-limits

        except Exception as e:
             print(f"Warning: Error during UI plot display: {e}")
             # Clear plot on error?
             self.line.set_data([], [])
             self.peaks_scatter.set_offsets(np.empty((0, 2)))


    def display_status(self, bpm, fps, method):
        """Updates the status text display."""
        if not self.is_open(): return
        bpm_str = f"{bpm:.1f}" if bpm > 0 else "--"
        status = f"BPM: {bpm_str} | FPS: {fps:.1f} | Method: {method}"
        self.status_text.set_text(status)

    def update_ui(self):
        """Redraws the UI."""
        if not self.is_open(): return False
        try:
            self.fig.canvas.flush_events()
            plt.pause(0.001)
            return True
        except Exception as e:
            # Handle specific backend errors if window is closed during pause/flush
            if "invalid command name" in str(e) or \
               "application has been destroyed" in str(e) or \
               "FigureCanvasAgg" in str(type(e)):
                 print("UI window seems closed.")
                 self._is_figure_open = False
                 return False
            else:
                 # Ignore other potential minor drawing errors if window still exists
                 if plt.fignum_exists(self.fig.number):
                     # print(f"Minor UI update error: {e}") # Can be noisy
                     return True
                 else:
                     print(f"Error updating UI and window closed: {e}")
                     self._is_figure_open = False
                     return False

    def close(self):
        """Closes the Matplotlib window."""
        if self.is_open():
            plt.close(self.fig)
            self._is_figure_open = False
            print("UI closed programmatically.")


# --- Main Application Class ---
class MainApplication:
    def __init__(self):
        self.monitor = None
        self.osc_client = None
        self.ui_manager = None # Add UI manager instance variable
        self.last_loop_time = time.time()
        self.current_fps = 0.0
        self.running = False

    def _initialize_osc(self):
        """Initializes the OSC client."""
        if OSC_CONFIG.get('ENABLE_OSC', False) and osc_available: # Use get()
            ip = OSC_CONFIG.get('OSC_IP_ADDRESS', "127.0.0.1")
            port = OSC_CONFIG.get('OSC_PORT', 9001)
            try:
                if osc_available: self.osc_client = udp_client.SimpleUDPClient(ip, port)
                else: self.osc_client = None; return
                print(f"OSC Client initialized. Sending to {ip}:{port}")
                self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_STATUS', "/respmon/status"), "connected")
            except Exception as e:
                print(f"Error initializing OSC client: {e}"); self.osc_client = None; print("Warning: OSC output disabled.")
        elif not osc_available: print("OSC output disabled: python-osc library missing.")
        else: print("OSC output disabled in configuration.")

    def _initialize_ui(self):
        """Initializes the UI Manager if enabled."""
        if UI_CONFIG.get('ENABLE_UI', False):
             print("Initializing UI...")
             try:
                 self.ui_manager = UIManager()
             except Exception as e:
                  print(f"Error initializing UI Manager: {e}")
                  print("Warning: UI display disabled.")
                  self.ui_manager = None # Ensure it's None if init fails
        else:
             print("UI display disabled in configuration.")

    def _send_osc_data(self, bpm, signal_value):
        """Sends BPM and signal data via OSC."""
        if not self.osc_client or not OSC_CONFIG.get('ENABLE_OSC', False) or not osc_available: return
        try:
            bpm_float = float(bpm) if bpm is not None and np.isfinite(bpm) else 0.0
            signal_float = float(signal_value) if signal_value is not None and np.isfinite(signal_value) else 0.0
            self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_BPM', "/respmon/bpm"), bpm_float)
            self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_SIGNAL', "/respmon/signal"), signal_float)
        except Exception as e:
            print(f"Error sending OSC message: {e}")

    def _calculate_fps(self):
        """Calculates FPS based on time since last loop."""
        current_time = time.time()
        delta_time = current_time - self.last_loop_time
        self.last_loop_time = current_time
        if delta_time > 0: self.current_fps = 1.0 / delta_time
        else: self.current_fps = 0.0

    def run(self):
        """Starts the monitor and runs the main loop."""
        self.running = True
        print("Starting Main Application...")

        # --- Initialize Monitor ---
        monitor_config_overrides = { }
        self.monitor = RespirationMonitor(config_overrides=monitor_config_overrides)
        if not self.monitor.initialize():
            print("Failed to initialize Respiration Monitor. Exiting."); self.running = False; return

        # --- Initialize OSC and UI ---
        self._initialize_osc()
        self._initialize_ui() # Initialize UI

        # --- Main Loop ---
        print("Starting main processing loop (Press Ctrl+C or close UI window to exit)...")
        frame_count = 0
        start_loop_time = time.time()
        self.last_loop_time = start_loop_time

        # Loop condition now also checks if UI is open (if UI is enabled)
        while self.running and (self.ui_manager is None or self.ui_manager.is_open()):
            try:
                self._calculate_fps()

                # Run monitor cycle
                results = self.monitor.run_cycle()

                if not results['success']:
                    print(f"Monitor cycle failed: {results.get('error', 'Unknown')}")
                    if results.get('error') == 'Video ended or failed': self.running = False
                    time.sleep(0.1); continue

                # Get results
                bpm = results['bpm']
                normalized_signal = results['normalized_signal']
                frame = results['frame'] # Get frame for UI
                roi = results['roi']     # Get ROI for UI
                method = results['method'] # Get method for UI status

                # Trigger Adaptive Control
                if self.monitor.adaptive_controller:
                    self.monitor.trigger_adaptation(self.current_fps)

                # Send OSC Data
                self._send_osc_data(bpm, normalized_signal)

                # --- Update UI (if enabled) ---
                if self.ui_manager:
                    self.ui_manager.display_frame(frame, roi)
                    # Get the filtered signal buffer directly from the processor for plotting
                    filtered_signal_buffer = self.monitor.signal_processor.last_valid_filtered_signal
                    # Get peaks from the results dict if needed for plotting (currently not directly used by plot func)
                    peaks = results.get('peaks') # Assuming run_cycle might return peaks if needed
                    self.ui_manager.display_plot(filtered_signal_buffer, peaks)
                    self.ui_manager.display_status(bpm, self.current_fps, method)
                    # update_ui() returgns False if the window was closed
                    if not self.ui_manager.update_ui():
                         self.running = False # Stop main loop if UI closed

                # Console Output
                frame_count += 1
                if frame_count % 60 == 0:
                    print(f"FPS: {self.current_fps:.1f} | BPM: {bpm:.1f} | NormSignal: {normalized_signal:.3f} | Method: {self.monitor.get_current_method()}")

            except KeyboardInterrupt:
                print("\nCtrl+C detected. Stopping..."); self.running = False
            except Exception as e:
                print(f"\n--- An error occurred in main loop ---"); import traceback; traceback.print_exc(); print(f"Error: {e}"); print("Attempting to continue..."); time.sleep(1)

        # --- Cleanup ---
        end_loop_time = time.time(); total_time = end_loop_time - start_loop_time
        print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds.")
        if total_time > 0 and frame_count > 0: print(f"Average FPS: {frame_count / total_time:.2f}")
        print("Cleaning up resources...")
        if self.monitor: self.monitor.stop()
        if self.ui_manager and self.ui_manager.is_open(): self.ui_manager.close() # Close UI window if open
        if self.osc_client and osc_available:
             try: self.osc_client.send_message(OSC_CONFIG.get('OSC_ADDRESS_STATUS', "/respmon/status"), "disconnected")
             except Exception: pass
        print("Main application finished.")

# --- Entry Point ---
if __name__ == "__main__":
    app = MainApplication()
    app.run()
