# Respmon Enhanced (OSC Output Version)

This project uses a webcam to monitor breathing patterns in real-time and outputs the detected respiratory signal and calculated breaths per minute (BPM) via Open Sound Control (OSC) for integration with other applications like Ableton Live (using Max for Live). It also provides an optional visual interface showing the video feed, selected Region of Interest (ROI), and the breathing signal plot.

This version is based on the analysis and recommendations from the research paper "Enhancing Real-Time Performance of the respmon Respiratory Monitoring System".

## Features

* Real-time breathing monitoring from webcam video.
* Manual Region of Interest (ROI) selection at startup.
* Optional automatic ROI calibration (Simplified EVM or Motion Variance methods).
* Selectable motion detection methods:
    * ROI Pixel Averaging (with optional Median/Gaussian pre-filtering).
    * Optical Flow (Sparse Lucas-Kanade with PCA-based signal extraction).
* Configurable signal filtering (Butterworth low-pass).
* Configurable peak detection using SciPy.
* Optional Gaussian fitting for peak validation.
* Optional adaptive control to switch motion methods based on FPS/signal quality.
* OSC output for:
    * Normalized breathing signal (suitable for modulation).
    * Calculated BPM.
    * Connection status.
* Optional live UI display (using Matplotlib) showing:
    * Webcam feed with ROI overlay.
    * Normalized breathing signal plot with detected peaks.
    * Current BPM, FPS, and detection method.

## Files

* **`respiration_monitor.py`**: Contains the core logic classes:
    * `VideoInput`: Handles webcam access.
    * `Calibration`: Implements automatic ROI detection algorithms.
    * `MotionDetector`: Implements ROI Averaging and Optical Flow methods.
    * `SignalProcessor`: Filters the signal, detects peaks, calculates BPM.
    * `AdaptiveController`: Handles optional dynamic method switching.
    * `RespirationMonitor`: Encapsulates the entire processing pipeline.
    * `DEFAULT_CONFIG`: Default settings for the core monitor.
* **`main.py`**: The main executable script:
    * Handles application startup and shutdown.
    * Manages the manual ROI selection phase using OpenCV.
    * Creates and runs the `RespirationMonitor`.
    * Initializes and manages the optional Matplotlib UI (`UIManager`).
    * Initializes and manages the OSC client for sending data.
    * Contains `OSC_CONFIG` and `UI_CONFIG` for output settings.

## Setup and Installation

1.  **Python:** Ensure you have Python 3 installed.
2.  **Dependencies:** Install the required Python libraries using pip:
    ```bash
    pip install opencv-python numpy scipy matplotlib python-osc
    ```
3.  **Webcam:** Make sure you have a webcam connected and recognized by your operating system. The script defaults to using the camera at index 0.
4.  **Files:** Place `main.py` and `respiration_monitor.py` in the same directory.

## Running the Application

1.  Navigate to the directory containing the files in your terminal or command prompt.
2.  Run the main script:
    ```bash
    python main.py
    ```

## Usage

1.  **ROI Selection:**
    * An OpenCV window titled "Select ROI" will appear, showing your webcam feed.
    * Follow the on-screen instructions (and check the console output).
    * Press the **'s'** key to enable drawing mode (the indicator circle next to the instruction should turn green).
    * **Click and drag** your mouse on the video feed to draw a rectangle over the area you want to monitor (e.g., your chest).
    * Release the mouse button. A green rectangle shows the selected area.
    * If you are satisfied, press **'c'** or **Enter** to confirm (the indicator circle next to the confirm instruction should be green).
    * If you want to redraw, press **'s'** again to reset.
    * Press **'q'** at any time to quit the application.
2.  **Monitoring:**
    * Once the ROI is confirmed, the selection window closes.
    * If the UI is enabled (`UI_CONFIG['ENABLE_UI'] = True` in `main.py`), a Matplotlib window will appear with two plots:
        * **Top Plot:** Shows the live webcam feed with the selected ROI drawn as a green rectangle. It also displays the current calculated BPM, processing FPS, and the active motion detection method.
        * **Bottom Plot:** Shows the normalized breathing signal over time (roughly -2 to +2 range). Detected peaks are marked with red 'x's.
    * If OSC is enabled (`OSC_CONFIG['ENABLE_OSC'] = True` in `main.py`), the script will continuously send OSC messages containing the BPM and the normalized signal value to the configured IP address and port.
3.  **Stopping:**
    * Close the Matplotlib UI window.
    * Alternatively, press **Ctrl+C** in the terminal where the script is running.

## Configuration Parameter Details (`configs/*.json`)

This section explains the parameters you can adjust in the JSON configuration files located in the `configs/` directory. These settings control the behavior of the `respiration_monitor.py` module.

---

### General Settings

* **`VIDEO_SOURCE`**:
    * **Purpose**: Specifies the video input source.
    * **Values**:
        * `0`: Default webcam.
        * `1`, `2`, etc.: Other connected webcams (index might vary).
        * `"path/to/your/video.mp4"`: Path to a video file for offline processing or testing.
    * **Range**: Integer index or string path.

* **`USE_OPENCL`**:
    * **Purpose**: Enables/disables attempting to use OpenCL (via OpenCV's UMat) for potential GPU acceleration of certain OpenCV operations (filtering, pyramids, etc.).
    * **Values**: `true` / `false`.
    * **Tuning**: Set to `true` only if you have compatible hardware and drivers, and have profiled to confirm a performance benefit. Often introduces overhead, so `false` is usually safer unless specifically optimized for.

* **`DEBUG_PRINT_VALUES`**:
    * **Purpose**: Enables extra print statements in the console showing raw and filtered signal values, useful for diagnosing numerical instability (very large numbers).
    * **Values**: `true` / `false`.
    * **Tuning**: Keep `false` for normal operation. Set `true` only when debugging issues like extreme plot scaling.

* **`DEBUG_CALIBRATION_VIZ`**:
    * **Purpose**: Shows extra OpenCV windows displaying the variance maps calculated during automatic calibration. Useful for understanding why the calibration might be choosing a specific ROI.
    * **Values**: `true` / `false`.
    * **Tuning**: Keep `false` for normal operation. Set `true` only when debugging ROI selection problems. Requires `cv2.waitKey()` calls to update, which might slightly affect timing if left on.

---

### Calibration Settings (Used for Automatic ROI Detection)

* **`CALIBRATION_METHOD`**:
    * **Purpose**: Selects the algorithm used for automatic ROI detection if a manual ROI isn't provided.
    * **Values**:
        * `"MotionVariance"`: Simpler, faster method based on variance of frame differences. Often good enough.
        * `"SimplifiedEVM"`: Attempts temporal filtering on a pyramid level to find areas with motion in the breathing frequency range. More complex, potentially more specific but slower.
    * **Tuning**: Start with `"MotionVariance"`. Try `"SimplifiedEVM"` if Motion Variance consistently fails to find a good ROI despite good lighting/stillness.

* **`CALIBRATION_DURATION_FRAMES`**:
    * **Purpose**: How many initial frames to analyze for automatic calibration.
    * **Values**: Integer (e.g., 150, 180, 200).
    * **Range**: Typically 100-300. Longer duration allows for more data but increases startup time. Needs to be long enough to capture several breathing cycles (e.g., 5-10 seconds worth of frames: `FPS * seconds`).

* **`EVM_PYRAMID_LEVEL`**: (Used only if `CALIBRATION_METHOD` is `"SimplifiedEVM"`)
    * **Purpose**: Specifies which level of the Gaussian image pyramid to analyze for temporal filtering. Lower levels (e.g., 0, 1) have higher resolution but more noise/detail. Higher levels (e.g., 2, 3) are smaller, smoother, and might capture larger-scale motion better, but lose fine detail.
    * **Values**: Integer (e.g., 1, 2, 3).
    * **Range**: Usually 1 to 4. `2` is often a reasonable starting point.

* **`EVM_LOW_FREQ_HZ` / `EVM_HIGH_FREQ_HZ`**: (Used only if `CALIBRATION_METHOD` is `"SimplifiedEVM"`)
    * **Purpose**: Defines the frequency band (in Hz) to isolate during the temporal filtering step of the EVM-like calibration. Aims to amplify motion within the expected breathing rate range.
    * **Values**: Float (Hz).
    * **Range**:
        * `EVM_LOW_FREQ_HZ`: Should be slightly below the slowest expected breathing rate (e.g., 10 BPM = 0.16 Hz, so maybe 0.15 or 0.2).
        * `EVM_HIGH_FREQ_HZ`: Should be slightly above the fastest expected breathing rate (e.g., 90 BPM = 1.5 Hz, so maybe 1.8 or 2.0). Must be less than `FPS / 2`.
    * **Tuning**: Adjust based on the expected breathing range of the subject. Too narrow might miss the signal; too wide might include noise.

* **`ROI_REFINE_SIZE`**:
    * **Purpose**: Whether to perform an extra step after automatic calibration to test slightly different ROI sizes around the automatically selected center and choose the one yielding the highest initial signal variance.
    * **Values**: `true` / `false`.
    * **Tuning**: Can potentially improve signal quality but adds a small amount of time to the calibration process. Try `true` if the automatically selected ROI seems slightly off in size.

---

### Motion Detection Settings

* **`DEFAULT_MOTION_METHOD`**:
    * **Purpose**: The motion detection method used *after* calibration/manual ROI selection. Can be overridden by adaptive control if enabled.
    * **Values**: `"ROI_Average"`, `"OpticalFlow"`.
    * **Tuning**: Start with `"ROI_Average"` for speed. Use `"OpticalFlow"` if ROI average gives a poor signal (e.g., low contrast area) or if more robustness to minor non-breathing movements within the ROI is needed (at the cost of performance).

* **`ROI_AVG_PREFILTER`**: (Used only if `DEFAULT_MOTION_METHOD` is `"ROI_Average"`)
    * **Purpose**: Type of image filter applied to the ROI *before* calculating the average pixel value. Aims to reduce noise.
    * **Values**:
        * `"Median"`: Good for salt-and-pepper noise, preserves edges reasonably well. Often a good default.
        * `"Gaussian"`: Good for general Gaussian noise, but blurs edges more.
        * `null` (JSON `null`): No pre-filtering applied.
    * **Tuning**: `"Median"` is often best. Try `"Gaussian"` or `null` if Median seems to distort the signal.

* **`ROI_AVG_MEDIAN_KSIZE`**: (Used only if `ROI_AVG_PREFILTER` is `"Median"`)
    * **Purpose**: Kernel size (must be odd) for the median filter. Larger kernels remove more noise but are slower and can blur more.
    * **Values**: Odd integer (e.g., 3, 5, 7).
    * **Range**: 3 to 9 is typical. `5` is a common starting point.

* **`ROI_AVG_GAUSSIAN_KERNEL`**: (Used only if `ROI_AVG_PREFILTER` is `"Gaussian"`)
    * **Purpose**: Kernel size (width, height - usually odd) for the Gaussian filter.
    * **Values**: List of two odd integers (e.g., `[3, 3]`, `[5, 5]`).
    * **Range**: `[3, 3]` or `[5, 5]` are common.

* **`OPTICAL_FLOW_PARAMS`**: (Used only if `DEFAULT_MOTION_METHOD` is `"OpticalFlow"`)
    * **Purpose**: Contains parameters for the feature detection and tracking steps. Tuning these is crucial for Optical Flow performance and accuracy.
    * **`feature_params`**: Parameters for `cv2.goodFeaturesToTrack`:
        * `"maxCorners"` (int): Max number of corners to find. Fewer corners = faster tracking, but potentially less robust. (Range: 50-200).
        * `"qualityLevel"` (float): Threshold for corner quality (0.0-1.0). Higher values = fewer, stronger corners. (Range: 0.1-0.5 is often good for tracking).
        * `"minDistance"` (int): Minimum pixel distance between detected corners. Higher values = better spatial distribution. (Range: 5-15 pixels).
        * `"blockSize"` (int): Size of the neighborhood for corner detection. (Usually 3, 5, or 7).
    * **`lk_params`**: Parameters for `cv2.calcOpticalFlowPyrLK`:
        * `"winSize"` (list `[w, h]`): Size of the search window at each pyramid level. Smaller = faster, sensitive to fine motion, less robust to noise/large local motion. Larger = slower, more robust, less sensitive. (Range: `[11, 11]` to `[31, 31]`). `[15, 15]` or `[21, 21]` are common.
        * `"maxLevel"` (int): Number of pyramid levels (0 = only original image). Higher levels handle larger overall motion but add cost and might blur small movements. (Range: 1-4). `2` or `3` are common.
        * `"criteria"` (list `[type_flags_int, max_iter, epsilon]`): Termination criteria for the iterative search.
            * `type_flags_int`: Combination of `cv2.TERM_CRITERIA_EPS` (2) and `cv2.TERM_CRITERIA_COUNT` (1). `3` means use both.
            * `max_iter` (int): Max number of iterations. Fewer = faster, less precise. (Range: 5-20).
            * `epsilon` (float): Minimum change threshold to stop. Smaller = more precise, slower. (Range: 0.01-0.1).
            * Example: `[3, 10, 0.03]` (Use both criteria, max 10 iterations, epsilon 0.03).

---

### Signal Processing Settings

* **`SIGNAL_BUFFER_SIZE`**:
    * **Purpose**: Number of recent motion values stored in the buffer for filtering and peak analysis. Determines the time window looked at.
    * **Values**: Integer.
    * **Range**: Should correspond to several seconds of data (e.g., `FPS * 10` for a 10-second window). 300-600 is typical for ~30 FPS. Larger buffers give smoother filtering/BPM but increase lag.

* **`LOWPASS_FILTER_CUTOFF_HZ`**:
    * **Purpose**: Cutoff frequency (Hz) for the low-pass filter applied to the raw motion signal. Aims to remove high-frequency noise (jitter, fast movements) while keeping the lower-frequency breathing signal.
    * **Values**: Float (Hz).
    * **Range**: Must be greater than 0 and less than `FPS / 2`. Typically set slightly above the highest expected *fundamental* breathing frequency (e.g., 120 BPM = 2 Hz, so maybe 2.0, 2.5, or 3.0 Hz). Lower values filter more noise but risk attenuating the breathing signal itself. (Common range: 1.0 - 3.0 Hz).

* **`LOWPASS_FILTER_ORDER`**:
    * **Purpose**: Order of the Butterworth low-pass filter. Higher orders create a steeper cutoff between passed and rejected frequencies but can introduce more phase distortion (signal delay).
    * **Values**: Integer (e.g., 2, 3, 4, 5).
    * **Range**: 3 to 6 is common. `4` is often a good balance.

* **`MAX_EXPECTED_BPM`**:
    * **Purpose**: Used to calculate the minimum expected time interval (and thus sample distance) between breaths. This helps `find_peaks` avoid detecting multiple peaks within a single breath cycle.
    * **Values**: Integer (BPM).
    * **Range**: Set slightly higher than the fastest plausible breathing rate for the subject (e.g., 90, 120, 150).

* **`PEAK_FINDING_DISTANCE_FACTOR`**:
    * **Purpose**: Multiplier applied to the minimum interval calculated from `MAX_EXPECTED_BPM`. A value of `0.5` means peaks must be separated by at least half the minimum expected interval.
    * **Values**: Float (usually <= 1.0).
    * **Range**: 0.4 to 0.8 is typical. Lower values allow detecting faster rates more easily but risk multiple detections per breath if the signal is noisy or has small bumps.

* **`PEAK_FINDING_PROMINENCE_FACTOR`**:
    * **Purpose**: Defines the minimum vertical distance a peak must rise above its surrounding baseline (relative to the signal's overall range) to be considered valid. This is a key parameter for rejecting noise spikes.
    * **Values**: Float (usually 0.0 to 1.0).
    * **Range**: Highly dependent on signal quality. Start around 0.1-0.15. Increase (e.g., 0.2, 0.3) if noise peaks are detected. Decrease (e.g., 0.08, 0.05) if true breaths are missed, but be cautious as this increases noise sensitivity.

* **`USE_GAUSSIAN_FIT`**:
    * **Purpose**: Enables/disables an extra validation step where detected peaks are fitted to a Gaussian curve. Peaks whose fitted shape (standard deviation) falls outside specified bounds are rejected.
    * **Values**: `true` / `false`.
    * **Tuning**: Adds computational cost per peak. Often unnecessary for basic BPM calculation if filtering and prominence are tuned well. Recommended to keep `false` unless you have a specific need for shape validation and have profiled the performance impact.

* **`GAUSSIAN_FIT_WINDOW_FACTOR`**: (Used only if `USE_GAUSSIAN_FIT` is `true`)
    * **Purpose**: Determines the size of the data window around a peak used for fitting, relative to the estimated peak width.
    * **Values**: Float (e.g., 2.0, 3.0, 4.0).
    * **Range**: `3` is a reasonable default.

* **`GAUSSIAN_FIT_MIN_STDDEV` / `GAUSSIAN_FIT_MAX_STDDEV`**: (Used only if `USE_GAUSSIAN_FIT` is `true`)
    * **Purpose**: Define the acceptable range for the standard deviation (width) of the fitted Gaussian curve, in samples. Peaks with fitted widths outside this range are rejected.
    * **Values**: Float (samples).
    * **Range**: Depends heavily on expected breath shape and sampling rate. Requires experimentation. (Defaults: 1.0 to 50.0).

---

### Adaptive Control Settings (Used only if `USE_ADAPTIVE_CONTROL` is `true`)

* **`USE_ADAPTIVE_CONTROL`**:
    * **Purpose**: Enables/disables the experimental feature that automatically switches between `ROI_Average` and `OpticalFlow` based on performance (FPS) and signal quality (BPM stability).
    * **Values**: `true` / `false`.
    * **Tuning**: Keep `false` unless you specifically want to test this feature. The logic is basic and may require significant tuning of thresholds.

* **`ADAPTIVE_FPS_THRESHOLD`**:
    * **Purpose**: If FPS drops below this threshold while using `OpticalFlow`, the controller may switch to the faster `ROI_Average` method.
    * **Values**: Integer (FPS).
    * **Range**: Depends on your target real-time performance (e.g., 10, 15, 20).

* **`ADAPTIVE_BPM_STABILITY_THRESHOLD`**:
    * **Purpose**: If the standard deviation of recent BPM calculations exceeds this threshold while using `ROI_Average` (indicating poor signal quality) AND the FPS is high enough, the controller may switch to the potentially more robust `OpticalFlow` method.
    * **Values**: Float (BPM standard deviation).
    * **Range**: Depends on acceptable BPM variability (e.g., 5, 10, 15). Lower values mean less variability is tolerated before switching.

* **`ADAPTIVE_HYSTERESIS_FRAMES`**:
    * **Purpose**: The minimum number of frames that must pass after a method switch before the controller will consider switching again. Prevents rapid oscillation between methods.
    * **Values**: Integer (frames).
    * **Range**: Typically 1-3 seconds worth of frames (e.g., 30-90 for 30 FPS).

---


## OSC / Ableton Live Setup (using Max for Live)

1.  **Install `python-osc`:** Make sure the library is installed (`pip install python-osc`).
2.  **Configure `main.py`:** Set `ENABLE_OSC = True` and configure the correct IP address and port. Note the OSC addresses used.
3.  **Create M4L Device:** Create a Max for Live Audio or MIDI Effect device in Ableton Live.
4.  **Add Max Objects:**
    * `udpreceive <port>`: (e.g., `udpreceive 9001`) - Listens for incoming OSC data.
    * `route /respmon/signal /respmon/bpm`: Separates messages based on address.
    * `flonum`: (Optional) Connect to `route` outlets to view incoming values.
    * `scale~ <in_min> <in_max> <out_min> <out_max>`: (Optional but recommended) Scales the incoming normalized signal (approx -2 to +2) to the desired output range (often 0.0 to 1.0 for Ableton parameters). Example: `scale~ -2. 2. 0. 1.`
    * `*~ <gain_factor>`: (Optional) Multiplies the signal after scaling if you need more amplitude. Example: `*~ 5.`
    * `curve~`: (Optional) Insert after scaling/gain to reshape the modulation curve. Double-click to edit the curve.
    * `live.remote~`: Connect the final processed signal (float, usually 0-1) to its left inlet. Use its Inspector or the "Map" button method to map it to an Ableton parameter.
    * `live.text` (Parameter Mode: `map_mode`): Creates a "Map" button. Connect its middle outlet to the *second* inlet of `live.remote~`. Add to Presentation.
    * `live.text` (Parameter Mode: `parameter_display`): (Optional) Connect the left outlet of `live.remote~` to its inlet to display the name of the mapped parameter. Add to Presentation.
5.  **Connect Objects:** Wire `udpreceive` -> `route`. Connect `route` outlets to display/processing chains. Connect the final signal output to `live.remote~`. Connect the map button to `live.remote~`.
6.  **Map Parameter:** Use the "Map" button or the `live.remote~` Inspector to link it to a knob/slider in Ableton.
7.  **Save Device:** Save your M4L patch.

Now, the Python script should control the mapped parameter in Ableton Live based on your breathing.

