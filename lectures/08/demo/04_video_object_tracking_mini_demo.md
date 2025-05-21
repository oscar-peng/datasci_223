# Mini-Demo: Introduction to Video Object Tracking

**Objective:** Briefly introduce video as a sequence of images and demonstrate two conceptual approaches to tracking objects across frames: detection-based tracking and optical flow-based tracking. This is a "show and tell" demo due to time constraints, focusing on visual understanding rather than deep implementation details.

**Dataset:** A short sample video clip.
*   **Action:** Download a sample video. A general-purpose video is fine for demonstrating the techniques.
    *   You can use `wget` in your terminal:
        ```bash
        wget https://videos.pexels.com/video-files/1572386/1572386-sd_640_360_30fps.mp4 -O sample_video.mp4
        ```
    *   Or download it manually from a source like Pexels/Pixabay and save it as `sample_video.mp4` in the same directory as this notebook.
*   For a health context, if a short, simple clip of movement (e.g., hand motion, a person walking in a clinic hallway, instrument movement if clear) is available, that could be used. The current code assumes `sample_video.mp4`.

**Tools:** Python, OpenCV, Ultralytics YOLO (for detection-based part).


## 1. Setup and Imports

Ensure `ultralytics` and `opencv-python` are installed. If not, run in your terminal or a code cell:
```bash
pip install ultralytics opencv-python requests
```
(Requests might have been installed by ultralytics, but good to ensure).

We'll import the necessary libraries:
*   `cv2` (OpenCV) for video processing, drawing, and display.
*   `numpy` for numerical operations (though minimally used directly here).
*   `matplotlib.pyplot` for an optional helper function (though `cv2.imshow` is primary for video).
*   `ultralytics.YOLO` for the pre-trained object detector.
*   `time` for potential small delays if needed for smoother visualization in some environments (not strictly used in current `cv2.waitKey`).

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt # For optional helper
from ultralytics import YOLO
import time # For a simple delay if needed for visualization
import requests # To download sample video if needed (manual download preferred for stability)
from io import BytesIO # To handle byte stream for video

# --- Optional: Helper function to display a single frame using Matplotlib ---
# (Primarily we will use cv2.imshow for real-time video display)
def show_frame_plt(frame_bgr, title='Frame'):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8,6))
    plt.imshow(frame_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()
```


## 2. Approach 1: Detection-based Tracking (e.g., YOLO frame-by-frame)

**Concept:**
This approach involves running an object detector (like YOLO) on each frame of the video independently. Once objects are detected in each frame (with bounding boxes and class labels), a subsequent step (which we won't implement in detail here) would be to *associate* these detections across frames to form individual object tracks. This association can be done using various techniques, such as:
*   **IoU (Intersection over Union) Matching:** Comparing bounding boxes in consecutive frames.
*   **Kalman Filters:** Predicting an object's next position and matching detections to predictions.
*   **More Advanced Trackers:** Algorithms like SORT (Simple Online and Realtime Tracking) or DeepSORT (which adds appearance information using deep learning).

For this mini-demo, we will focus only on the **frame-by-frame detection part** to visualize how a detector sees objects in a video sequence.

```python
# --- Load Pre-trained YOLO Model ---
yolo_model_instance_video = None
try:
    yolo_model_instance_video = YOLO('yolov8n.pt') # Nano model for speed
    print("YOLO model for video loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

# --- Video Path ---
# Ensure 'sample_video.mp4' is in the same directory as this notebook,
# or provide the full path to your video file.
video_path_yolo = 'sample_video.mp4' 

cap_yolo_video = cv2.VideoCapture(video_path_yolo)

if not cap_yolo_video.isOpened():
    print(f"Error: Could not open video file '{video_path_yolo}' for YOLO demo.")
    if yolo_model_instance_video is None: print("YOLO model also failed to load.")
else:
    if yolo_model_instance_video is None:
        print("YOLO model not loaded. Cannot proceed with detection demo.")
    else:
        print("\nStarting Detection-based Tracking Demo (YOLO frame-by-frame)...")
        print("Press 'q' in the video window to quit early.")
        frame_count_yolo = 0
        max_frames_to_show_yolo = 150 # Limit frames for a quick demo, e.g., 5 seconds at 30fps

        while cap_yolo_video.isOpened() and frame_count_yolo < max_frames_to_show_yolo:
            ret, frame = cap_yolo_video.read() # Read a frame
            if not ret: # If no frame is returned (end of video or error)
                print("End of video or cannot read frame.")
                break

            # Perform detection on the current frame
            # 'verbose=False' reduces the amount of console output from YOLO.
            results = yolo_model_instance_video(frame, verbose=False) 
            
            # The 'results' object contains detections.
            # 'results[0].plot()' is a handy method from Ultralytics that
            # draws the bounding boxes, labels, and confidences directly onto the frame.
            annotated_frame_yolo = results[0].plot() 
            
            cv2.imshow('YOLO Detection (Frame-by-Frame)', annotated_frame_yolo)
            
            frame_count_yolo += 1
            # Wait for 30ms. If 'q' is pressed, break the loop.
            if cv2.waitKey(30) & 0xFF == ord('q'): 
                print("User pressed 'q'. Exiting YOLO demo.")
                break
                
        cap_yolo_video.release() # Release the video capture object
        cv2.destroyAllWindows() # Close all OpenCV windows
        print("Detection-based Tracking Demo (YOLO) finished.")
```
**Expected Visual Output (Approach 1):**
A video window titled "YOLO Detection (Frame-by-Frame)" will open.
*   The video will play (or the first `max_frames_to_show_yolo` frames).
*   In each frame, objects that YOLO can detect (based on its COCO training, e.g., 'person', 'car', 'bus' in the sample video) will be enclosed in bounding boxes.
*   Each box will have a label (the object class) and a confidence score.
*   You'll observe how detections might vary slightly frame-to-frame (e.g., confidence scores changing, occasional missed detections or new detections). This highlights why a separate tracking algorithm is needed to link these detections into smooth tracks.


## 3. Approach 2: Optical Flow-based Tracking (Lucas-Kanade)

**Concept:**
Optical flow estimates the motion of image pixels or features between consecutive frames. The Lucas-Kanade method is a popular technique for tracking a sparse set of feature points (like corners).
Unlike detection-based methods, optical flow doesn't inherently identify *what* the objects are. Instead, it tracks the movement of salient points. This can be useful for understanding motion patterns or for tracking objects once they've been initially identified by other means.

We will:
1.  Detect good features to track (e.g., corners using Shi-Tomasi algorithm) in the first frame.
2.  For each subsequent frame, calculate the new positions of these features using Lucas-Kanade optical flow.
3.  Draw lines (tracks) showing the movement of these points.

```python
# --- Video Path (can use the same video) ---
video_path_flow = 'sample_video.mp4' # Defined above

cap_flow_video = cv2.VideoCapture(video_path_flow)

if not cap_flow_video.isOpened():
    print(f"Error: Could not open video file '{video_path_flow}' for Optical Flow demo.")
else:
    print("\nStarting Optical Flow-based Tracking Demo (Lucas-Kanade)...")
    print("Press 'q' in the video window to quit early.")
    
    # Parameters for ShiTomasi corner detection (finding good features to track)
    feature_params = dict(maxCorners=100,    # Max number of corners to detect
                         qualityLevel=0.3,  # Minimal accepted quality of image corners
                         minDistance=7,     # Minimum possible Euclidean distance between corners
                         blockSize=7)       # Size of an average block for computing a derivative covariation matrix
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), # Size of the search window at each pyramid level
                     maxLevel=2,       # 0-based maximal pyramid level number
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                     # Termination criteria: stop after 10 iterations or if accuracy is better than 0.03
    
    # Create some random colors for drawing the tracks
    track_colors = np.random.randint(0, 255, (100, 3)) # For up to 100 tracks
    
    # Take the first frame and find corners in it
    ret_old, old_frame_flow = cap_flow_video.read()
    if not ret_old:
        print("Error: Could not read the first frame for Optical Flow.")
    else:
        old_gray_flow = cv2.cvtColor(old_frame_flow, cv2.COLOR_BGR2GRAY)
        # Detect initial strong corners using Shi-Tomasi
        p0_flow = cv2.goodFeaturesToTrack(old_gray_flow, mask=None, **feature_params)
        
        # Create a mask image for drawing tracks
        # This mask will accumulate the track lines.
        flow_mask = np.zeros_like(old_frame_flow)
        
        frame_count_flow_demo = 0
        max_frames_to_show_flow_demo = 200 # Limit frames for a quick demo

        while cap_flow_video.isOpened() and frame_count_flow_demo < max_frames_to_show_flow_demo:
            ret_new, frame_flow = cap_flow_video.read()
            if not ret_new:
                print("End of video or cannot read frame for optical flow.")
                break
            
            if p0_flow is not None and len(p0_flow) > 0: # If we have points to track
                frame_gray_flow = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow using Lucas-Kanade
                # p1: calculated new positions of input features in the current frame
                # st: status vector (1 if flow for the corresponding feature has been found, else 0)
                # err: error vector
                p1_flow, st_flow, err_flow = cv2.calcOpticalFlowPyrLK(old_gray_flow, frame_gray_flow, p0_flow, None, **lk_params)
                
                # Select good points (where flow was found)
                if p1_flow is not None and st_flow is not None:
                    good_new_points = p1_flow[st_flow == 1]
                    good_old_points = p0_flow[st_flow == 1]
                
                    # Draw the tracks
                    for i, (new_pt, old_pt) in enumerate(zip(good_new_points, good_old_points)):
                        a, b = new_pt.ravel().astype(int)
                        c, d = old_pt.ravel().astype(int)
                        # Draw line on the mask from old point to new point
                        flow_mask = cv2.line(flow_mask, (a, b), (c, d), track_colors[i % 100].tolist(), 2)
                        # Draw a circle at the new point's position on the current frame
                        frame_flow = cv2.circle(frame_flow, (a, b), 5, track_colors[i % 100].tolist(), -1)
                
                # Display the current frame with tracks overlaid
                img_display_flow = cv2.add(frame_flow, flow_mask)
                cv2.imshow('Optical Flow Tracking (Lucas-Kanade)', img_display_flow)

                # Update the previous frame and previous points for the next iteration
                old_gray_flow = frame_gray_flow.copy()
                p0_flow = good_new_points.reshape(-1, 1, 2) if 'good_new_points' in locals() and len(good_new_points) > 0 else None
            else:
                # If no points are being tracked (e.g., all lost), re-detect on current frame
                print("No points to track, re-detecting features...")
                old_gray_flow = cv2.cvtColor(frame_flow, cv2.COLOR_BGR2GRAY)
                p0_flow = cv2.goodFeaturesToTrack(old_gray_flow, mask=None, **feature_params)
                flow_mask = np.zeros_like(old_frame_flow) # Reset mask
                cv2.imshow('Optical Flow Tracking (Lucas-Kanade)', frame_flow) # Show current frame

            frame_count_flow_demo += 1
            if cv2.waitKey(30) & 0xFF == ord('q'):
                print("User pressed 'q'. Exiting Optical Flow demo.")
                break

        cap_flow_video.release()
        cv2.destroyAllWindows()
        print("Optical Flow-based Tracking Demo finished.")
```
**Expected Visual Output (Approach 2):**
A video window titled "Optical Flow Tracking (Lucas-Kanade)" will open.
*   The video will play.
*   Initially, salient feature points (corners) will be detected in the first frame.
*   In subsequent frames, these points will be tracked. Small circles will indicate their current positions.
*   Lines (tracks) will be drawn on a mask and overlaid on the video, showing the path these points have taken over time.
*   If points are lost (e.g., go out of frame or become occluded), they will disappear. If all points are lost, the code attempts to re-detect new features.
*   This demo visually shows how groups of pixels that move together can be tracked.


**Self-Check / Validation:**
*   Were you able to download/provide a sample video file and did it load correctly for both demos?
*   **For Approach 1 (YOLO Detection-based):**
    *   Did the video play in an OpenCV window?
    *   Were objects detected by YOLO and visualized with bounding boxes on each frame?
*   **For Approach 2 (Optical Flow-based):**
    *   Did the video play in an OpenCV window?
    *   Were feature points detected and tracked across frames?
    *   Were track lines visible, showing the motion paths of these points?
*   Remember to press 'q' in the active OpenCV window to close it and proceed or end the demo. If windows don't close automatically, you might need to manually ensure `cv2.destroyAllWindows()` is effective in your environment or restart the kernel.