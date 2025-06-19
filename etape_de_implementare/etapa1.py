import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque


TAD = 0.002  
TRA = 2  
alpha = 2  
beta = 0.1 
NUM_LARGEST_AREAS = 2  
SIGNIFICANT_AREA_THRESHOLD = 2000  
MAX_DISTANCE = 2.0  
SMOOTHING_HISTORY = 5  

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)


prev_feature_matrix = []
colors = {}

# Dictionary to store bounding box history for smoothing
bbox_history = {}

def segment_image(depth_image, depth_scale):
    # Apply distance threshold to the depth image
    max_distance_in_mm = int(MAX_DISTANCE / depth_scale)  # Convert meters to millimeters
    depth_filtered = np.where((depth_image > 0) & (depth_image < max_distance_in_mm), depth_image, 0)

    Gx = cv2.Sobel(depth_filtered, cv2.CV_16S, 1, 0, ksize=3)
    Gy = cv2.Sobel(depth_filtered, cv2.CV_16S, 0, 1, ksize=3)
    Gx, Gy = np.abs(Gx), np.abs(Gy)
    
    gradient_magnitude = cv2.addWeighted(Gx, 0.5, Gy, 0.5, 0)
    gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    threshold = 2 * np.mean(gradient_magnitude)
    _, binary_gradient = cv2.threshold(gradient_magnitude, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(binary_gradient, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - dilated_edges)
    
    feature_matrix = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        x_center, y_center = int(centroids[label][0]), int(centroids[label][1])
        bounding_box = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP], stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        
        # Calculate average depth within the region
        region_mask = (labels == label).astype(np.uint8)
        average_depth = np.mean(depth_filtered[region_mask == 1]) * depth_scale  # Convert depth back to meters

        # Only consider regions with a significant area
        if area >= SIGNIFICANT_AREA_THRESHOLD:
            feature_matrix.append([label, x_center, y_center, area, bounding_box, average_depth])
            if label not in colors:
                colors[label] = np.random.randint(0, 255, 3).tolist()
    
    return feature_matrix, labels

def smooth_bounding_box(label, bounding_box):
    """Smooth bounding box with moving average over SMOOTHING_HISTORY frames."""
    if label not in bbox_history:
        bbox_history[label] = deque(maxlen=SMOOTHING_HISTORY)
    
    # Add current bounding box to history
    bbox_history[label].append(bounding_box)
    
    # Calculate average bounding box
    x_avg = int(np.mean([box[0] for box in bbox_history[label]]))
    y_avg = int(np.mean([box[1] for box in bbox_history[label]]))
    w_avg = int(np.mean([box[2] for box in bbox_history[label]]))
    h_avg = int(np.mean([box[3] for box in bbox_history[label]]))
    
    return (x_avg, y_avg, w_avg, h_avg)

def display_segmentation(segmented_image, feature_matrix):
    output_image = np.zeros((segmented_image.shape[0], segmented_image.shape[1], 3), dtype=np.uint8)

    
    sorted_features = sorted(feature_matrix, key=lambda x: x[3], reverse=True)
    largest_regions = sorted_features[:NUM_LARGEST_AREAS]  # Select top N largest regions

    for label_info in feature_matrix:
        label, x_center, y_center, _, bounding_box, _ = label_info
        color = colors.get(label, [255, 255, 255])
        output_image[segmented_image == label] = color

    for label_info in largest_regions:
        label, x_center, y_center, area, bounding_box, average_depth = label_info
        # Mark center point
        cv2.circle(output_image, (x_center, y_center), 5, (255, 255, 255), -1)  # White dot as center point
        
        # Smooth bounding box using moving average
        smoothed_bbox = smooth_bounding_box(label, bounding_box)
        x, y, w, h = smoothed_bbox
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

    return output_image

def display_simplified_view(feature_matrix):
    """Create a simplified view with filled rectangles and depth annotations."""
    simplified_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Initialize blank image
    
    
    for label_info in feature_matrix:
        label, x_center, y_center, area, bounding_box, average_depth = label_info
        x, y, w, h = smooth_bounding_box(label, bounding_box)  # Apply smoothing for stability
        
       
        color = colors.get(label, [255, 255, 255])
        cv2.rectangle(simplified_image, (x, y), (x + w, y + h), color, -1)  # Filled rectangle
                
        depth_text = f"{average_depth:.2f} m"
        cv2.putText(simplified_image, depth_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        cv2.circle(simplified_image, (x_center, y_center), 5, (0, 0, 0), -1)  # Black dot for contrast
    
    return simplified_image

try:
    # Get depth scale from the camera
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()  # This is in meters per depth unit

    start_time = time.time()
    frame_count = 0
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        
        current_feature_matrix, labels = segment_image(depth_image, depth_scale)
        
        labeled_image = display_segmentation(labels, current_feature_matrix)

        simplified_image = display_simplified_view(current_feature_matrix)

        combined_image = np.hstack((labeled_image, simplified_image))

        frame_count += 1
        fps = frame_count / (time.time() - start_time)
        cv2.putText(combined_image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Original and Simplified Segmented Image', combined_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
