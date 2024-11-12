# main.py
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import time
import os
from tracker import Tracker  # Import the Tracker class

app = Flask(__name__)

# Define the polygonal zones and initialize state tracking
polygon_zones = {
    'Zone1': [(100, 100), (300, 100), (300, 300), (100, 300)],
}
occupancy = {zone: False for zone in polygon_zones}
occupancy_start_time = {zone: 0 for zone in polygon_zones}
alert_duration = 2  # seconds
output_dir = 'captured_frames'
os.makedirs(output_dir, exist_ok=True)

# Threshold settings for frame counts and occupancy
occupancy_frame_counts = {zone: 0 for zone in polygon_zones}
frame_threshold = 5
occupancy_threshold = 3

# Initialize Tracker
tracker = Tracker(max_distance=35, max_history=30)

def is_vehicle_in_zone(x_min, y_min, x_max, y_max, polygon):
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (center_x, center_y), False) >= 0

def check_occupancy(bboxes):
    global occupancy, occupancy_frame_counts
    inside_zone = {zone: False for zone in polygon_zones.keys()}

    # Update tracker and get objects with IDs
    tracked_objects = tracker.update(bboxes)

    for obj in tracked_objects:
        x_min, y_min, x_max, y_max, obj_id = obj

        for zone, polygon in polygon_zones.items():
            if is_vehicle_in_zone(x_min, y_min, x_max, y_max, polygon):
                inside_zone[zone] = True
                print(f"Vehicle ID {obj_id} is in {zone}")

    # Update occupancy based on inside_zone with frame count logic
    for zone in polygon_zones.keys():
        if inside_zone[zone]:
            occupancy_frame_counts[zone] += 1
        else:
            occupancy_frame_counts[zone] -= 1

        # Ensure frame counts stay within [0, frame_threshold]
        occupancy_frame_counts[zone] = min(max(occupancy_frame_counts[zone], 0), frame_threshold)

        # Update actual occupancy status based on threshold
        if occupancy_frame_counts[zone] >= occupancy_threshold and not occupancy[zone]:
            occupancy[zone] = True
            occupancy_start_time[zone] = time.time()
        elif occupancy_frame_counts[zone] < occupancy_threshold and occupancy[zone]:
            occupancy[zone] = False
            occupancy_start_time[zone] = time.time()

        # Capture an image if a vehicle stays in the zone for the alert duration
        if occupancy[zone] and time.time() - occupancy_start_time[zone] > alert_duration:
            capture_filename = os.path.join(output_dir, f"{zone}_{int(time.time())}.jpg")
            cv2.imwrite(capture_filename, np.zeros((720, 960, 3), np.uint8))
            print(f"Captured image saved as {capture_filename}")

@app.route('/')
def index():
    return render_template('index2.html', occupancy=occupancy)

@app.route('/add_message', methods=['POST'])
def add_message():
    message = request.json
    for key, value in message['BBoxes_xyxy'].items():
        if key != 'ROI':
            check_occupancy(value)
    return '', 204

@app.route('/occupancy', methods=['GET'])
def get_occupancy():
    return jsonify({
        'occupancy': occupancy,
        'empty_zones': sum(1 for occupied in occupancy.values() if not occupied),
        'filled_zones': sum(1 for occupied in occupancy.values() if occupied)
    })

if __name__ == '__main__':
    app.run(debug=True)
