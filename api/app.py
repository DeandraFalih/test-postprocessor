from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import time
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Define the spots with their coordinates
spots = {
    'Spot1': [(865, 326), (1053, 328), (1266, 511), (981, 516)]
    # 'Spot2': [(488, 395), (658, 432), (598, 474), (420, 420)],
    # 'Spot3': [(393, 418), (591, 478), (518, 523), (347, 437)],
    # 'Spot4': [(417, 581), (510, 536), (319, 449), (265, 466)],
    # 'Spot5': [(246, 662), (96, 508), (200, 468), (400, 588)]
}

occupancy = {spot: False for spot in spots}
occupancy_frame_counts = {spot: 0 for spot in spots}
occupancy_timestamps = {spot: time.time() for spot in spots}
occupancy_durations = {spot: 0 for spot in spots}
frame_threshold = 0.35
occupancy_threshold = 0.35

# Track frame counts and midpoints
frame_count = 0

def check_occupancy(bboxes):
    global occupancy, occupancy_frame_counts, occupancy_timestamps, occupancy_durations
    inside_spot = {spot: False for spot in spots.keys()}

    for i in range(0, len(bboxes), 4):
        x_min, y_min, x_max, y_max = map(int, bboxes[i:i+4])

        # Calculate midpoint
        midpoint = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        # Check if midpoint is inside any spot polygon
        for spot, polygon in spots.items():
            mask = np.zeros((1280, 720), dtype=np.uint8)  # Updated resolution to (1280, 720)
            cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], color=(255, 255, 255))
            if mask[midpoint[1], midpoint[0]] == 255:
                inside_spot[spot] = True

    # Update occupancy based on inside_spot with frame count logic
    for spot in spots.keys():
        if inside_spot[spot]:
            occupancy_frame_counts[spot] += 1
        else:
            occupancy_frame_counts[spot] -= 1

        # Ensure frame counts stay within [0, frame_threshold]
        occupancy_frame_counts[spot] = min(max(occupancy_frame_counts[spot], 0), frame_threshold)

        # Update the actual occupancy status based on the threshold
        if occupancy_frame_counts[spot] >= occupancy_threshold and not occupancy[spot]:
            occupancy[spot] = True
            occupancy_durations[spot] = time.time() - occupancy_timestamps[spot]
            occupancy_timestamps[spot] = time.time()
        elif occupancy_frame_counts[spot] < occupancy_threshold and occupancy[spot]:
            occupancy[spot] = False
            occupancy_durations[spot] = time.time() - occupancy_timestamps[spot]
            occupancy_timestamps[spot] = time.time()

@app.route('/')
def index():
    return render_template('index3.html', occupancy=occupancy, occupancy_durations=occupancy_durations)

@app.route('/add_message', methods=['POST'])
def add_message():
    global occupancy
    message = request.json
    for key, value in message['BBoxes_xyxy'].items():
        if key != 'ROI':
            check_occupancy(value)
    return '', 204

@app.route('/occupancy', methods=['GET'])
def get_occupancy():
    return jsonify({
        'occupancy': occupancy,
        'empty_spots': sum(1 for occupied in occupancy.values() if not occupied),
        'filled_spots': sum(1 for occupied in occupancy.values() if occupied),
        'occupancy_durations': occupancy_durations
    })

if __name__ == '__main__':
    app.run(debug=True)
