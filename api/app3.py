from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
import time
import os

app = Flask(__name__)

# Konfigurasi direktori penyimpanan untuk gambar tangkapan
output_dir = 'captured_frames'
os.makedirs(output_dir, exist_ok=True)

# Inisialisasi variabel dan konfigurasi
CONFIDENCE_THRESHOLD = 0.35
ALERT_DURATION = 2
frame_threshold = 5
occupancy_threshold = 3

# Definisikan area poligon ROI
polygon_points = [(865, 326), (1053, 328), (1266, 511), (981, 516)]
inside_polygon_start_time = {}
occupancy_durations = {}

# Inisialisasi status dan data occupancy
occupancy = False
occupancy_frame_count = 0
occupancy_start_time = time.time()

# Fungsi untuk memeriksa apakah titik tengah mobil ada dalam poligon
def is_car_in_polygon(x_min, y_min, x_max, y_max, polygon):
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return cv2.pointPolygonTest(np.array(polygon, np.int32), (center_x, center_y), False) >= 0

# Fungsi untuk memperbarui occupancy berdasarkan bounding box yang terdeteksi
def check_occupancy(bboxes):
    global occupancy, occupancy_frame_count, occupancy_start_time, occupancy_durations

    for i in range(0, len(bboxes), 4):
        x_min, y_min, x_max, y_max = map(int, bboxes[i:i+4])

        # Periksa apakah titik tengah mobil berada dalam poligon ROI
        if is_car_in_polygon(x_min, y_min, x_max, y_max, polygon_points):
            occupancy_frame_count += 1
        else:
            occupancy_frame_count -= 1

        # Batasi frame count agar berada di rentang [0, frame_threshold]
        occupancy_frame_count = min(max(occupancy_frame_count, 0), frame_threshold)

        # Perbarui status occupancy berdasarkan threshold frame
        if occupancy_frame_count >= occupancy_threshold and not occupancy:
            occupancy = True
            occupancy_durations[time.time()] = 0  # Mulai waktu durasi baru
            occupancy_start_time = time.time()
        elif occupancy_frame_count < occupancy_threshold and occupancy:
            occupancy = False
            occupancy_durations[time.time()] = time.time() - occupancy_start_time

# Endpoint untuk menerima data bounding box dari Nx Meta
@app.route('/add_message', methods=['POST'])
def add_message():
    global occupancy
    message = request.json
    for key, value in message['BBoxes_xyxy'].items():
        if key != 'ROI':
            check_occupancy(value)
    return '', 204

# Endpoint untuk memberikan status occupancy
@app.route('/occupancy', methods=['GET'])
def get_occupancy():
    return jsonify({
        'occupancy': occupancy,
        'occupancy_durations': occupancy_durations
    })

# Endpoint untuk heatmap (jika diperlukan fitur heatmap)
@app.route('/heatmap', methods=['GET'])
def get_heatmap():
    # Ini contoh kosong. Anda bisa menambahkan logika untuk heatmap jika dibutuhkan.
    return send_file('static/heatmap.jpg', mimetype='image/jpeg')

# Mulai server Flask
if __name__ == '__main__':
    app.run(debug=True)
