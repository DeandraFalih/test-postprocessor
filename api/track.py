import cv2
from tracker import Tracker

# Inisialisasi Tracker
tracker = Tracker(max_distance=35, max_history=30)


# Fungsi utama untuk post-processor
def process_frame(frame, detected_objects):
    """
    Process frame to assign IDs to detected vehicles and display them on the video.

    Parameters:
    - frame: np.ndarray, the current video frame
    - detected_objects: list of tuples [(x1, y1, x2, y2), ...] with bounding boxes of detected vehicles

    Returns:
    - frame: np.ndarray, the frame with vehicle IDs drawn
    """

    # Update tracker dengan bounding boxes dari objek yang terdeteksi
    tracked_objects = tracker.update(detected_objects)

    # Gambarkan bounding box dan ID di setiap objek yang dilacak
    for (x1, y1, x2, y2, object_id) in tracked_objects:
        # Gambarkan kotak pembatas
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Tampilkan ID di atas bounding box
        label = f'ID: {object_id}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


# Contoh penggunaan pada aliran video
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Contoh deteksi kendaraan - asumsi bounding box statis untuk demonstrasi
        # Untuk integrasi nyata, ini harus berasal dari detektor objek
        detected_objects = [
            (100, 200, 150, 250),  # Contoh bounding box pertama
            (300, 400, 350, 450)  # Contoh bounding box kedua
        ]

        # Proses frame untuk menambahkan ID pada setiap objek
        frame = process_frame(frame, detected_objects)

        # Tulis frame ke video output
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Jalankan pemrosesan video
input_video_path = 'input_video.mp4'
output_video_path = 'output_video_with_ids.mp4'
process_video(input_video_path, output_video_path)
