import cv2
import face_recognition

# Inisialisasi kamera
video_capture = cv2.VideoCapture(0)  # 0 untuk kamera laptop, 1 untuk kamera eksternal

while True:
    # Baca setiap frame dari kamera
    ret, frame = video_capture.read()
    
    # Konversi frame dari BGR (format OpenCV) ke RGB (format face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi lokasi wajah dalam frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Loop melalui setiap lokasi wajah
    for (top, right, bottom, left) in face_locations:
        # Gambar kotak di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
    # Tampilkan frame hasil dengan kotak di sekitar wajah
    cv2.imshow('Video', frame)
    
    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Setelah selesai, lepaskan objek video_capture dan tutup semua jendela OpenCV
video_capture.release()
cv2.destroyAllWindows()
