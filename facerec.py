import cv2
import face_recognition
import mysql.connector
from datetime import datetime
import csv
import threading

# Inisialisasi kamera
video_capture = cv2.VideoCapture(0)  # 0 untuk kamera laptop, 1 untuk kamera eksternal
video_capture.set(3, 1080) # Atur lebar
video_capture.set(4, 720) # Atur tinggi

# Inisialisasi Cascade Classifier untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Koneksi ke database
db = mysql.connector.connect(
    host = "localhost",
    user = "root",
    passwd = "",
    database = "face"
)

# Cursor
cursor = db.cursor()

# Ambil data pengguna dari database
cursor.execute("SELECT * FROM users")
rows = cursor.fetchall()

# Inisialisasi list wajah yang dikenali
known_face_names = []
known_face_encodings = []

for r in rows:
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("images/"+r[1]))[0]) # Proses encoding
    known_face_names.append(r[0]) # Nama orang

people = known_face_names.copy() # Simpan nama orang

# Ambil tanggal sekarang
current_date = datetime.now().strftime("%Y-%m-%d")

# Buat file CSV
f = open(current_date +'.csv', 'w+', newline = '')
lnwriter = csv.writer(f)

# Set untuk melacak wajah yang telah terdeteksi
detected_people = set()

# Fungsi untuk membaca frame dari kamera dan deteksi wajah
def process_frames():
    global video_capture, known_face_encodings, known_face_names, detected_people
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Gagal membaca frame, keluar...")
            break

        frame = cv2.flip(frame, 1) # Mirror

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengubah dari BGR (format openCV) ke RGB (format face_recognition)

        face_locations = face_recognition.face_locations(rgb_frame) # Mendeteksi lokasi wajah dalam gambar 
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations) # Mengambil encoding dari wajah yang terdeteksi

        # Proses pengenalan wajah dan tampilan
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding) # Membandingkan encoding wajah yang terdeteksi dengan encoding wajah yang sudah dikenali 
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if name != "Unknown":  # Jika wajah dikenali
                if name not in detected_people:  # Hanya tulis masuk jika wajah belum terdeteksi
                    current_time = datetime.now().strftime("%H:%M:%S")
                    lnwriter.writerow([name, current_time])
                    detected_people.add(name)  # Tambahkan wajah ke set deteksi

            # Tampilkan kotak dan nama wajah
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)  
                cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

            else:   
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)  
                cv2.putText(frame, name, (left, bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Thread untuk membaca frame dan deteksi wajah
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

# Tunggu sampai thread selesai
processing_thread.join()

# Tutup koneksi dan file
video_capture.release()
cv2.destroyAllWindows()
f.close()