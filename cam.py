import cv2
import numpy as np

# Definisikan rentang warna yang ingin dideteksi
#lower_red = np.array([0, 100, 100]) 
lower_red = np.array([0, 179, 179]) # 0, 70%, 70%
upper_red = np.array([10, 255, 255]) # 10, 100%, 100%

# Warna hijau harus mengikut hex #0E7560
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

maximum_width = 200
maximum_height = 200

jarak_deket_middle = 50

# Buat objek VideoCapture untuk mengakses video dari kamera
video = cv2.VideoCapture(0)

# Definisikan label objek yang ingin ditampilkan
object_red = "Merah_0033"
object_green = "Hijau_0105"

if not video.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Baca frame dari video
    ret, frame = video.read()

    frame_width = frame.shape[1]
    frame_height =  frame.shape[0]

    middle_line_x = frame_width // 2
    middle_line_y = frame_height // 2

    lower_middle = middle_line_x - jarak_deket_middle
    upper_middle = middle_line_x + jarak_deket_middle

    # Memeriksa apakah frame terbaca dengan benar
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Ubah frame ke dalam ruang warna HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Buat mask dengan rentang warna yang ditentukan
    maskR = cv2.inRange(frame_hsv, lower_red, upper_red)
    maskG = cv2.inRange(frame_hsv, lower_green, upper_green)

    # Lakukan operasi morfologi untuk membersihkan noise
    kernel = np.ones((50, 50), np.uint8)
    maskR = cv2.morphologyEx(maskR, cv2.MORPH_OPEN, kernel)
    maskG = cv2.morphologyEx(maskG, cv2.MORPH_OPEN, kernel)

    # Temukan kontur objek yang dideteksi
    contoursR, _ = cv2.findContours(maskR, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursG, _ = cv2.findContours(maskG, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merah_melewati_batas = False
    hijau_melewati_batas = False

    # Gambar persegi dan berikan nama objek di sekitar objek yang dideteksi
    for contourRed in contoursR:
        x, y, w, h = cv2.boundingRect(contourRed)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        percentage = ((w*h)/(maximum_width*maximum_height)) * 100 # Percentage of a color of how close it is to the camera
        cv2.putText(frame, f"{object_red} [{str(percentage)}]", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        teks_perintah = "Lurus"

        # Di kanan
        if x > upper_middle:
            teks_perintah = "Lurus"

        # Di kiri
        elif x < middle_line_x:
            if percentage > 15:
                teks_perintah = "Belok Kiri!!"
                merah_melewati_batas = True

        # Di tengah
        else:
            if percentage > 30:
                teks_perintah = "Belok kiri"

        cv2.putText(frame, teks_perintah, (frame_width - 150, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        

    for contourGreen in contoursG:
        x, y, w, h = cv2.boundingRect(contourGreen)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        percentage = ((w*h)/(maximum_width*maximum_height)) * 100
        cv2.putText(frame, f"{object_green} [{str(percentage)}] x : {x} y : {y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        teks_perintah = "Lurus"

        # Di kiri
        if x < lower_middle:
            teks_perintah = "Lurus"

        # Di kanan
        elif x > middle_line_x:
            if percentage > 15:
                teks_perintah = "Belok Kanan!!"
                hijau_melewati_batas = True

        # Di tengah
        else:
            if percentage > 30:
                teks_perintah = "Belok Kanan"
        
        cv2.putText(frame, teks_perintah, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if merah_melewati_batas and hijau_melewati_batas:
            cv2.putText(frame, "PUTAR BALIK!!", (middle_line_x - 20, middle_line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)    
            

    # Gabungkan dua mask untuk mendapatkan objek yang dideteksi oleh kedua pendeteksi warna
    combined_mask = cv2.bitwise_or(maskR, maskG)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask = combined_mask)

    # Tampilkan frame asli
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', combined_mask)
    cv2.imshow('Final Result', res)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hentikan akses ke video dan tutup jendela
video.release()
cv2.destroyAllWindows()
