import os
import cv2

frontal_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
profile_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml")

# Loop through images in the folder
this_folder = os.path.join(os.path.dirname(__file__), "images")
for file in os.listdir(this_folder):
    if file.lower().endswith((".jpg", ".png")):
        my_file = os.path.join(this_folder, file)

        img = cv2.imread(my_file)
        if img is None:
            print(f"Could not read {file}, skipping...")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect frontal faces
        faces = frontal_face_cascade.detectMultiScale(gray_img,
                                                      scaleFactor=1.1,
                                                      minNeighbors=5,
                                                      minSize=(30, 30))

        # Detect right profile faces
        if len(faces) == 0:
            faces = profile_face_cascade.detectMultiScale(gray_img,
                                                          scaleFactor=1.1,
                                                          minNeighbors=5,
                                                          minSize=(30, 30))

        # Flip the image and detect left profile faces
        if len(faces) == 0:
            flipped_gray = cv2.flip(gray_img, 1)  # Flip horizontally
            faces = profile_face_cascade.detectMultiScale(flipped_gray,
                                                          scaleFactor=1.1,
                                                          minNeighbors=5,
                                                          minSize=(30, 30))

            # Flip it all back
            for i in range(len(faces)):
                x, y, w, h = faces[i]
                faces[i] = (
                gray_img.shape[1] - x - w, y, w, h)

        # Draw rectangles around faces
        if len(faces) > 0:
            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (128, 254, 0), 3)
            print(f"Detected {len(faces)} face(s) in {file}")
        else:
            print(f"No faces detected in {file}")

        # Make image half as large
        resized = cv2.resize(img,
                             (int(img.shape[1] / 2), int(img.shape[0] / 2)))

        # Show image
        cv2.imshow("Face Detection", resized)
        cv2.waitKey(500)

# Close everything
cv2.destroyAllWindows()
