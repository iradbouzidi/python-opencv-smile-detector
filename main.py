import cv2

# Face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Grab Webcam feed
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # If there's an error, abroat
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale, 1.3, 5)

    # Run smile detection within each of those faces
    for (x, y, w, h) in faces:

        # Draw arectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Create the face sub-image
        # (opencv allowes you to subindex like this.
        # It's build on numpy.
        # Slice a n-dimensional array )
        face = frame[y:y+h, x:x+w]

        # Grayscale the face
        face_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect smile in the face
        smile = smile_detector.detectMultiScale(face_grayscale, 1.7, 20)

        # Draw a rectangle around the smile
        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(face, (x_, y_), (x_+w_, y_+h_), (50, 50, 200), 4)

        # Label this face as smiling
        if len(smile) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # Show the current frame
    cv2.resize(frame, (720, 480))
    cv2.imshow('Why so serious?', frame)

    # Stop if Q is pressed
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()
cv2.destroyAllWindows()

print("Code Complete")
