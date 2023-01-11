import cv2


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def mark_faces(frame):
    """
    Detect faces in the given frame and mark them with rectangles.

    :param frame: Pixel array
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        _, frame = capture.read()
        mark_faces(frame)
        cv2.imshow("Face detector", frame)
        if cv2.waitKey(1) == ord("q"):
            break
    capture.release()
    cv2.destroyAllWindows()
