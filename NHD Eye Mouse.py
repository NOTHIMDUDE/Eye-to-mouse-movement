import cv2
import dlib
from pynput.mouse import Controller

# Load the pre-trained face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize the mouse controller
mouse = Controller()

# Function to get the midpoint
def midpoint(p1, p2):
    return (p1.x + p2.x) // 2, (p1.y + p2.y) // 2

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_mid = midpoint(landmarks.part(36), landmarks.part(39))
        right_eye_mid = midpoint(landmarks.part(42), landmarks.part(45))

        eyes_mid = ((left_eye_mid[0] + right_eye_mid[0]) // 2, (left_eye_mid[1] + right_eye_mid[1]) // 2)

        # Move the mouse cursor
        screen_width, screen_height = mouse._display.size()
        mouse.position = (eyes_mid[0] * screen_width // frame.shape[1], eyes_mid[1] * screen_height // frame.shape[0])

        # Draw circles on eyes for visualization
        cv2.circle(frame, left_eye_mid, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_eye_mid, 3, (0, 255, 0), -1)
        cv2.circle(frame, eyes_mid, 3, (255, 0, 0), -1)

    cv2.imshow('Eye Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
