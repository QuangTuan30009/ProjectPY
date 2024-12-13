import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

# Initialize Mediapipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Define colors and labels
colors = [
    (0, 0, 255),   # Red
    (255, 0, 0),   # Blue
    (0, 255, 0),   # Green
    (226, 43, 138),# Purple
    (255, 255, 0), # Cyan
    (0, 255, 255), # Yellow
    (0, 0, 0)      # Erase
]

labels = ["Red", "Blue", "Green", "Purple", "Cyan", "Yellow", "Erase"]

# Initial settings
current_color = colors[0]
xp, yp = 0, 0
brush_thickness = 10
erase_thickness = 50
header_height = 100

canvas = None

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Initialize canvas if not created
        if canvas is None:
            canvas = np.zeros((h, w, 3), np.uint8)

        # Draw color selection header
        sections_width = w // len(colors)
        for i, (label, col) in enumerate(zip(labels, colors)):
            cv2.rectangle(frame, (i * sections_width, 0), ((i + 1) * sections_width, header_height), col, -1)
            text_color = (255, 255, 255) if col != (255, 255, 255) else (0, 0, 0)
            cv2.putText(frame, label, (i * sections_width + 20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # Process hand landmarks
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                landmarks = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((id, cx, cy))

                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                # Get tip positions of index (8) and middle (12) fingers
                x1, y1 = landmarks[8][1:]
                x2, y2 = landmarks[12][1:]

                # Check if the hand is in the header (select color)
                if y1 < header_height and y2 < header_height:
                    xp, yp = 0, 0
                    section_index = min(max(x1 // sections_width, 0), len(colors) - 1)
                    current_color = colors[section_index]

                # Drawing mode
                elif y1 < landmarks[6][2] and y2 > landmarks[10][2]:
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    thickness = erase_thickness if current_color == (0, 0, 0) else brush_thickness
                    cv2.line(canvas, (xp, yp), (x1, y1), current_color, thickness)
                    xp, yp = x1, y1

                else:
                    xp, yp = 0, 0

        # Merge canvas and frame
        imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)

        # Display frames
        cv2.imshow("Painter", frame)
        cv2.imshow("Canvas", canvas)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("Execution stopped by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
