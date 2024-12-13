import cv2
import mediapipe as mp
import pyautogui
import math

# Hàm tính khoảng cách giữa hai điểm
def kc(a, b):
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    return math.hypot(x2 - x1, y2 - y1)

# Cấu hình Mediapipe
mp_drawing_util = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Kích thước màn hình
screenWidth, screenHeight = pyautogui.size()

# Hệ số tỉ lệ di chuột
scale_factor = 1

# Khung hình chữ nhật để giới hạn di chuột
rectangle = {
    'top': 0,
    'left': 0,
    'width': screenWidth,
    'height': screenHeight
}

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        myHand = []
        h, w, _ = img.shape
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing_util.draw_landmarks(
                img, hand_landmarks, mp_hand.HAND_CONNECTIONS)

            # Lấy tọa độ các điểm
            for id, lm in enumerate(hand_landmarks.landmark):
                myHand.append([int(lm.x * w), int(lm.y * h)])

            if len(myHand) >= 20:
                xMouse, yMouse = myHand[8][0], myHand[8][1]

                kcChuan = kc(myHand[6], myHand[5])
                click = kc(myHand[4], myHand[8])
                rclick = kc(myHand[4], myHand[20])
                bclick = kc(myHand[19], myHand[14])
                kclick = kc(myHand[8], myHand[12])

                # Di chuyển chuột nếu ngón tay nằm trong khung
                if rectangle['left'] < xMouse < rectangle['left'] + rectangle['width'] and \
                   rectangle['top'] < yMouse < rectangle['top'] + rectangle['height']:
                    x = int(xMouse * screenWidth / w)
                    y = int(yMouse * screenHeight / h)
                    pyautogui.moveTo(x, y, duration=0.1)

                # Nhấp chuột trái
                if click < kcChuan:
                    pyautogui.click(button='left')

                # Nhấp chuột phải
                if rclick < kcChuan / 2.1:
                    pyautogui.click(button='right')

                # Cuộn xuống
                if bclick < kcChuan / 1.9:
                    pyautogui.scroll(-160)

                # Cuộn lên
                if kclick < kcChuan:
                    pyautogui.scroll(160)

    # Vẽ khung giới hạn
    cv2.rectangle(
        img,
        (rectangle['left'], rectangle['top']),
        (rectangle['left'] + rectangle['width'], rectangle['top'] + rectangle['height']),
        (0, 255, 0),
        2
    )

    cv2.imshow("Nhận dạng bàn tay", img)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
