import cv2
import mediapipe as mp
import pyautogui

# Initialize camera, Mediapipe Hands, and Drawing Utilities
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    img_height, img_width, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                x, y = int(lm.x * img_width), int(lm.y * img_height)
                lm_list.append((id, x, y))
                # Mark thumb tip and index finger tip
                if id == 8 or id == 4:
                    cv2.circle(img, (x, y), 10, (0, 255, 255), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Move mouse with index finger
            x_index, y_index = lm_list[8][1], lm_list[8][2]
            mouse_x = int(screen_width / img_width * x_index)
            mouse_y = int(screen_height / img_height * y_index)
            pyautogui.moveTo(mouse_x, mouse_y)

            # Detect distance between index finger tip and thumb tip for click
            x_thumb, y_thumb = lm_list[4][1], lm_list[4][2]
            distance = ((x_thumb - x_index) ** 2 + (y_thumb - y_index) ** 2) ** 0.5
            if distance < 20:
                pyautogui.click()

    cv2.imshow("Hand Mouse Control", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
