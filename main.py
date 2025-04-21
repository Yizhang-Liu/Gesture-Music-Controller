import cv2
import mediapipe as mp
import pyautogui
import time

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # 只检测一只手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 防止重复触发的计时器
last_action_time = 0
cooldown = 5  # 每个手势间隔至少5秒


def get_direction(hand_landmarks):
    """获取手势方向"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dx = index_tip.x - wrist.x
    dy = index_tip.y - wrist.y

    if abs(dx) > abs(dy):
        if dx > 0.2:
            return "RIGHT"
        elif dx < -0.2:
            return "LEFT"
    else:
        if dy < -0.2:
            return "UP"
        elif dy > 0.2:
            return "DOWN"
    return None


def is_open_hand(hand_landmarks):
    """判断是否为张开的手掌"""
    fingers = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return all(hand_landmarks.landmark[f].y < wrist_y for f in fingers)


def main():
    global last_action_time

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 镜像翻转并转换颜色空间
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理手势识别
        result = hands.process(rgb)
        now = time.time()

        if result.multi_hand_landmarks:
            # 只处理检测到的第一只手（因为设置了max_num_hands=1）
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if now - last_action_time > cooldown:
                if is_open_hand(hand_landmarks):
                    pyautogui.press('space')  # 播放/暂停
                    print("暂停/播放")
                    last_action_time = now
                else:
                    direction = get_direction(hand_landmarks)
                    if direction == "RIGHT":
                        pyautogui.hotkey('ctrl', 'right')
                        print("下一首")
                    elif direction == "LEFT":
                        pyautogui.hotkey('ctrl', 'left')
                        print("上一首")
                    elif direction == "UP":
                        pyautogui.hotkey('ctrl', 'up')
                        print("音量增加")
                    elif direction == "DOWN":
                        pyautogui.hotkey('ctrl', 'down')
                        print("音量减少")
                    if direction:
                        last_action_time = now

        # 显示窗口
        cv2.imshow("Gesture Music Controller", frame)

        # ESC键退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()

