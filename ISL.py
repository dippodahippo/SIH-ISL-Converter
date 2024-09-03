import cv2
import mediapipe as mp
import numpy as np

#this shit defines the functionalities of hand detection, mediapipe is de god
mph = mp.solutions.hands
hands = mph.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpd = mp.solutions.drawing_utils

#this function is for calculating angle between keypoints, using a very common technique called cosine angle
def calc_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

#these are the keypoints as defined by mediapipe itself, each of 0-20 are the 21 points on your hand
def get_angles(lmks):
    angs = []
    pairs = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    for (i, j, k) in pairs:
        angs.append(calc_angle(lmks[i], lmks[j], lmks[k]))
    return angs

#you calculate angles, get said angles and then you match them with the difference between their absolute values
def comp_angles(a1, a2, marg=10, th=0.8):
    match = [abs(x1 - x2) <= marg for x1, x2 in zip(a1, a2)]
    ratio = sum(match) / len(match)
    return ratio >= th

#these are image paths (this is a non CNN approach, so we can use N images per pose, just gotta give the
#same values - i love dictionaries)
img_text = {
    'C:/Users/dipto/Desktop/ISL SiH/dataset/i love you.jpg': 'I Love You',
    'C:/Users/dipto/Desktop/ISL SiH/dataset/hello.jpg': 'Hello'
}

ref_data = {} #this is like buffer memory- stores angles temporarily before being displayed
for path, text in img_text.items():
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)
    if res.multi_hand_landmarks:
        lmks = [(lm.x, lm.y, lm.z) for lm in res.multi_hand_landmarks[0].landmark]
        angs = get_angles(lmks)
        ref_data[path] = angs
        print(f"angles for {text}- {angs}")
    else:
        print(f"cannot detect angles in this image- {path}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(img_rgb)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if res.multi_hand_landmarks:
        for hand_lmks in res.multi_hand_landmarks:
            mpd.draw_landmarks(img_bgr, hand_lmks, mph.HAND_CONNECTIONS)
            lmks = [(lm.x, lm.y, lm.z) for lm in hand_lmks.landmark]
            angs = get_angles(lmks)
            print(f"detected- {angs}") #printing these for reference, can be used for training with RNN or possibly resnet based fine-tuning later 
            matched = False
            for path, ref_angs in ref_data.items(): #iterate between all hand positions in the dictionary
                if comp_angles(angs, ref_angs):
                    disp_text = img_text[path]
                    cv2.putText(img_bgr, disp_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) #displaying the corresponding text as detected
                    matched = True
                    break

            if not matched:
                cv2.putText(img_bgr, "no match seen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) #if a hand is seen but no valid pose, then return ille
    
    cv2.imshow('ISL Detection', img_bgr)
    if cv2.waitKey(5) & 0xFF == ord('m'): #press m to bye bye
        break

cap.release()
cv2.destroyAllWindows()
