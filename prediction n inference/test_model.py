import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

video_path = './Test/EAT_FOOD.mp4'

cap = cv2.VideoCapture(video_path)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

data_list = []
ROWS_PER_FRAME = 543  # Constant number of landmarks per frame

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh, \
        mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands, \
        mp_pose.Pose(static_image_mode=False) as pose:
    frame_number = 0
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB for Mediapipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process face landmarks
        results_face = face_mesh.process(image_rgb)
        if results_face.multi_face_landmarks:
            face_landmarks = results_face.multi_face_landmarks[0]
            for idx, landmark in enumerate(face_landmarks.landmark):
                data_list.append(
                    [frame_number, f"{frame_number}-face-{idx}", "face", idx, landmark.x, landmark.y, landmark.z])

        # Process hand landmarks
        results_hands = hands.process(image_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    data_list.append(
                        [frame_number, f"{frame_number}-right_hand-{idx}", "right-hand", idx, landmark.x, landmark.y,
                         landmark.z])
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Process pose landmarks
        results_pose = pose.process(image_rgb)
        if results_pose.pose_landmarks:
            pose_landmarks = results_pose.pose_landmarks.landmark
            for idx, landmark in enumerate(pose_landmarks):
                data_list.append(
                    [frame_number, f"{frame_number}-pose-{idx}", "pose", idx, landmark.x, landmark.y, landmark.z])

        # Pad the landmarks with NaN values if the number of landmarks is less than ROWS_PER_FRAME
        while len(data_list) < (frame_number + 1) * ROWS_PER_FRAME:
            data_list.append(
                [frame_number, f"{frame_number}-right_hand-{len(data_list) % ROWS_PER_FRAME}", "right-hand",
                 len(data_list) % ROWS_PER_FRAME, np.nan, np.nan, np.nan])

        # Draw the landmarks on the frame (optional)
        mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame (optional)
        cv2.imshow('MediaPipe', image)
        frame_number += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data_list, columns=["frame", "row_id", "type", "landmark_index", "x", "y", "z"])
df.to_parquet("extracted_features.parquet", index=False)

# test_data = pd.read_parquet('./1006440534.parquet')
# test_data_kaggle = pd.read_parquet('1001373962.parquet')
# test_data_kaggle2 = pd.read_parquet('./100015657.parquet')
# test_data_kaggle3 = pd.read_parquet('./1003700302.parquet')
# test_data_kaggle4 = pd.read_parquet('./1007127288.parquet')
test_data_my_own = pd.read_parquet('extracted_features.parquet')
test_data_my_own['frame'] = test_data_my_own['frame'].astype('int16')
test_data_my_own['landmark_index'] = test_data_my_own['landmark_index'].astype('int16')

test_data_my_own.shape


def load_relevant_data_subset(pq_path, ROWS_PER_FRAME=543):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    print(f"Data: {len(data)} Number of Frames: {n_frames}")
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


    # emo_raw_data = load_relevant_data_subset('./1006440534.parquet')


demo_raw_data = load_relevant_data_subset('./extracted_features.parquet')
# demo_raw_data = load_relevant_data_subset('./1003700302.parquet', test_data_kaggle3['frame'].nunique())
# demo_raw_data = load_relevant_data_subset('./extracted_features.parquet')

ORD2SIGN = {206: 'sticky',
            20: 'before',
            178: 'pretty',
            114: 'hen',
            221: 'tomorrow',
            230: 'up',
            25: 'blow',
            236: 'weus',
            184: 'read',
            191: 'say',
            248: 'zebra',
            189: 'sad',
            62: 'drawer',
            5: 'animal',
            167: 'pen',
            60: 'donkey',
            41: 'cheek',
            51: 'cowboy',
            192: 'scissors',
            181: 'quiet',
            63: 'drink',
            94: 'girl',
            200: 'sleepy',
            249: 'zipper',
            171: 'pig',
            13: 'bad',
            9: 'arm',
            61: 'down',
            123: 'if',
            240: 'why',
            166: 'pajamas',
            203: 'snow',
            137: 'loud',
            195: 'shirt',
            31: 'brown',
            146: 'moon',
            23: 'bird',
            210: 'sun',
            76: 'fast',
            1: 'after',
            54: 'cute',
            77: 'feet',
            4: 'alligator',
            87: 'food',
            113: 'hello',
            93: 'giraffe',
            180: 'puzzle',
            211: 'table',
            132: 'like',
            153: 'no',
            122: 'icecream',
            67: 'duck',
            69: 'elephant',
            141: 'many',
            18: 'bedroom',
            205: 'stay',
            74: 'fall',
            246: 'yourself',
            183: 'rain',
            135: 'listen',
            44: 'chocolate',
            124: 'into',
            11: 'awake',
            40: 'chair',
            7: 'any',
            155: 'nose',
            118: 'home',
            161: 'open',
            58: 'dog',
            50: 'cow',
            241: 'will',
            149: 'mouth',
            177: 'pretend',
            172: 'pizza',
            75: 'farm',
            163: 'outside',
            234: 'water',
            81: 'finish',
            159: 'old',
            121: 'hungry',
            112: 'helicopter',
            130: 'lamp',
            222: 'tongue',
            194: 'shhh',
            6: 'another',
            103: 'gum',
            214: 'thankyou',
            128: 'kiss',
            101: 'grass',
            64: 'drop',
            157: 'now',
            233: 'wake',
            116: 'hide',
            201: 'smile',
            226: 'toy',
            216: 'there',
            147: 'morning',
            10: 'aunt',
            102: 'green',
            36: 'car',
            213: 'taste',
            39: 'cereal',
            207: 'store',
            66: 'dryer',
            162: 'orange',
            218: 'thirsty',
            83: 'first',
            45: 'clean',
            3: 'all',
            198: 'sick',
            129: 'kitty',
            96: 'glasswindow',
            202: 'snack',
            150: 'nap',
            53: 'cut',
            73: 'face',
            99: 'grandma',
            209: 'stuck',
            91: 'garbage',
            115: 'hesheit',
            95: 'give',
            104: 'hair',
            125: 'jacket',
            165: 'owl',
            82: 'fireman',
            227: 'tree',
            16: 'because',
            17: 'bed',
            30: 'brother',
            143: 'minemy',
            127: 'jump',
            245: 'yesterday',
            145: 'mom',
            111: 'hear',
            174: 'police',
            223: 'tooth',
            212: 'talk',
            224: 'toothbrush',
            164: 'owie',
            47: 'closet',
            169: 'penny',
            24: 'black',
            85: 'flag',
            238: 'white',
            134: 'lips',
            231: 'vacuum',
            8: 'apple',
            105: 'happy',
            151: 'napkin',
            92: 'gift',
            70: 'empty',
            46: 'close',
            52: 'cry',
            138: 'mad',
            49: 'clown',
            204: 'stairs',
            42: 'child',
            173: 'please',
            65: 'dry',
            72: 'eye',
            235: 'wet',
            32: 'bug',
            109: 'haveto',
            228: 'uncle',
            199: 'sleep',
            176: 'potty',
            29: 'boy',
            136: 'look',
            107: 'hate',
            71: 'every',
            12: 'backyard',
            22: 'better',
            84: 'fish',
            56: 'dance',
            139: 'make',
            98: 'goose',
            38: 'cat',
            232: 'wait',
            14: 'balloon',
            247: 'yucky',
            2: 'airplane',
            88: 'for',
            126: 'jeans',
            154: 'noisy',
            142: 'milk',
            239: 'who',
            90: 'frog',
            35: 'can',
            215: 'that',
            117: 'high',
            244: 'yes',
            196: 'shoe',
            108: 'have',
            48: 'cloud',
            170: 'person',
            187: 'ride',
            34: 'callonphone',
            37: 'carrot',
            100: 'grandpa',
            120: 'hot',
            131: 'later',
            229: 'underwear',
            0: 'TV',
            140: 'man',
            217: 'think',
            220: 'time',
            80: 'finger',
            86: 'flower',
            15: 'bath',
            28: 'book',
            193: 'see',
            208: 'story',
            26: 'blue',
            78: 'find',
            148: 'mouse',
            79: 'fine',
            179: 'puppy',
            55: 'dad',
            21: 'beside',
            225: 'touch',
            89: 'frenchfries',
            188: 'room',
            19: 'bee',
            27: 'boat',
            156: 'not',
            59: 'doll',
            97: 'go',
            190: 'same',
            144: 'mitten',
            160: 'on',
            57: 'dirty',
            182: 'radio',
            197: 'shower',
            186: 'refrigerator',
            158: 'nuts',
            175: 'pool',
            242: 'wolf',
            243: 'yellow',
            110: 'head',
            237: 'where',
            33: 'bye',
            133: 'lion',
            152: 'night',
            106: 'hat',
            43: 'chin',
            68: 'ear',
            168: 'pencil',
            119: 'horse',
            219: 'tiger',
            185: 'red'}

import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter("./fine_tuned_model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

prediction_fn(inputs=demo_raw_data)

output = prediction_fn(inputs=demo_raw_data)
sign = output['outputs'].argmax()
print("PRED : ", ORD2SIGN.get(sign), f'[{sign}]')
