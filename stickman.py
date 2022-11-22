import numpy as np
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp


# PREPARATION for madiapipe
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec

_THICKNESS_POSE_LANDMARKS = 2
_WHITE = (224, 224, 224)

_POSE_LANDMARKS_LEFT = frozenset([
    PoseLandmark.LEFT_EYE_INNER,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_WRIST,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.LEFT_INDEX,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_ANKLE,
    PoseLandmark.LEFT_HEEL,
    PoseLandmark.LEFT_FOOT_INDEX
])

_POSE_LANDMARKS_RIGHT = frozenset([
    PoseLandmark.RIGHT_EYE_INNER,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.RIGHT_SHOULDER, #
    PoseLandmark.RIGHT_ELBOW, #
    PoseLandmark.RIGHT_WRIST,
    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.RIGHT_HIP,
    PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_ANKLE,
    PoseLandmark.RIGHT_HEEL,
    PoseLandmark.RIGHT_FOOT_INDEX
])

# full imitaion mp.solutions.drawing_styles.get_default_pose_landmarks_style()
pose_landmark_style = {}
left_spec = DrawingSpec(color=(0, 138, 255),
                        thickness=_THICKNESS_POSE_LANDMARKS)
right_spec = DrawingSpec(color=(231, 217, 0),
                         thickness=_THICKNESS_POSE_LANDMARKS)
for landmark in _POSE_LANDMARKS_LEFT:
    pose_landmark_style[landmark] = left_spec
for landmark in _POSE_LANDMARKS_RIGHT:
    pose_landmark_style[landmark] = right_spec
pose_landmark_style[PoseLandmark.NOSE] = DrawingSpec(
      color=_WHITE, thickness=_THICKNESS_POSE_LANDMARKS)

# my changes
BLACK_COLOR = (0,0,0)
DrawSpec_Black = DrawingSpec(color=BLACK_COLOR,
                             thickness=0,
                            circle_radius=0
                            )

pose_landmark_style[PoseLandmark.RIGHT_EYE_INNER] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_EYE] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_EYE_OUTER] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_EAR] = DrawSpec_Black
pose_landmark_style[PoseLandmark.MOUTH_RIGHT] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_WRIST] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_THUMB] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_PINKY] = DrawSpec_Black
pose_landmark_style[PoseLandmark.RIGHT_INDEX] = DrawSpec_Black


pose_landmark_style[PoseLandmark.LEFT_EYE_INNER] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_EYE] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_EYE_OUTER] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_EAR] = DrawSpec_Black
pose_landmark_style[PoseLandmark.MOUTH_LEFT] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_WRIST] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_THUMB] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_PINKY] = DrawSpec_Black
pose_landmark_style[PoseLandmark.LEFT_INDEX] = DrawSpec_Black

# POSE CONNECTION
POSE_CONNECTIONS = frozenset([
#     (0, 1), (1, 2), (2, 3), (3, 7), # LEFT EYE
#     (0, 4), (4, 5), (5, 6), (6, 8),  # RIGHT EYE
#     (9, 10), # MOUTH
    (11, 12), (11, 13),
    (13, 15),
#     (15, 17), # LEFT HAND
#     (15, 19), # LEFT HAND
#     (15, 21), # LEFT HAND
#     (17, 19), # LEFT HAND
    (12, 14), (14, 16),
#     (16, 18), # RIGHT HAND
#     (16, 20), # RIGHT HAND
#     (16, 22), # RIGHT HAND
#     (18, 20), # RIGHT HAND
    (11, 23), (12, 24), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
    (29, 31), (30, 32), (27, 31), (28, 32)])


def image_pos_hands(image,
                    min_detection_confidence=0.5,
                    return_img=True,
                    plot_result=False,
                    output_image_res=(360, 360),
                    plot_size=(6, 6),
                    resize_to=False):
    """
    image - input image for recognition
    """
    image_height, image_width, _ = image.shape

    # creata image if return_img=True
    if return_img:
        # annotated_image = np.zeros_like(image)
        annotated_image = np.zeros((output_image_res[0],
                                    output_image_res[1],
                                    3))
        # annotated_image = np.zeros_like(image)

        # annotated_image = image.copy()

    hands = []
    # getting mediapipe results for hand
    with mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence) as hands:
        results_hand = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # hands result processing
    if results_hand.multi_hand_landmarks:
        for hand_landmarks in results_hand.multi_hand_landmarks:
            # using function 'landmark_to_nparray'
            # print(landmark_to_nparray(hand_landmarks))
            # drawing if True
            if return_img:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    # landmark_drawing_spec = pose_landmark_style,
                    # connection_drawing_spec = mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    connection_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=3)
                )

    # getting mediapipe results for body pose
    with mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        results_pose = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # pose result processing
    if results_pose.pose_landmarks:
        # print(plt.imshow(results_pose.segmentation_mask))
        if return_img:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=results_pose.pose_landmarks,
                connections=POSE_CONNECTIONS,
                # connections = mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_style,
                # connection_drawing_spec = mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
                connection_drawing_spec=DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=3),
                # landmark_drawing_spec = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

    # drawing if plot_result=True
    if return_img and plot_result:
        fig = plt.figure(figsize=plot_size)
        plt.title("Resultant Image");
        plt.axis('off');
        plt.imshow(annotated_image[:, :, ::-1].astype('uint8'));
        plt.show()
        # del annotated_image
    return annotated_image.astype('uint8')



