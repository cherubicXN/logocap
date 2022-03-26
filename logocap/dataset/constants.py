import numpy as np


COCO_PERSON_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]

KINEMATIC_TREE_SKELETON = [
    (1, 2), (2, 4),  # left head
    (1, 3), (3, 5),
    (1, 6),
    (6, 8), (8, 10),  # left arm
    (1, 7),
    (7, 9), (9, 11),  # right arm
    (6, 12), (12, 14), (14, 16),  # left side
    (7, 13), (13, 15), (15, 17),
]


COCO_KEYPOINTS = [
    'Nose',            # 1
    'L-Eye',        # 2
    'R-Eye',       # 3
    'L-Ear',        # 4
    'R-Ear',       # 5
    'L-Shoulder',   # 6
    'R-Shoulder',  # 7
    'L-Elbow',      # 8
    'R-Elbow',     # 9
    'L-Wrist',      # 10
    'R-Wrist',     # 11
    'L-Hip',        # 12
    'R-Hip',       # 13
    'L-Knee',       # 14
    'R-Knee',      # 15
    'L-Ankle',      # 16
    'R-Ankle',     # 17
]


HFLIP = {
    'left_eye': 'right_eye',
    'right_eye': 'left_eye',
    'left_ear': 'right_ear',
    'right_ear': 'left_ear',
    'left_shoulder': 'right_shoulder',
    'right_shoulder': 'left_shoulder',
    'left_elbow': 'right_elbow',
    'right_elbow': 'left_elbow',
    'left_wrist': 'right_wrist',
    'right_wrist': 'left_wrist',
    'left_hip': 'right_hip',
    'right_hip': 'left_hip',
    'left_knee': 'right_knee',
    'right_knee': 'left_knee',
    'left_ankle': 'right_ankle',
    'right_ankle': 'left_ankle',
}


DENSER_COCO_PERSON_SKELETON = [
    (1, 2), (1, 3), (2, 3), (1, 4), (1, 5), (4, 5),
    (1, 6), (1, 7), (2, 6), (3, 7),
    (2, 4), (3, 5), (4, 6), (5, 7), (6, 7),
    (6, 12), (7, 13), (6, 13), (7, 12), (12, 13),
    (6, 8), (7, 9), (8, 10), (9, 11), (6, 10), (7, 11),
    (8, 9), (10, 11),
    (10, 12), (11, 13),
    (10, 14), (11, 15),
    (14, 12), (15, 13), (12, 15), (13, 14),
    (12, 16), (13, 17),
    (16, 14), (17, 15), (14, 17), (15, 16),
    (14, 15), (16, 17),
]


DENSER_COCO_PERSON_CONNECTIONS = [
    c
    for c in DENSER_COCO_PERSON_SKELETON
    if c not in COCO_PERSON_SKELETON
]


COCO_PERSON_SIGMAS = [
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]


COCO_CATEGORIES = [
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'street sign',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'hat',
    'backpack',
    'umbrella',
    'shoe',
    'eye glasses',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'plate',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'mirror',
    'dining table',
    'window',
    'desk',
    'toilet',
    'door',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'blender',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush',
    'hair brush',
]

#TODO@DEPRECATED
CROWDPOSE_PERSON_SIGMAS = [
    0.079, 
    0.079, 
    0.072, 
    0.072, 
    0.062, 
    0.062, 
    0.107, 
    0.107, 
    0.087, 
    0.087, 
    0.089, 
    0.089, 
    0.079, 
    0.079]

#TODO@DEPRECATED
CROWDPOSE_SKELETON = [
    [13, 14],  # head-neck
    [14, 1], [14, 2],  # neck to shoulders
    [1, 2],  # shoulders
    [7, 8],  # hips
    [1, 3], [3, 5],  # left arm
    [2, 4], [4, 6],  # right arm
    [1, 7],  # left shoulder-hip
    [2, 8],  # right shoulder-hip
    [7, 9], [9, 11],  # left leg
    [8, 10], [10, 12],  # right leg
]

PERSON_SKELETON_DICT = {
    'coco': [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),],
    'crowdpose': [
        [13, 14],  # head-neck
        [14, 1], [14, 2],  # neck to shoulders
        [1, 2],  # shoulders
        [7, 8],  # hips
        [1, 3], [3, 5],  # left arm
        [2, 4], [4, 6],  # right arm
        [1, 7],  # left shoulder-hip
        [2, 8],  # right shoulder-hip
        [7, 9], [9, 11],  # left leg
        [8, 10], [10, 12],  # right leg
    ],
}

PERSON_SIGMA_DICT = {
    'coco': [
        0.026,  # nose
        0.025,  # eyes
        0.025,  # eyes
        0.035,  # ears
        0.035,  # ears
        0.079,  # shoulders
        0.079,  # shoulders
        0.072,  # elbows
        0.072,  # elbows
        0.062,  # wrists
        0.062,  # wrists
        0.107,  # hips
        0.107,  # hips
        0.087,  # knees
        0.087,  # knees
        0.089,  # ankles
        0.089,  # ankles
    ],
    'crowdpose': [
        0.079, 
        0.079, 
        0.072, 
        0.072, 
        0.062, 
        0.062, 
        0.107, 
        0.107, 
        0.087, 
        0.087, 
        0.089, 
        0.089, 
        0.079, 
        0.079
    ],
}


