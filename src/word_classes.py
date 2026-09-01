"""Single source of truth for the Words Mode class-index mapping.

The order must exactly match the numeric IDs in the YOLO label files.  Do not
reorder existing entries after a model has been trained; append new classes.
"""

WORD_CLASSES = [
    "Hello",
    "My_Name_IS",
    "Nice_to_Meet_You",
    "Are_You_Alright?",
    "I_am_Fine",
    "I_am_Thirsty",
    "Wait",
    "ThankYou",
]
