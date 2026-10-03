"""COCO class names, UI categories and presets."""

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

_N = {n: i for i, n in enumerate(COCO_NAMES)}


def ids(*names):
    return [_N[n] for n in names]


CATEGORIES = [
    ("People", ids("person")),
    ("Vehicles", ids("car", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane")),
    ("Bags & carried items", ids("backpack", "handbag", "suitcase", "umbrella", "tie")),
    ("Tools & possible weapons", ids("knife", "scissors", "baseball bat")),
    ("Electronics", ids("cell phone", "laptop", "tv", "keyboard", "mouse", "remote")),
    ("Animals", list(range(14, 24))),
    ("Street objects", ids("traffic light", "fire hydrant", "stop sign", "parking meter", "bench")),
    ("Sports", ids("sports ball", "skateboard", "frisbee", "skis", "snowboard", "kite",
                   "baseball glove", "surfboard", "tennis racket")),
    ("Kitchen & food", [i for i in range(39, 56) if i != _N["knife"]]),
    ("Furniture & household", ids("chair", "couch", "potted plant", "bed", "dining table", "toilet",
                                  "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                                  "clock", "vase", "teddy bear", "hair drier", "toothbrush")),
]

VEHICLES = set(ids("car", "truck", "bus", "motorcycle", "train", "boat", "airplane"))

PRESETS = {
    "Security: people & vehicles": ids("person", "bicycle", "car", "motorcycle", "bus", "truck"),
    "People only": ids("person"),
    "Vehicles only": ids("bicycle", "car", "motorcycle", "bus", "truck"),
    "Theft: people, bags & valuables": ids("person", "backpack", "handbag", "suitcase", "cell phone",
                                           "laptop", "knife", "scissors", "baseball bat"),
    "Wildlife camera": ids("person") + list(range(14, 24)),
    "Animals only": list(range(14, 24)),
    "Everything": list(range(80)),
}

DEFAULT_PRESET = "Security: people & vehicles"


def color_for(cls_id: int):
    """Stable BGR color per class."""
    fixed = {0: (40, 40, 255), 1: (255, 255, 0), 2: (0, 200, 255), 3: (0, 255, 0),
             5: (255, 0, 255), 7: (255, 128, 0)}
    if cls_id in fixed:
        return fixed[cls_id]
    h = (cls_id * 47) % 180
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 180, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)
