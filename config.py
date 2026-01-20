DATA_PATH = 'Final_Data'
MODEL_PATH = 'final_model.keras'
SEQUENCE_LENGTH = 50
KEYPOINT_SIZE = 1662
NUM_SEQUENCES = 70
CONFIDENCE_THRESHOLD = 0.85

MAX_ACTIONS = 20
COOLDOWN_SECONDS = 1.5
STABILITY_COUNT = 2

ACTIONS = [
    'Beach', 'Blue', 'Car', 'Dance', 'Deaf', 'Family', 'Flower',
    'Friend', 'Happy', 'Hello', 'Help', 'I', 'Jump', 'Laugh',
    'Man', 'Play', 'Please', 'Red', 'Restaurant', 'Run', 'Sit',
    'Sorry', 'Stand', 'Stop', 'Thanks', 'Wait', 'Woman', 'Work', 'You'
]

LLM_BASE_URL = "http://localhost:8000/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "local-model"
