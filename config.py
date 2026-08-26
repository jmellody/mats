from pathlib import Path

MODEL = "Qwen/Qwen3.5-9B"
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # generator; must differ from MODEL
THINKING = False


ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

N_PER_CELL = 40
BATCH_SIZE = 16
SEED = 0

# spread across technicality / stakes / demographic association
TOPICS = {
    "sourdough": "sourdough bread baking",
    "chemo": "chemotherapy treatment decisions",
    "python": "python programming",
    "mortgage": "mortgage refinancing",
    "carrepair": "car engine repair",
    "musictheory": "music theory and composition",
    "taxlaw": "personal income tax law",
    "houseplants": "houseplant care",
    "ml": "machine learning research",
    "fitness": "strength training programming",
}

LABELS = ["expert", "novice", "neutral"]
