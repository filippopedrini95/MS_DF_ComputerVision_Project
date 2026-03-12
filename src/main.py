#print model performance on test 
print("========================================================")
print("MODEL PERFORMANCE ON TEST SET")
print("========================================================")

from colorama import Fore, Style
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
model_path = MODELS_DIR / "imgclf_resnet50_cifar10_v1.keras"

print(Fore.MAGENTA + f"\ncheck" + Style.RESET_ALL) 