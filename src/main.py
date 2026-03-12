#print model performance on test 
print("========================================================")
print("MODEL PERFORMANCE ON TEST SET")
print("========================================================")

from colorama import Fore, Style
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
model_path = MODELS_DIR / "imgclf_resnet50_cifar10_v1.keras"

t = 'cacca'

print(Fore.BROWN + f"\nmodel saved at: {model_path}", t + Style.RESET_ALL) 