import json
from transformers import MarianMTModel, MarianTokenizer
from marianmt import get_advice
import settings

def load_data():
    try:
        with open("output_corrected.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("File not found")

def preload():
    data = load_data()
    for item in data:
        item["advice"] = get_advice(item["original"])
        print(f"Preloaded {item["number"]}: \"{item["advice"][:50].strip()}...\"")
    with open("output_corrected.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    preload()

