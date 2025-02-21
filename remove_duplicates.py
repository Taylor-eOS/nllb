import json

def load_data():
    try:
        with open("output_corrected.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("JSON file not found")

def preload():
    data = load_data()
    #print(data[1]["translation"][:15])
    i = 1
    while i < len(data):
        if data[i]["translation"][:15] == data[i - 1]["translation"][:15]:
            print(f"Deleting {data[i]['number']}: \"{data[i]['translation'][:50].strip()}...\"")
            del data[i]
        else:
            i += 1
            if(i < 10): print(i)
    with open("output_corrected.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    preload()

