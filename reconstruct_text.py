import json

def load_data():
    try:
        with open("output_corrected.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        with open("output.json", "r", encoding="utf-8") as f:
            return json.load(f)

def reconstruct_text(data):
    output_parts = []
    for item in data:
        if item["original"] == "[PARAGRAPH_BREAK]":
            output_parts.append("\n\n")
        else:
            output_parts.append(item["translation"] + " ")
    output_text = "".join(output_parts)
    return output_text.strip()

def main():
    data = load_data()
    output_text = reconstruct_text(data)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(output_text)
    print("Text reconstruction written to output.txt.")

if __name__ == "__main__":
    main()
