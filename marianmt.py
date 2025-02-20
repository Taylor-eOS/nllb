from transformers import MarianMTModel, MarianTokenizer
import settings

source_l = settings.SOURCE_LANG
target_l = settings.TARGET_LANG

def load_model(source, target):
    model_name = f"Helsinki-NLP/opus-mt-{source[:2]}-{target[:2]}" #This might not work with all languages
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

def get_advice(text):
    model, tokenizer = load_model(source=source_l, target=target_l)
    translated_text = translate_text(model, tokenizer, text)
    for char, replacement in [('"', ''), ('`', ''), ('\n', ' '), ('..', '.'), ("'", '')]:
        translated_text = translated_text.replace(char, replacement)
    return translated_text

def translate_text(model, tokenizer, text, max_length=512, overlap=30):
    chunks = split_into_chunks(text, tokenizer, max_length=max_length, overlap=overlap)
    translated_chunks = [translate_chunk(model, tokenizer, chunk) for chunk in chunks]
    return " ".join(translated_chunks)

def translate_chunk(model, tokenizer, chunk):
    inputs = {"input_ids": chunk.unsqueeze(0)}
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def split_into_chunks(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_length, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

if __name__ == "__main__":
    get_advice("I am a cat")

