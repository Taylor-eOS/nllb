import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import split_sentences, split_long_sentence, translate_sentence, tokenizer_lock, PARAGRAPH_PLACEHOLDER, debug

def main():
    print("Loading tokenizer and model.")
    counter = 1
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_NAME)
    tokenizer.src_lang = settings.SOURCE_LANG
    target_lang_token_id = tokenizer.convert_tokens_to_ids(settings.TARGET_LANG)
    with open(settings.INPUT_FILE, "r", encoding="utf-8") as infile:
        text = infile.read()
    elements = split_sentences(text)
    # Process elements to split long sentences based on token count
    processed_elements = []
    for elem in elements:
        if elem == PARAGRAPH_PLACEHOLDER:
            processed_elements.append(elem)
            continue
        with tokenizer_lock:
            token_count = len(tokenizer.encode(elem))
        if token_count > settings.TOKEN_LIMIT:
            parts = split_long_sentence(elem, tokenizer, settings.TOKEN_LIMIT)
            processed_elements.extend(parts)
        else:
            processed_elements.append(elem)
    elements = processed_elements
    translations = [None] * len(elements)
    print("Starting translation:")
    with ThreadPoolExecutor(max_workers=settings.MAX_THREADS) as executor:
        futures = []
        for idx, element in enumerate(elements):
            if element == PARAGRAPH_PLACEHOLDER:
                translations[idx] = PARAGRAPH_PLACEHOLDER
                continue
            future = executor.submit(
                translate_sentence,
                element,
                tokenizer,
                model,
                target_lang_token_id)
            future.idx = idx
            futures.append(future)
        for future in futures:
            try:
                result = future.result()
                translations[future.idx] = result
            except Exception as e:
                print(f"Error translating sentence: {e}")
                translations[future.idx] = "[TRANSLATION ERROR]"
    output_parts = []
    for i, elem in enumerate(translations):
        output_parts.append(elem)
        if i < len(translations)-1 and \
           elem != PARAGRAPH_PLACEHOLDER and \
           translations[i+1] != PARAGRAPH_PLACEHOLDER:
            output_parts.append(' ')
    output_text = ''.join(output_parts)
    with open(settings.OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(output_text)
    #print(f"Translation complete")
    translation_pairs = []
    for original, translation in zip(elements, translations):
        if original == PARAGRAPH_PLACEHOLDER:
            translation_pairs.append({
                "number": counter,
                "original": "[PARAGRAPH_BREAK]",
                "translation": "[PARAGRAPH_BREAK]",
                "advice": "",
                "corrected": True})
        else:
            translation_pairs.append({
                "number": counter,
                "original": original,
                "translation": translation,
                "advice": "",
                "corrected": False})
        counter = counter + 1
    json_output_file = settings.OUTPUT_FILE.replace(".txt", ".json")
    with open(json_output_file, "w", encoding="utf-8") as jsonfile:
        json.dump(translation_pairs, jsonfile, ensure_ascii=False, indent=4)
    print(f"Translation written to output files")

if __name__ == "__main__":
    main()
