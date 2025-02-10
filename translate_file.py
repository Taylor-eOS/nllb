import pysbd
import re
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
import threading
import settings

tokenizer_lock = threading.Lock()
PARAGRAPH_PLACEHOLDER = "\n\n"

def split_sentences(text, language=settings.SENTENCE_SPLIT_LANGUAGE):
    paragraph_breaks = re.split(r'(\n{2,})', text)
    segmenter = pysbd.Segmenter(language=language, clean=True)
    elements = []
    
    for part in paragraph_breaks:
        if not part:
            continue
        if re.match(r'\n{2,}', part):
            elements.append(PARAGRAPH_PLACEHOLDER)
        else:
            elements.extend(segmenter.segment(part))
    return elements

def translate_sentence(sentence, tokenizer, model, target_lang_token_id):
    if sentence == PARAGRAPH_PLACEHOLDER:
        return PARAGRAPH_PLACEHOLDER
        
    with tokenizer_lock:
        inputs = tokenizer(sentence, return_tensors="pt")
    tokens = model.generate(**inputs, forced_bos_token_id=target_lang_token_id)
    with tokenizer_lock:
        translation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(f'\"{translation}\"')
    return translation

def main():
    print("Remember to split yout input into lines.")
    print("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.MODEL_NAME)
    tokenizer.src_lang = settings.SOURCE_LANG
    target_lang_token_id = tokenizer.convert_tokens_to_ids(settings.TARGET_LANG)
    
    with open(settings.INPUT_FILE, "r", encoding="utf-8") as infile:
        text = infile.read()
    
    elements = split_sentences(text)
    translations = [None] * len(elements)
    
    print("Starting translation:")
    with ThreadPoolExecutor(max_workers=settings.MAX_THREADS) as executor:
        futures = []
        for idx, element in enumerate(elements):
            if element == PARAGRAPH_PLACEHOLDER:  #Paragraph break
                translations[idx] = PARAGRAPH_PLACEHOLDER
                continue
                
            future = executor.submit(
                translate_sentence,
                element,
                tokenizer,
                model,
                target_lang_token_id
            )
            future.idx = idx  #Track original position
            futures.append(future)

        #Collect results in completion order but store in original positions
        for future in futures:
            try:
                result = future.result()
                translations[future.idx] = result
            except Exception as e:
                print(f"Error translating sentence: {e}")
                translations[future.idx] = "[TRANSLATION ERROR]"

    #Reconstruct text with proper spacing
    output_parts = []
    for i, elem in enumerate(translations):
        output_parts.append(elem)
        #Add space between consecutive non-empty elements
        if i < len(translations)-1 and \
           elem != PARAGRAPH_PLACEHOLDER and \
           translations[i+1] != PARAGRAPH_PLACEHOLDER:
            output_parts.append(' ')
    
    output_text = ''.join(output_parts)
    
    with open(settings.OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(output_text)
    print(f"Translation complete and written to {settings.OUTPUT_FILE}.")
    translation_pairs = []
    for original, translation in zip(elements, translations):
        if original == PARAGRAPH_PLACEHOLDER:
            #Mark paragraph breaks in the JSON
            translation_pairs.append({
                "original": "[PARAGRAPH_BREAK]",
                "translation": "[PARAGRAPH_BREAK]"
            })
        else:
            translation_pairs.append({
                "original": original,
                "translation": translation
            })
    json_output_file = settings.OUTPUT_FILE.replace(".txt", ".json")
    with open(json_output_file, "w", encoding="utf-8") as jsonfile:
        json.dump(translation_pairs, jsonfile, ensure_ascii=False, indent=4)
    print(f"Translation pairs written to {json_output_file}.")

if __name__ == "__main__":
    main()
