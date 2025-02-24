import re
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import settings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import split_sentences, split_long_sentence, translate_sentence, tokenizer_lock, PARAGRAPH_PLACEHOLDER, debug
from llm import get_summary

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
    processed_elements = []
    for elem in elements:
        if elem == PARAGRAPH_PLACEHOLDER:
            processed_elements.append(elem)
            continue
        with tokenizer_lock:
            token_count = len(tokenizer.encode(elem))
            if debug: print(token_count)
        if token_count > settings.TOKEN_LIMIT:
            parts = split_long_sentence(elem, tokenizer, settings.TOKEN_LIMIT)
            processed_elements.extend(parts)
        else:
            processed_elements.append(elem)
    if debug: print(f"Processed elements: \"{processed_elements}\"")
    elements = processed_elements
    translations = [None] * len(elements)
    previous_sentence = None
    print("Starting translation:")
    futures = {}
    with ThreadPoolExecutor(max_workers=settings.MAX_THREADS) as executor:
        for idx, element in enumerate(elements):
            if element == PARAGRAPH_PLACEHOLDER:
                translations[idx] = PARAGRAPH_PLACEHOLDER
                previous_sentence = None
                continue
            #Remove any trailing punctuation and keep it for later reattachment.
            trailing = ""
            if element and element[-1] in ".!?,;:":
                trailing = element[-1]
                element_mod = element[:-1].strip()
            else:
                element_mod = element.strip()
            #If there's a previous sentence, check if adding context exceeds token limit
            input_text = element_mod
            if debug: print(f'(Summary: \"{get_summary(input_text)}\")')
            if previous_sentence:
                context_text = f"{element_mod} ({previous_sentence.strip()})."
                with tokenizer_lock:
                    context_token_count = len(tokenizer.encode(context_text))
                if context_token_count <= settings.TOKEN_LIMIT:
                    input_text = context_text
                    if debug: print("Use context")
                else:
                    summary_text = f"{element_mod} ({get_summary(previous_sentence).strip()})."
                    with tokenizer_lock:
                        summary_token_count = len(tokenizer.encode(summary_text))
                    if summary_token_count <= settings.TOKEN_LIMIT:
                        input_text = summary_text
                        if debug: print("Use summary")
                    else:
                        print("Sentence too long, not adding context.")
            if debug: print("Input:")
            if debug: print(f"\"{input_text}\"")
            #Submit translation task
            futures[idx] = executor.submit(translate_sentence, input_text, tokenizer, model, target_lang_token_id)
            previous_sentence = element
        for idx, future in futures.items():
            try:
                result = future.result().strip()
            except Exception as e:
                print(f"Error translating sentence at index {idx}: {e}")
                result = "[TRANSLATION ERROR]"
                print("Translation error")
            if "(" in result:
                #Remove the injected context
                base_translation = result.split("(", 1)[0].strip()
            else:
                base_translation = result.strip()
            #Clean up punctuation spacing
            base_translation = re.sub(r"\s+([.,:;!?])", r"\1", base_translation)
            #Reattach original trailing punctuation if needed
            if trailing and not base_translation.endswith(trailing):
                base_translation = base_translation.rstrip(".,:;!?") + trailing
            translations[idx] = base_translation
    output_parts = []
    for i, elem in enumerate(translations):
        output_parts.append(elem)
        if i < len(translations) - 1 and elem != PARAGRAPH_PLACEHOLDER and translations[i + 1] != PARAGRAPH_PLACEHOLDER:
            output_parts.append(' ')
        print(f"Final translation: \"{elem}\"")
    output_text = ''.join(output_parts)
    with open(settings.OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write(output_text)
    translation_pairs = []
    for original, translation in zip(elements, translations):
        if original == PARAGRAPH_PLACEHOLDER:
            translation_pairs.append({
                "number": counter,
                "original": "[PARAGRAPH_BREAK]",
                "translation": "[PARAGRAPH_BREAK]",
                "advice": "",
                "corrected": True
            })
        else:
            translation_pairs.append({
                "number": counter,
                "original": original,
                "translation": translation,
                "advice": "",
                "corrected": False
            })
        counter += 1
    json_output_file = settings.OUTPUT_FILE.replace(".txt", ".json")
    with open(json_output_file, "w", encoding="utf-8") as jsonfile:
        json.dump(translation_pairs, jsonfile, ensure_ascii=False, indent=4)
    print("Translation finished and written to output files")

if __name__ == "__main__":
    main()

