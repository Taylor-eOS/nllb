#!/usr/bin/env python
import pysbd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
import threading

tokenizer_lock = threading.Lock()

def split_sentences(text, language="de"):
    segmenter = pysbd.Segmenter(language=language, clean=True)
    return segmenter.segment(text)

def translate_sentence(sentence, tokenizer, model, target_lang_token_id):
    with tokenizer_lock:
        inputs = tokenizer(sentence, return_tensors="pt")
    tokens = model.generate(**inputs, forced_bos_token_id=target_lang_token_id)
    with tokenizer_lock:
        # Use batch_decode as in the working script
        translation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    return translation

def main():
    model_name = "facebook/nllb-200-distilled-1.3B"
    print("Loading tokenizer and model. This may take a while...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.src_lang = "deu_Latn"
    # Use convert_tokens_to_ids as in the working script
    target_lang_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    with open("input.txt", "r", encoding="utf-8") as infile:
        text = infile.read()
    sentences = split_sentences(text, language="de")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(translate_sentence, sentence, tokenizer, model, target_lang_token_id) for sentence in sentences]
        translations = [f.result() for f in futures]
    output_text = " ".join(translations)
    with open("output.txt", "w", encoding="utf-8") as outfile:
        outfile.write(output_text)
    print("Translation complete. See output.txt.")

if __name__ == "__main__":
    main()
