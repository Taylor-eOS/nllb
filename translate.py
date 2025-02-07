#!/usr/bin/env python
"""
A script to translate German text to English using the
facebook/nllb-200-distilled-1.3B model.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    model_name = "facebook/nllb-200-distilled-1.3B"

    print("Loading tokenizer and model. This may take a while...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Manually define the language codes and their corresponding IDs
    lang_code_to_id = {
        "deu_Latn": tokenizer.convert_tokens_to_ids(f"<<deu_Latn>>"),
        "eng_Latn": tokenizer.convert_tokens_to_ids(f"<<eng_Latn>>")
    }

    # Prompt the user for German text.
    german_text = input("Enter text in German: ")

    # Set the source language
    tokenizer.src_lang = "deu_Latn"

    # Tokenize the input text.
    inputs = tokenizer(german_text, return_tensors="pt")

    # Set the target language ID for English
    target_lang_token_id = lang_code_to_id["eng_Latn"]

    # Generate the translation
    translated_tokens = model.generate(**inputs, forced_bos_token_id=target_lang_token_id)

    # Decode the generated tokens.
    english_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print("\nTranslated text (English):")
    print(english_text)

if __name__ == "__main__":
    main()

