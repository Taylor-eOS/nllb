#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    model_name = "facebook/nllb-200-distilled-1.3B"
    print("Loading tokenizer and model. This may take a while...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenizer.src_lang = "deu_Latn"
    target_lang_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

    while True:
        german_text = input("\nEnter text in German (or type 'exit' to quit): ")
        if german_text.lower() == "exit":
            print("Exiting.")
            break
        inputs = tokenizer(german_text, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=target_lang_token_id)
        english_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print("\nTranslated text (English):")
        print(english_text)

if __name__ == "__main__":
    main()

