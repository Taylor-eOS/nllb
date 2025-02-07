import pysbd
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
import threading

tokenizer_lock = threading.Lock()
output_file = "output.txt"

def split_sentences(text, language="de"):
    paragraph_breaks = re.split(r'(\n{2,})', text)
    segmenter = pysbd.Segmenter(language=language, clean=True)
    elements = []
    
    #Preserve paragraph breaks and process text segments
    for part in paragraph_breaks:
        if not part:
            continue
        if re.match(r'\n{2,}', part):
            elements.append(part)
        else:
            elements.extend(segmenter.segment(part))
    return elements

def translate_sentence(sentence, tokenizer, model, target_lang_token_id):
    if sentence.strip() == '':
        return sentence
        
    with tokenizer_lock:
        inputs = tokenizer(sentence, return_tensors="pt")
    tokens = model.generate(**inputs, forced_bos_token_id=target_lang_token_id)
    with tokenizer_lock:
        translation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(f'\"{translation}\"')
    return translation

def main():
    model_name = "facebook/nllb-200-distilled-1.3B"
    print("Loading tokenizer and model.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.src_lang = "deu_Latn"
    target_lang_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    
    with open("input.txt", "r", encoding="utf-8") as infile:
        text = infile.read()
    elements = split_sentences(text, language="de")
    translations = [None] * len(elements)
    
    print("Starting translation:")
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, element in enumerate(elements):
            if element.strip() == '':  #Paragraph break
                translations[idx] = element
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
           elem.strip() != '' and \
           translations[i+1].strip() != '':
            output_parts.append(' ')
    output_text = ''.join(output_parts)
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(output_text)
    print(f"Translation complete and written to {output_file}.")

if __name__ == "__main__":
    main()
