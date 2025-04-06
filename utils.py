import re
import pysbd
import threading
from concurrent.futures import ThreadPoolExecutor
import settings

tokenizer_lock = threading.Lock()
PARAGRAPH_PLACEHOLDER = "\n\n"
debug = False

def load_settings(key):
    with open('settings.json', 'r') as f:
        settings = json.load(f)
        value = settings.get(key)
    if not key:
        raise ValueError("Read token not found in settings.json")
    return value

def split_sentences(text, language=None):
    if language is None:
        language = get_two_letter_code(settings.SOURCE_LANG)
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

def split_long_sentence(sentence, tokenizer, max_tokens):
    parts = []
    remaining = sentence.strip()
    while True:
        with tokenizer_lock:
            token_count = len(tokenizer.encode(remaining))
        if token_count <= max_tokens:
            parts.append(remaining.strip())
            break
        #Find the nearest comma to split
        split_pos = remaining.rfind(',', 0, len(remaining) // 2)
        if split_pos == -1:
            #If no comma, split in the middle
            split_pos = len(remaining) // 2
        part = remaining[:split_pos + 1].strip()
        remaining = remaining[split_pos + 1:].strip()
        parts.append(part)
    return parts

def translate_sentence(sentence, tokenizer, model, target_lang_token_id):
    if sentence == PARAGRAPH_PLACEHOLDER:
        return PARAGRAPH_PLACEHOLDER
    with tokenizer_lock:
        inputs = tokenizer(sentence, return_tensors="pt")
    tokens = model.generate(**inputs, forced_bos_token_id=target_lang_token_id)
    with tokenizer_lock:
        translation = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    if debug: print(f'Full translated sentence: \"{translation}\"')
    return translation

def get_two_letter_code(long_code):
    three_to_two = {
        'spa': 'es',  # Spanish
        'bul': 'bg',  # Bulgarian
        'pol': 'pl',  # Polish
        'jpn': 'ja',  # Japanese
        'chi': 'zh',  # Chinese (alternative)
        'kaz': 'kk',  # Kazakh
        'slk': 'sk',  # Slovak
    }
    lang_part = long_code.split('_', 1)[0]
    return three_to_two.get(lang_part, lang_part[:2])

