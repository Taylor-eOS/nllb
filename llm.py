import settings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sentencepiece

MODEL_NAME = "stabilityai/stablelm-zephyr-3b"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device) #non-dolphin
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device) #dolphin

TABLE = {'eng_Latn': 'English', 'deu_Latn': 'German', 'spa_Latn': 'Spanish', 'jpn_Jpan': 'Japanese', 'fra_Latn': 'French', 'fin_Latn': 'Finnish', 'dan_Latn': 'Danish'} #Add as needed

def get_advice(text, source=settings.SOURCE_LANG, target=settings.TARGET_LANG):
    prompt = f"Translate from {TABLE[source]} to {TABLE[target]} without any other comment (because I need this for an automatic translation): \"{text}\". Translation:"
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_tokens = model.generate(
        inputs.input_ids, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=False)
    result = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    if "Translation:" in result:
        result = result.split("Translation:")[-1].strip()
    elif text in result:
        result = result.replace(text, "").strip()
    result = cl(result)
    return result

def get_advice_dolphin(text, source=settings.SOURCE_LANG, target=settings.TARGET_LANG):
    prompt = f"Translate from {TABLE[source]} to {TABLE[target]}:\n{text}\nTranslation:"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    output_tokens = model.generate(inputs.input_ids, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=False)
    result = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    if "Translation:" in result:
        result = result.split("Translation:")[-1].strip()
    return result.replace('"', '')

def remove_ending(string):
    if string.endswith("_Latn"):
        return string[:-len("_Latn")]
    return string

def cl(str):
    return str.replace('"', '')

if __name__ == "__main__":
    translation_string = get_advice("Ich bin eine Katze", "deu_Latn", "dan_Latn")
    print(translation_string)
