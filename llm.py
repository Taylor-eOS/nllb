from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sentencepiece
import settings

deb = False

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(device)
TABLE = {'eng_Latn': 'English', 'deu_Latn': 'German', 'spa_Latn': 'Spanish', 'jpn_Jpan': 'Japanese', 'fra_Latn': 'French', 'fin_Latn': 'Finnish', 'dan_Latn': 'Danish'} #Add as needed

def get_advice(text, source=settings.SOURCE_LANG, target=settings.TARGET_LANG):
    prompt = f"Write this {TABLE[source]} sentence translated to {TABLE[target]}:  \"`{text}`\". (Only write the sentence itself, no other comment.) Translation: "
    if deb: print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #output_tokens = model.generate(inputs.input_ids, max_new_tokens=300, temperature=0.2, top_p=0.9, do_sample=False)
    output_tokens = model.generate(inputs.input_ids, max_new_tokens=800, temperature=0.2, top_p=0.9)
    result = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    if "Translation:" in result:
        result = result.split("Translation:")[-1].strip()
    elif text in result:
        result = result.replace(text, "").strip()
    return replace_characters(result)

def get_summary(text):
    prompt = f"Rewrite this sentence concisely: \"`{text}`\". (Write no other comment.) Rewriting: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #output_tokens = model.generate(inputs.input_ids, max_new_tokens=150, temperature=0.2, top_p=0.9, do_sample=False)
    output_tokens = model.generate(inputs.input_ids, max_new_tokens=500, temperature=0.2, top_p=0.9)
    result = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    if "Rewriting:" in result:
        result = result.split("Rewriting:")[-1].strip()
    elif text in result:
        result = result.replace(text, "").strip()
    result = result.replace('.', '')
    return replace_characters(result)

def replace_characters(sentence):
    #print(f"Original sentence: {sentence}")
    print("")
    sentence = sentence.replace('#', '')
    sentence = sentence.replace('\n', ' ')
    sentence = sentence.replace('"', '')
    sentence = sentence.replace('```', '')
    sentence = sentence.replace('`', '')
    sentence = sentence.replace('...', '')
    sentence = sentence.replace('..', '.')
    sentence = sentence.replace('\'', '')
    return sentence.strip()

def remove_ending(string):
    if string.endswith("_Latn"):
        return string[:-len("_Latn")]
    return string

if __name__ == "__main__":
    print(f"---{get_advice("I am a cat!", "eng_Latn", "deu_Latn")}---")
    print(f"---{get_summary("I am a fairly large feline, of that I am sure.")}---")
    print(f"---{get_summary("I have all the attributes that would make for a great husband, if you were to visit my farm, you would be convinced of that.")}---")

