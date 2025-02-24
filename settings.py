#Model and tokenizer settings
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
#MODEL_NAME = "facebook/nllb-200-3.3B"

#Languages
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "deu_Latn"

#Input and output file paths
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"

#Threading settings
MAX_THREADS = 6
TOKEN_LIMIT = 1000 #Upper limit seems to be 1024
