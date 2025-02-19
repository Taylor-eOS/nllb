#Model and tokenizer settings
MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

#Languages
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "deu_Latn"
SENTENCE_SPLIT_LANGUAGE = SOURCE_LANG[:2]

#Input and output file paths
INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"

#Threading settings
MAX_THREADS = 6
TOKEN_LIMIT = 1000
