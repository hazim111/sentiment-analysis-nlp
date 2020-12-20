import transformers

DEVICE = "cpu"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
ACCUMULATION = 2
BERT_PATH = "s3://static-nlp-data/bert_base_uncased/"
MODEL_PATH = "s3://static-nlp-data/model/model.bin"
TRAINING_FILE = "IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
