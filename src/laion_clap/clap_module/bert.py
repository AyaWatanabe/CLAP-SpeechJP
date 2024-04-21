from transformers import AutoTokenizer, RobertaModel

tokenizer = AutoTokenizer.from_pretrained('rinna/japanese-roberta-base')
model = RobertaModel.from_pretrained('rinna/japanese-roberta-base')
text = "Replace me by any text you'd like."
def Roberta_embeddings(text):
    # text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    return output