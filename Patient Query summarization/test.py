from transformers import AutoTokenizer,AutoModelForSeq2SeqLM


finetuned_tokenizer = AutoTokenizer.from_pretrained("SUMM_QUERY")
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained("TOK_QUERY")

def predict(text):
    inputs = finetuned_tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    outputs = finetuned_model(**inputs)
    
    
    return outputs


text = """

'hi doctor I am just wondering what is abutting and abutment of the nerve root means in a back issue please explain what treatment is required for annular bulging and tear'


"""


predict(text)