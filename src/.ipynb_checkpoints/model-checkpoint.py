from transformers import AutoModelForCausalLM, AutoTokenizer
import os



def load_model(model_name: str):

    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_path = os.path.join(os.getcwd(), "..", "models")

    return model
    # model.save_pretrained(model_path)
def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer