from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from rouge import Rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_gpt = AutoTokenizer.from_pretrained("distilgpt2")
model_gpt = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
rouge = Rouge()

def generate_gpt2(text, max_new_tokens=20):
    inputs = tokenizer_gpt(text, return_tensors="pt").to(device)
    outputs = model_gpt.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8
    )
    return tokenizer_gpt.decode(outputs[0], skip_special_tokens=True)

def evaluate_gpt2(texts):
    scores = []
    for text in texts:
        input_text = ' '.join(text.split()[:int(len(text.split())*0.75)])
        target_text = ' '.join(text.split()[int(len(text.split())*0.75):])
        pred_text = generate_gpt2(input_text)
        score = rouge.get_scores(pred_text, target_text)[0]['rouge-l']['f']
        scores.append(score)
    return sum(scores)/len(scores)
