import torch
import os
import openai
from transformers import GPTNeoForCausalLM, AutoTokenizer
from flask import Flask, render_template, request, url_for

def get_prompt_response(prompt):
    model_name = "EleutherAI/gpt-neo-125M"
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Generate text
    output = model.generate(input_ids, max_length=50, early_stopping=True, temperature=0.7)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:", generated_text)
    return generated_text


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method=='POST':
        input_prompt = request.form['prompt']
        processed_response = get_prompt_response(input_prompt)
        return render_template('index.html', result=processed_response)

    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
