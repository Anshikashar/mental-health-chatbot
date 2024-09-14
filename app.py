from flask import Flask, request, jsonify
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Initialize Flask app
app = Flask(_name_)

# Load the model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to('cuda')  # Ensure model is on GPU

# Define a route for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    inputs = tokenizer(user_input, return_tensors='pt').to('cuda')
    reply_ids = model.generate(**inputs)
    reply_text = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    
    return jsonify({'response': reply_text})

# Run the app
if _name_ == '_main_':
    app.run(host='0.0.0.0', port=8080)
