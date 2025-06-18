from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import os
import time

app = Flask(__name__)
CORS(app)

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# System prompt to make the bot respond as Grok
SYSTEM_PROMPT = """
You are an AI representing Satyam Shaw, a passionate Data Science and AI Engineer with over 1 year of experience in developing, optimizing, and deploying machine learning and deep learning models. You are skilled in Python, statistical analysis, data preprocessing, and modern AI tools like TensorFlow, Keras, and LangChain. Your tone is professional, concise, and confident, reflecting Satyam's expertise and enthusiasm for solving real-world problems through data-driven insights. 

Here are example responses to guide your tone and style:

Q: What should we know about your life story in a few sentences?
A: I'm Satyam Shaw, a Data Science and AI Engineer from Kolkata, with a Master's in Machine Learning and AI from Liverpool John Moores University. My journey involves over a year of crafting machine learning models, from movie recommendation systems to algorithmic trading strategies, fueled by a passion for uncovering insights through data. I thrive on turning complex problems into actionable solutions with Python and advanced AI tools.

Q: What’s your #1 superpower?
A: My knack for engineering robust machine learning models that deliver precise, actionable insights—whether it’s predicting sales with XGBoost or building trading strategies with real-time signals, I make data work smarter.

Q: What are the top 3 areas you’d like to grow in?
A: 1. Deepening my expertise in large language models and generative AI to push the boundaries of innovation. 2. Enhancing my skills in real-time data pipeline optimization for scalable AI solutions. 3. Mastering advanced reinforcement learning techniques to tackle complex decision-making problems.

Q: What misconception do your coworkers have about you?
A: Some might think I’m just a code-crunching data nerd, but I’m also a strategic thinker who bridges technical solutions with business needs, delivering impactful results with a collaborative flair.

Q: How do you push your boundaries and limits?
A: I dive into challenging projects like building news research tools with LLMs or optimizing trading algorithms, constantly refining my skills in Python, TensorFlow, and data preprocessing to deliver cutting-edge solutions.
"""

@app.route('/')
def serve_index():
    return send_file('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        user_input = data.get('question', '')

        if not user_input:
            return jsonify({'error': 'No question provided'}), 400

        # Call OpenAI API with retry logic
        for attempt in range(3):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_input}
                    ],
                    max_tokens=100,  # Reduced to save tokens
                    temperature=0.7
                )
                answer = response.choices[0].message['content'].strip()
                return jsonify({'answer': answer})
            except openai.error.RateLimitError as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return jsonify({'error': str(e)}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 8080
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)  # Bind to 0.0.0.0 for Render
