from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Gemini API key
api_key = "AIzaSyCCz5Vrv76PE01k4ENPnhBYmgP-qcnbAJg"
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=api_key)

# Load UBIK reference data from JSON
with open('ubik_data.json', 'r', encoding='utf-8') as f:
    ubik_info = json.load(f)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)

# Default system message for Gemini
DEFAULT_SYSTEM_MESSAGE = """You are UBIK AI, a professional enterprise-grade assistant with comprehensive knowledge about UBIK solutions and its subsidiaries, engineered for precision and reliability.
IMPORTANT: You can only understand and respond in English.
IMPORTANT: Always provide the best suitable answer possible based on the provided context and your knowledge.
IMPORTANT - UBIK'S WEBSITE IS https://www.ubiksolution.com

CORE OPERATIONAL DIRECTIVES:
[...same content you had above... for brevity, keep the full content here unchanged...]
"""

# Spelling correction for internal use only
def correct_spelling(user_input):
    input_lower = user_input.lower()
    corrections = {
        'ubeek': 'UBIK', 'ubiik': 'UBIK', 'youbik': 'UBIK',
        'ethiglo': 'EthiGlo', 'ethi glo': 'EthiGlo', 'ethiglow': 'EthiGlo', 'ethigloo': 'EthiGlo',
        'sisonext': 'SisoNext',
        'tehnology': 'technology', 'wat': 'what', 'prodacts': 'products', 'soultion': 'solution'
    }
    for wrong, right in corrections.items():
        if wrong in input_lower:
            user_input = user_input.replace(wrong, right)
    return user_input.capitalize()

# Static routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/quiz-instruction')
def quiz_instruction():
    return send_from_directory('static', 'quiz-instruction.html')

@app.route('/quiz')
def quiz():
    return send_from_directory('static', 'quiz.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

# Quiz question generation
@app.route('/api/questions', methods=['GET'])
def get_questions():
    try:
        prompt = """
        You are an official representative of UBIK Solutions. Generate 5 open-ended quiz questions based on general knowledge about UBIK Solutions, focusing on services, mission, or product categories (e.g., Anti-Acne, Anti-Ageing). Each question should start with 'How', 'What', or 'Why'. Return the questions as a JSON array, e.g., ["question1", "question2", ...].
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Quiz API response:", response.text)

        questions = json.loads(response.text.strip()) if response.text.strip().startswith('[') else [
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to dermatology unique?",
            "What are the benefits of UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products work?"
        ]
        return jsonify(questions[:5])
    except Exception as e:
        print("Error generating questions:", e)
        return jsonify([
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to dermatology unique?",
            "What are the benefits of UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products work?"
        ])

# Answer evaluation
@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    answer = data.get('answer', '')

    print(f"Evaluating question: {question}")
    print(f"Answer: {answer}")

    try:
        context_data = json.dumps(ubik_info, indent=2)

        prompt = f"""
        You are an official representative of UBIK Solutions.

        Reference data about UBIK:
        {context_data}

        Evaluate the following answer to the question: '{question}'
        Answer: '{answer}'

        Provide feedback on the correctness and completeness of the answer based on the data above. 
        Return a JSON object with 'feedback' (string) and 'score' (float, 0 to 1). 
        Give a partial score (0.5–0.8) if the answer includes relevant keywords.
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Gemini API response:", response.text)

        try:
            evaluation = json.loads(response.text.strip())
            if not isinstance(evaluation, dict) or 'feedback' not in evaluation or 'score' not in evaluation:
                raise ValueError("Invalid format")
            evaluation['score'] = float(evaluation['score'])
            if not 0 <= evaluation['score'] <= 1:
                evaluation['score'] = 0.0
                evaluation['feedback'] = "Invalid score returned; please try again."
        except Exception as e:
            print("Parse error:", e)
            evaluation = {
                'feedback': "Unable to evaluate due to format issue. Please try again with a clear answer.",
                'score': 0.0
            }

        return jsonify(evaluation)
    except Exception as e:
        print("Evaluation error:", e)
        return jsonify({
            'feedback': "Error evaluating answer. Please try again later.",
            'score': 0.0
        })

@app.route('/api/quiz-questions', methods=['GET'])
def get_quiz_questions():
    try:
        context_data = json.dumps(ubik_info, indent=2)
        prompt = f"""
        You are an official representative of UBIK Solutions.
        Reference data about UBIK:
        {context_data}
        Generate 5 open-ended quiz questions based on the provided UBIK Solutions data, focusing on services, mission, or product categories (e.g., Anti-Acne, Anti-Ageing). Each question should start with 'How', 'What', or 'Why'. Return the questions as a JSON array, e.g., ["question1", "question2", ...].
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        print("Quiz Questions API response:", response.text)

        questions = json.loads(response.text.strip()) if response.text.strip().startswith('[') else [
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to dermatology unique?",
            "What are the benefits of UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products work?"
        ]
        return jsonify(questions[:5])
    except Exception as e:
        print("Error generating quiz questions:", e)
        return jsonify([
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to dermatology unique?",
            "What are the benefits of UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products work?"
        ])
# Chatbot API with UBIK JSON info
# Chatbot API with UBIK JSON info
@app.route('/api/chat', methods=['POST'])
def chatbot_reply():
    data = request.get_json()
    msg = data.get("message", "")
    user_message = msg["text"] if isinstance(msg, dict) and "text" in msg else msg
    print("User message:", user_message)

    # Clean and spell-correct user input (internal only)
    corrected_message = correct_spelling(user_message)

    # Load UBIK data for context
    context_data = json.dumps(ubik_info, indent=2)

    # Precision-tuned prompt
    prompt = f"""{DEFAULT_SYSTEM_MESSAGE}

UBIK REFERENCE DATA:
{context_data}

User asked: "{corrected_message}"

Instructions:
- ONLY answer what the user asked.
- DO NOT include unrelated info or guess intent.
- If the question is clearly about UBIK (products, people, services, brands like EthiGlo or SisoNext), use the reference data above.
- If unrelated to UBIK, answer as best you can or politely say you don't have info.
- Keep answers short, clear, and relevant.
"""

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        reply = response.text.strip()

        # Fallback in case of bad or blank reply
        if not reply or "sorry" in reply.lower() or "not enough" in reply.lower():
            reply = "I'm not sure I have enough information to answer that. Could you please clarify or ask something else? 😊"
    except Exception as e:
        print("Chat Error:", e)
        reply = "Oops! Something went wrong while processing your question. Please try again. 😕"

    print("Sending reply:", reply)
    return jsonify({"reply": reply})


# Start server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
