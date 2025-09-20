from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------
# Load environment
# -------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or "AIzaSyCCz5Vrv76PE01k4ENPnhBYmgP-qcnbAJg"
if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set.")
genai.configure(api_key=api_key)

# -------------------------
# Load UBIK JSON data
# -------------------------
with open('ubik_data.json', 'r', encoding='utf-8') as f:
    ubik_info = json.load(f)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, static_folder='static')
CORS(app)

# -------------------------
# Spell correction
# -------------------------
def correct_spelling(user_input):
    corrections = {
        'ubeek': 'UBIK', 'ubiik': 'UBIK', 'youbik': 'UBIK', 'ubique': 'UBIK', 'yogic': 'UBIK',
        'ethiglo': 'EthiGlo', 'ethi glo': 'EthiGlo', 'ethiglow': 'EthiGlo', 'ethigloo': 'EthiGlo',
        'sisonext': 'SisoNext', 'tehnology': 'technology', 'wat': 'what',
        'prodacts': 'products', 'soultion': 'solution'
    }
    for wrong, right in corrections.items():
        user_input = user_input.replace(wrong, right)
    return user_input

# -------------------------
# JSON search helper
# -------------------------
def search_json(data, query):
    results = []

    def _search(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if query.lower() in str(k).lower() or query.lower() in str(v).lower():
                    results.append({"path": path + k, "value": v})
                _search(v, path + k + ".")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _search(item, path + f"[{idx}].")
        elif isinstance(obj, (str, int, float)):
            if query.lower() in str(obj).lower():
                results.append({"path": path[:-1], "value": obj})

    _search(data)
    return results

# -------------------------
# Generate answer from JSON
# -------------------------
def generate_answer_from_json(query, max_items=5):
    matches = search_json(ubik_info, query)
    if not matches:
        return None
    answer_lines = []
    for item in matches[:max_items]:
        value = item['value']
        if isinstance(value, dict):
            value = ", ".join(f"{k}: {v}" for k, v in value.items())
        elif isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        answer_lines.append(str(value))
    return " ".join(answer_lines)

# -------------------------
# Hybrid answer (JSON first, Gemini fallback)
# -------------------------
def generate_answer(query):
    # 1. JSON search
    answer = generate_answer_from_json(query)
    if answer:
        return answer  # Direct, confident answer

    # 2. Gemini fallback
    try:
        context_json = json.dumps(ubik_info, indent=2)
        prompt = f"""
You are UBIK AI, a professional assistant.
User asked: "{query}"
Reference JSON data:
{context_json}

Provide a short, precise, confident answer in English. Do NOT mention JSON or data sources.
"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        reply = response.text.strip().replace("*", "")
        if not reply:
            reply = "I could not find the information."
        return reply
    except Exception as e:
        print("Gemini fallback error:", e)
        return "I could not find the information."

# -------------------------
# Routes (static pages)
# -------------------------
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

# -------------------------
# Quiz questions
# -------------------------
@app.route('/api/questions', methods=['GET'])
def get_questions():
    try:
        context_data = json.dumps(ubik_info, indent=2)
        prompt = f"""
Generate 5 open-ended quiz questions:
{context_data}
Each question should start with 'How', 'What', or 'Why'.
Return as a JSON array.
"""
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        questions = json.loads(response.text.strip()) if response.text.strip().startswith('[') else [
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to dermatology unique?",
            "What are the benefits of UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products work?"
        ]
        return jsonify(questions[:5])
    except Exception as e:
        print("Quiz generation error:", e)
        return jsonify([
            "How does UBIK Solutions leverage AI for dermatology applications?",
            "What are the key services offered by UBIK Solutions?",
            "Why is UBIK Solutions' approach to dermatology unique?",
            "What are the benefits of UBIK Solutions' Anti-Acne products?",
            "How do UBIK Solutions' Anti-Ageing products work?"
        ])

# -------------------------
# Evaluate answers
# -------------------------
@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    question = data.get('question', '')
    user_answer = data.get('answer', '')

    correct_answer = generate_answer(question)

    # Simple scoring
    score = 1.0 if any(word.lower() in correct_answer.lower() for word in user_answer.split()) else 0.0

    return jsonify({
        'feedback': "Compare your answer with the correct information.",
        'score': score,
        'correct_answer': correct_answer,
        'user_answer': user_answer
    })

# -------------------------
# Chat API
# -------------------------
@app.route('/api/chat', methods=['POST'])
def chatbot_reply():
    data = request.get_json()
    msg = data.get("message", "")
    user_message = msg["text"] if isinstance(msg, dict) and "text" in msg else msg
    print("User message:", user_message)

    corrected_message = correct_spelling(user_message)

    # Special case: UBIK full form
    if "ubik" in corrected_message.lower() and ("full form" in corrected_message.lower() or "meaning" in corrected_message.lower()):
        reply = (
            "UBIK stands for:\n"
            "- U = Utsav Khakkar\n"
            "- B = Bhavini Khakkar\n"
            "- I = Ilesh Khakkhar\n"
            "- K = Khakkar"
        )
        return jsonify({"reply": reply})

    # Hybrid answer
    reply = generate_answer(corrected_message)

    return jsonify({"reply": reply})

# -------------------------
# Start server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)