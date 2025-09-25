from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import google.generativeai as genai
import re
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
# JSON search helper (enhanced to prioritize relevant fields)
# -------------------------
def search_json(data, query):
    results = []

    def _search(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Skip irrelevant paths
                if any(skip in path.lower() for skip in ['address', 'contact', 'name', 'isbn', 'publisher']):
                    continue
                # Prioritize matches in specific fields
                if query.lower() in str(k).lower() or query.lower() in str(v).lower():
                    if k in ['description', 'examples', 'kpis', 'title']:
                        results.append({"path": path + k, "value": v, "priority": 2})
                    else:
                        results.append({"path": path + k, "value": v, "priority": 1})
                _search(v, path + k + ".")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                _search(item, path + f"[{idx}].")
        elif isinstance(obj, (str, int, float)):
            if query.lower() in str(obj).lower() and not any(skip in path.lower() for skip in ['address', 'contact']):
                results.append({"path": path[:-1], "value": obj, "priority": 1})

    _search(data)
    return results

# -------------------------
# Generate answer from JSON (improved to avoid JSON dumps)
# -------------------------
def generate_answer_from_json(query, max_items=3):
    matches = search_json(ubik_info, query)
    if not matches:
        return None
    
    # Relevance score: prioritize by field priority, query in value, and path depth
    def relevance_score(match):
        score = match['priority']
        val_str = str(match['value']).lower()
        if query.lower() in val_str:
            score += 1.0  # Bonus for query in value
        depth = match['path'].count('.')
        score += depth * 0.5
        # Penalize top-level matches to avoid broad dumps
        if not match['path']:
            score -= 1.0
        return score
    
    # Sort by relevance (descending)
    matches.sort(key=relevance_score, reverse=True)
    
    # Take top relevant matches
    top_matches = matches[:max_items]
    
    # Build concise response
    summaries = []
    for match in top_matches:
        value = match['value']
        path = match['path']
        
        if isinstance(value, dict):
            # Extract specific fields, avoid dumping entire dict
            if 'description' in value:
                desc = value['description'][:150] + '...' if len(value['description']) > 150 else value['description']
                summaries.append(desc)
            elif 'examples' in value:
                examples = value['examples'][:2]
                summaries.append("\n".join(f"- {example}" for example in examples))
            elif 'kpis' in value:
                kpis = value['kpis'][:2]
                summaries.append("\n".join(f"- {kpi}" for kpi in kpis))
            else:
                title = next((k for k in value.keys() if k.lower() == 'title'), None)
                if title and 'description' in value:
                    summaries.append(f"{value[title]}: {value['description'][:100]}")
                else:
                    continue  # Skip broad dicts
        elif isinstance(value, list):
            summaries.append("\n".join(f"- {str(v)}" for v in value[:2]))
        else:
            summaries.append(str(value)[:150])
    
    # Combine into natural response
    if summaries:
        return "\n".join(summaries)
    
    return None

# -------------------------
# Hybrid answer (JSON first, Gemini fallback with enhanced prompt)
# -------------------------
def generate_answer(query):
    # Correct single-word queries using JSON first
    answer = generate_answer_from_json(query)
    
    if answer:
        return answer  # Only return what's actually found

    # If nothing found, fallback to Gemini but with a **strict instruction**
    try:
        context_json = json.dumps(ubik_info, indent=2)
        prompt = f"""
        You are UBIK AI, an assistant for Ubik Solutions.
        The user asked: "{query}"

        ONLY answer the question. Do NOT add extra details unrelated to the question.
        Do NOT explain UBIK unless explicitly asked.
        Base your answer solely on the JSON data below:

        {context_json}
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        reply = response.text.strip().replace("*", "")
        return reply if reply else "I could not find the information."
    except Exception as e:
        print("Gemini fallback error:", e)
        return "I could not find the information."

    
    # Try JSON search first for multi-word queries
    answer = generate_answer_from_json(query)
    if answer and len(answer) < 200:  # Avoid overly long JSON responses
        return answer

    # Gemini fallback for multi-word queries if JSON result is absent or too broad
    try:
        context_json = json.dumps(ubik_info, indent=2)
        prompt = f"""
        You are UBIK AI, a professional assistant for Ubik Solutions.
        User asked: "{query}"
        Craft a short (50-100 words), precise, natural answer in English using ONLY the reference JSON data.
        Do NOT return JSON or mention sources. Focus on key facts; infer context if needed (e.g., for 'products', highlight dermatology portfolio).
        Reference JSON data:
        {context_json}
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        reply = response.text.strip().replace("*", "")
        return reply if reply else "I could not find the information."
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
        Generate 5 open-ended quiz questions based on the JSON data below.
        Each question should start with 'How', 'What', or 'Why' and be relevant to Ubik Solutions.
        Return as a JSON array.
        JSON data:
        {context_data}
        """
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        questions = json.loads(response.text.strip()) if response.text.strip().startswith('[') else [
            "How does UBIK Solutions foster a supportive company culture?",
            "What are the core values of UBIK Solutions?",
            "Why is innovation important to UBIK Solutions' mission?",
            "What roles do Medical Representatives play at UBIK Solutions?",
            "How does UBIK Solutions engage with dermatologists?"
        ]
        return jsonify(questions[:5])
    except Exception as e:
        print("Quiz generation error:", e)
        return jsonify([
            "How does UBIK Solutions foster a supportive company culture?",
            "What are the core values of UBIK Solutions?",
            "Why is innovation important to UBIK Solutions' mission?",
            "What roles do Medical Representatives play at UBIK Solutions?",
            "How does UBIK Solutions engage with dermatologists?"
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

    # Simple scoring: partial match for relevance
    score = 1.0 if any(word.lower() in correct_answer.lower() for word in user_answer.split()) else 0.0

    return jsonify({
        'feedback': "Compare your answer with the correct information.",
        'score': score,
        'correct_answer': correct_answer,
        'user_answer': user_answer
    })

# -------------------------
# Chat API (enhanced for single-word queries)
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
            "- I = Ilesh Khakhkhar\n"
            "- K = Khakkar"
        )
        return jsonify({"reply": reply})

    # Check for "more" without additional context
    if "more" in corrected_message.lower() and not any(word.lower() in corrected_message.lower() for word in ["about", "details", "information", "on"]):
        reply = "Please specify what you want more details about."
        return jsonify({"reply": reply})

    # Handle single-word queries with Gemini for better intent
    reply = generate_answer(corrected_message)

    return jsonify({"reply": reply})

# -------------------------
# Start server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)