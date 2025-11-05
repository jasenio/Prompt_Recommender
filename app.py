# import files
from flask import Flask, render_template, request, jsonify
import psycopg
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv
import json, time

# load vars
load_dotenv()
PG_URL = os.getenv("PG_URL")
KEY = os.getenv("SECRET")
ORG = os.getenv("ORG")
PROJ = os.getenv("PROJ")
MODEL = os.getenv("MODEL")

# init app and OpenAI client
app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = KEY
client = OpenAI(organization=ORG, project=PROJ)

# get generic response from OpenAI API
def get_completion(prompt, prev_id):
    request = {
        "model": MODEL,
        "input": prompt,
        "store": True,
    }

    # store id to maintain conversation state
    if prev_id is not None:
        request["previous_response_id"] = prev_id

    resp = client.responses.create(**request)

    text = resp.output_text
    new_id = resp.id
    return text, new_id

# get small embedding with OpenAI API
def embed_query(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    return emb.tolist()


# pure LLM recommendations
def llm_recommendations(user_input: str, bot_response: str, k: int = 5):
    # recommend based on (1) user input, (2) bot repsonse to give (3) K recommendations
    k = max(1, min(int(k), 20)) 

    sys_instructions = r"""
    You are an expert generator of FOLLOW-UP prompts for students.

    ROLE
    - Read (1) the student's original prompt and (2) YOUR LAST ANSWER to that prompt.
    - Propose K broad, actionable FOLLOW-UP prompts about clarifying and expanding on the current topic (not just next steps; focus ~70% on current material, 30% on advancing).
    - Prompts should be flexible so the student can easily edit them.
    - DO NOT write answers—only follow-up prompts.

    OUTPUT FORMAT (JSON ONLY)
    Return valid JSON with this exact shape:
    {
    "results": [
        {
        "task": "<category label>",
        "context": "<context pulled from the prior answer>",
        "title": "<task + context as a short title>",
        "output": "<what the student wants to receive>",
        "recommendation": "<the full, student-ready follow-up prompt combining task + context + output>"
        }
    ]
    }
    Rules:
    - Return exactly K items in "results".
    - No prose before/after the JSON. JSON must be parseable.

    ALLOWED CATEGORIES (pick the best fit):
    Idea Generation
    Explain/Clarify Concepts
    Identify
    Drafting
    Refining / Revision
    Reflect (metacognitive self-checks)
    Create (grounded in prior answer)

    STYLE & DIVERSITY REQUIREMENTS
    - Mix conversational and academic tones.
    - Include at least ONE Reflect and ONE Create item if possible.
    - Prefer soft counts ("a few," "several") over strict numbers.
    - Each "recommendation" should be concise, broad, and self-editable with brackets
    - Ensure each item names a clear desired OUTPUT (tone, length, format).

    TINY ALGORITHM
    - INPUTS: student_prompt, assistant_answer, K (default 8)
    - Map student needs as observed to: clarify/explain (majority), extend/generate, reflect, draft, revise, create.
    - Use clarifying and explanatory prompts for current answer (~70%), next-step prompts (~30%).
    - Enforce diversity AND broadness. Prompts should be easily adaptable.
    - Validate JSON.

    EXAMPLES
    [
        {
        "task": "Explain quarterly trends.",
        "context": "Reference Q2 and Q1 sales data.",
        "output": "Focus on key factors affecting performance.",
        "recommendation": "Explain quarterly sales trends referencing Q2 and Q1 data, focusing on the main factors that affected sales."
        },
        {
            "task": "Explain/Clarify Concepts",
            "context": "The answer mentioned inconsistent color coding and navigation issues.",
            "title": "Clarify usability principles involved",
            "output": "a short paragraph (100 words)",
            "recommendation": "Explain which usability heuristics (e.g., consistency, feedback, efficiency) are most relevant to improving the dashboard's color scheme and navigation."
        },
        {
        "task": "Refining / Revision",
        "context": "The answer discussed emotional tone but not specific imagery.",
        "title": "Revise for imagery support",
        "output": "a revised analytical paragraph",
        "recommendation": "Revise your analysis to include two or three concrete images or phrases from the text that strengthen your claim about mood."
        },
    ]

    VALIDATION
    - Output only valid JSON object with "results".
    - Each item must have: task, output, recommendation. Context is preferred.
    """

    # augment prompt with inputs
    prompt = f"""
    ### Inputs
    User input:
    {user_input}

    Bot's Response:
    {bot_response}

    K:
    {k}
    """

    # try completion end point
    resp = client.responses.create(
        model=MODEL,
        instructions = sys_instructions,
        input=[
            {"role": "user", "content": prompt},
        ],
        store=False,
    )

    raw = resp.output_text

    # load into json and extract results
    data = json.loads(raw)
    results_list = data["results"] 

    return results_list


# root route
@app.route("/")
def home():
    # check if labels are present
    pid = request.args.get("pid")
    group = request.args.get("group")
    task = request.args.get("task")

     # validation rules
    valid_groups = {"A", "B", "C", "D"}
    valid_tasks = {"1", "2"}

    # check missing or invalid params
    if not pid:
        return "Missing participant ID (?pid=)", 403
    if group not in valid_groups:
        return f"Invalid group '{group}'. Must be one of A-D.", 403
    if task not in valid_tasks:
        return f"Invalid task '{task}'. Must be 1 or 2.", 403
    
    return render_template("index.html")

# get generic response route
@app.route("/get", methods=["POST"])
def get_bot_response():    
    data = request.get_json()
    userText = data.get("msg", "")
    prev_id = data.get("prev_id")

    text, new_id = get_completion(userText, prev_id)

    return jsonify({
        "text": text,
        "new_id": new_id,
    })


# get top recommendations route
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)

        # recommend items based on (1) user input and (2) bot response
        user_input   = data.get("user_input", "")     
        bot_response = data.get("bot_response", "")  
        k            = int(data.get("k", 5))

        if not (user_input or bot_response):
            return jsonify({"error": "Provide at least 'user_input' or 'bot_response'"}), 400

        results = llm_recommendations(user_input, bot_response, k)

        return jsonify({
            "query": (user_input or "").strip(),
            "results": results
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"/recommend failed: {str(e)}"}), 500

# collect user behavior (use/like)
@app.route("/feedback", methods=["POST"])
def feedback():
    try:
        data = request.get_json(force=True)
        session_id = data.get("session_id")
        timestamp = data.get("timestamp")
        action = data.get("action")
        meta = json.dumps(data.get("meta", {}))
        server_time = time.time()

        # store in postgres 
        with psycopg.connect(PG_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feedback (session_id, timestamp, action, meta, server_time)
                    VALUES (%s, %s, %s, %s::jsonb, %s)
                """, (session_id, timestamp, action, meta, server_time))
                conn.commit()

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        print("Feedback insert failed:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run(debug=True)