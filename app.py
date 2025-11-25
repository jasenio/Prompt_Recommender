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
        "reasoning": {
            "effort": "none",  
        },
        "service_tier": "priority",
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
    You are an expert generator of FOLLOW-UP prompts for writers.

    ROLE
    - Read (1) the writers's original prompt and (2) YOUR LAST ANSWER to that prompt.
    - Propose K actionable, concise FOLLOW-UP prompts that will help the writer.

    - Include questions or clarifications for cases where the user didn't understand the prior answer.
    - Consider follow-ups where the writer disliked the answer.
    - Consider follow-ups where the writer liked the answer.

    - Prompts should be concise and flexible for the writer by using brackets of what to fill in.
        (GOOD FLEXIBILITY EXAMPLES:   "Rewrite the text for a [ general readers / specialists . executives].", "Adjust the tone to be more [ formal / conversational / persuasive].", "Change the length to about [150-200 words / 2-3 paragraphs / few sentences].")
        (BAD FLEXIBILITY EXAMPLES: Rewrite for middle school audience, Write longer version, Write shorter version)
        (Use brackets sparingly because NOT ALL SUGGESTIONS NEED BRACKETS)
        (If brackets are used, use 2/3 short options in [option 1 / option 2 / option 3] format from varying sides of spectrum)
    - Prompts should be important and direct so the writer can quickly understand and use them. (1-2 sentences max).
    - Prompts should be relevant and generic but tailored to context
    - Prompts should be written from the perspective of the writer (e.g. ("Write for me" or "Explain to me"))


    OUTPUT FORMAT (JSON ONLY)
    Return valid JSON with this exact shape:
    {
    "results": [
        {
        "task": "<the specific task to perform>",
        "category": "<category label>",
        "context": "<context pulled from the prior answer>",
        "title": "<task + context as a very short title>",
        "recommendation": "<the full prompt combining task + context + output; task should usually be in the start>"
        }
    ]
    }

    NOTE: The K recommendations should generally be varied across different categories but depends on context


    ALLOWED CATEGORIES (MUST CHOOSE ONE):
    - Brainstorming and Ideation - Help generate ideas, develop concepts, explore different angles on a topic, or work through writer's block.
    - Drafting - Write first drafts of various content: essays, stories.
    - Editing and Revision - Refine existing writing by improving clarity, flow, tone, grammar, and structure
    - Research and fact-checking - Search for current information to support writing, verify facts, or provide context and examples.
    - Explanation and Summarization - Explain parts of last answer or summarize content
    - Structure and organization - Help outline complex pieces, reorganize content for better flow, or suggest ways to structure argument or narrative.
    - Feedback - Provide constructive critique on writing, pointing out strengths and areas for improvement.


    EXAMPLES
    [
        {
            "task": "Brainstorm writing ideas",
            "category": "Brainstorming and Ideation",
            "context": "The original prompt was about writing a poem",
            "title": "Brainstorm creative writing ideas for poems",
            "recommendation": "Brainstorm short, creative writing ideas for poems"
        },
        {
            "task": "Revise analysis",
            "category": "Editing and Revision",
            "context": "The answer discussed emotional tone but not specific imagery.",
            "title": "Revise analysis to include imagery",
            "recommendation": "Revise your analysis by adding specific imagery to support your discussion of emotional tone."
        },
        {
            "task": "Explain thesis statement",
            "category": "Explanation and Summarization",
            "context": "The writer has a thesis but has not clearly explained its meaning or implications.",
            "title": "Explain thesis statement about implications",
            "recommendation": "Explain to me the thesis statement clearly, focusing on its meaning and implications. Provide a brief explanation of the thesis."
        }
        {
            "task": "Explain concepts",
            "category": "Explanation and Summarization",
            "context": "The original essay talked about World War II, including the Axis Powers.",
            "title": "Explain who the Axis Powers were in World War II",
            "recommendation": "Explain to me briefly who the Axis Powers were in World War II."
        }

        NOTE:
        If the user's inputs are extremely short with little to no context (e.g. "Hello"), prompts should be extremely basic, cold start recommendations for writing
    ]


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
        reasoning= {
            "effort": "none",  
        },
        store=False,
        service_tier="priority",
    )

    raw = resp.output_text

    # load into json and extract results
    data = json.loads(raw)
    results_list = data["results"] 
    print(prompt)
    print(results_list)

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
    valid_tasks = {"1", "2", "3", "4"}

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
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)