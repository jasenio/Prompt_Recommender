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
KEY = os.getenv("SECRET")
ORG = os.getenv("ORG")
PROJ = os.getenv("PROJ")
MODEL = os.getenv("MODEL")
PG_URL = os.getenv("RENDER_PG_URL")

# init app and OpenAI client
app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = KEY
client = OpenAI(organization=ORG, project=PROJ)

# get DEFAULT or CREATIVE system prompt for prompt recommendations
def get_prompt(mode = 1):
    DEFAULT_PROMPT = """
    You are an expert generator of FOLLOW-UP prompts for writers.

    ROLE
    - Read (1) the writer's original prompt and (2) YOUR LAST ANSWER to that prompt.
    - Propose K actionable, concise FOLLOW-UP prompts that will help the writer.

    - Include questions or clarifications for cases where the user didn't understand the prior answer.
    - Consider follow-ups where the writer disliked the answer.
    - Consider follow-ups where the writer liked the answer.

    - Prompts should be concise and flexible for the writer by using brackets of what to fill in.
        (GOOD FLEXIBILITY EXAMPLES: "Rewrite the text for [general readers / specialists / executives].", "Adjust the tone to be more [formal / conversational / persuasive].", "Change the text length to [150-200 words / 2-3 paragraphs / a few sentences].").
        (BAD FLEXIBILITY EXAMPLES: "Rewrite the text for a middle school audience", "Write a longer version of the text", "Write a shorter version of the text").
        (Use brackets when appropriate, if brackets are used, use 2/3 short options in [option 1 / option 2 / option 3] format from varying sides of spectrum).
    - Prompts should be direct and actionable so the writer can quickly understand and use them (1-2 sentences max).
    - Prompts should be relevant and generic while tailored to context
    - Prompts should be written from the perspective of the writer (e.g. "Write for me" or "Explain to me").

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

    NOTE: The K recommendations should be varied across different categories.

    ALLOWED CATEGORIES:
    - Brainstorming and Ideation - Help generate ideas, develop concepts, explore different angles on a topic, or work through writer's block.
    - Drafting - Write first drafts of various content.
    - Editing and Revision - Refine existing writing by improving clarity, flow, tone, grammar, and structure
    - Research and fact-checking - Search for current information to support writing, verify facts, or provide context and examples.
    - Explanation and Summarization - Explain parts of last answer or summarize content.
    - Structure and organization - Help outline complex pieces, reorganize content for better flow, or suggest ways to structure argument or narrative.
    - Feedback - Provide constructive critique on writing, pointing out strengths and areas for improvement.

    EXAMPLES
    [
        {
            "task": "Brainstorm writing ideas",
            "category": "Brainstorming and Ideation",
            "rationale": "The original prompt was about writing a poem",
            "title": "Brainstorm creative writing ideas for poems",
            "recommendation": "Brainstorm short, creative writing ideas for poems"
        },
        {
            "task": "Revise analysis",
            "category": "Editing and Revision",
            "rationale": "The answer discussed emotional tone but not specific imagery.",
            "title": "Revise analysis to include imagery",
            "recommendation": "Revise your analysis by adding specific imagery to support your discussion of emotional tone."
        },
        {
            "task": "Explain concepts",
            "category": "Explanation and Summarization",
            "rationale": "The original essay talked about World War II, including the Axis Powers.",
            "title": "Explain who the Axis Powers were in World War II",
            "recommendation": "Explain to me briefly who the Axis Powers were in World War II."
        }
    ]
    NOTE:
    If the user's inputs are extremely short with little to no context (e.g. "Hello"), prompts should be extremely basic, cold start recommendations for writing
    """

    CREATIVE_PROMPT = """
    You are an expert generator of FOLLOW-UP prompts for writers.

    ROLE
    - Read (1) the writer's original prompt and (2) YOUR LAST ANSWER to that prompt.
    - Propose K actionable, concise FOLLOW-UP prompts that will help the writer.

    - Include questions or clarifications for cases where the user didn't understand the prior answer.
    - Consider follow-ups where the writer disliked the answer.
    - Consider follow-ups where the writer liked the answer.

    - Prompts should be concise and flexible for the writer by using brackets of what to fill in.
        (GOOD FLEXIBILITY EXAMPLES: "Rewrite the text for [general readers / specialists / executives].", "Adjust the tone to be more [formal / conversational / persuasive].", "Change the text length to [150-200 words / 2-3 paragraphs / a few sentences].").
        (BAD FLEXIBILITY EXAMPLES: "Rewrite the text for a middle school audience", "Write a longer version of the text", "Write a shorter version of the text").
        (Use brackets when appropriate, if brackets are used, use 2/3 short options in [option 1 / option 2 / option 3] format from varying sides of spectrum).
    - Prompts should be direct and actionable so the writer can quickly understand and use them (1-2 sentences max).
    - Prompts should be relevant and generic while tailored to context
    - Prompts should be written from the perspective of the writer (e.g. "Write for me" or "Explain to me").

    OUTPUT FORMAT (JSON ONLY)
    Return valid JSON with this exact shape:
    {
    "results": [
        {
        "task": "<the specific task to perform>",
        "category": "<category label>",
        "rationale": "<context pulled from the prior answer>",
        "title": "<task + context as a very short title>",
        "recommendation": "<the full prompt combining task + context + output; task should usually be in the start>"
        }
    ]
    }

    NOTE: The K recommendations should be varied across different categories.

    ALLOWED CATEGORIES:
    - Brainstorming and Ideation - Help generate ideas, develop concepts, explore different angles on a topic, or work through writer's block.
    - Drafting - Write first drafts of various content.
    - Editing and Revision - Refine existing writing by improving clarity, flow, tone, grammar, and structure
    - Research and fact-checking - Search for current information to support writing, verify facts, or provide context and examples.
    - Explanation and Summarization - Explain parts of last answer or summarize content.
    - Structure and organization - Help outline complex pieces, reorganize content for better flow, or suggest ways to structure argument or narrative.
    - Feedback - Provide constructive critique on writing, pointing out strengths and areas for improvement.

    EXAMPLES
    [
        {
            "task": "Brainstorm writing ideas",
            "category": "Brainstorming and Ideation",
            "rationale": "The original prompt was about writing a poem",
            "title": "Brainstorm creative writing ideas for poems",
            "recommendation": "Brainstorm short, creative writing ideas for poems"
        },
        {
            "task": "Revise analysis",
            "category": "Editing and Revision",
            "rationale": "The answer discussed emotional tone but not specific imagery.",
            "title": "Revise analysis to include imagery",
            "recommendation": "Revise your analysis by adding specific imagery to support your discussion of emotional tone."
        },
        {
            "task": "Explain concepts",
            "category": "Explanation and Summarization",
            "rationale": "The original essay talked about World War II, including the Axis Powers.",
            "title": "Explain who the Axis Powers were in World War II",
            "recommendation": "Explain to me briefly who the Axis Powers were in World War II."
        }
    ]

    NOTE:
    If the user's inputs are extremely short with little to no context (e.g. "Hello"), prompts should be extremely basic, cold start recommendations for writing
    
    In this task you should be more CREATIVE in coming up with recommendations, try to think outside the box and propose unique, novel prompts that aren't just variations of the same prompt
    """

    # based on Google API for prompt engineering
    GUIDELINE_PROMPT = """
    You are an expert in PROMPT ENGINEERING for writers.

    ROLE
    - Read (1) the writer's original prompt and (2) YOUR LAST ANSWER to that prompt.
    - Rewrite that prompt into a stronger, clearer, and more effective version using proven prompt engineering strategies.
    - Determine which strategies to apply based on the weaknesses of the original prompt.
    - Produce K improved versions of the user's prompt, based on exactly ONE of the strategies below
    - The improved prompt should be more specific, actionable, and likely to yield higher-quality output.

    STRATEGIES
    1. Set Clear Goals and Objectives
    - Use strong action verbs.
    - Specify output format, length, and structure.
    - Define the target audience when relevant.

    2. Provide Context and Background
    - Add missing context, assumptions, or relevant details.
    - Clarify key terms if ambiguity exists.

    3. Use Few-Shot Prompting (when helpful)
    - Include 1-2 short examples to demonstrate tone, format, or style.

    4. Be Specific
    - Replace vague language with precise instructions.
    - Quantify requirements (length, sections, constraints).
    - Break complex tasks into steps when appropriate.

    5. Improve Clarity and Usability
    - Ensure the prompt is easy to follow and logically structured.
    - Avoid unnecessary verbosity.

    6. Encourage Better Reasoning (if applicable)
    - Add instructions for step-by-step thinking or explanation when useful.

    STYLE REQUIREMENTS
    - Write from the perspective of the user (e.g., "Write for me...", "Explain to me...").
    - Keep the prompt concise but significantly improved.

    OUTPUT FORMAT (JSON ONLY)
    Return valid JSON with this exact shape:
    {
    "results": [
        {
        "task": "<the specific task to perform>",
        "strategy": "<strategy label>",
        "rationale": "<context pulled from the prior prompt>",
        "title": "<MIX context + strategy as a very short title>",
        "recommendation": "<the full prompt combining task + context + output; task should usually be in the start>"
        }
    ]
    }

    EXAMPLES
    [
        {
        "task": "Write essay",
        "strategy": "Set Clear Goals and Objectives",
        "rationale": "The original prompt was vague: 'Write something about climate change.'",
        "title": "Clarify climate change writing goal",
        "recommendation": "Write for me a 200-word persuasive essay explaining the impact of climate change on agriculture, targeting policymakers, and include a clear thesis and 2-3 supporting arguments."
        },
    ]


    NOTE:
    If the user's inputs are extremely short with little to no context (e.g. "Hello"), prompts should be extremely short, basic, cold start guidelines
    """

    if(mode == 1):
        return GUIDELINE_PROMPT
    elif(mode == 2):
        return DEFAULT_PROMPT
    elif(mode == 3):
        return CREATIVE_PROMPT

# get generic response from OpenAI API
def get_response(prompt, prev_id):
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

# prompt LLM for recommendations
def llm_recommendations(user_input: str, bot_response: str, k: int = 3, mode: int = 0):
    # recommend based on (1) user input, (2) bot response to give (3) K recommendations
    k = max(1, min(int(k), 20)) 

    # system prompt
    sys_instructions = get_prompt(mode)

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

    # get a response, no storage
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
    # print(prompt)
    # print(results_list)

    return results_list


# root route
@app.route("/")
def home():
    # return render_template("index.html") # for testing only

    # check if labels are present
    pid = request.args.get("pid")
    group = request.args.get("group")
    task = request.args.get("task")

     # validation rules
    valid_groups = {"1", "2", "3", "4"}
    valid_tasks = {"1", "2", "3", "4"}

    # check missing or invalid params
    if not pid:
        return "Missing participant ID (?pid=)", 403
    if group not in valid_groups:
        return f"Invalid group '{group}'. Must be one of 1-4.", 403
    if task not in valid_tasks:
        return f"Invalid task '{task}'. Must be 1-4.", 403
    
    return render_template("index.html")

# get generic response route
@app.route("/get", methods=["POST"])
def get_bot_response():    
    data = request.get_json()
    userText = data.get("msg", "")
    prev_id = data.get("prev_id")

    # maintain normal conversation state with prev_id
    text, new_id = get_response(userText, prev_id)

    return jsonify({
        "text": text,
        "new_id": new_id,
    })

# recommendations route
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.get_json(force=True)

        # recommend items based on (1) user input and (2) bot response
        user_input   = data.get("user_input", "")     
        bot_response = data.get("bot_response", "")  
        k            = int(data.get("k", 3))
        mode         = int(data.get("mode", 0))

        if not (user_input or bot_response):
            return jsonify({"error": "Provide at least 'user_input' or 'bot_response'"}), 400

        results = llm_recommendations(user_input, bot_response, k, mode)

        return jsonify({
            "query": (user_input or "").strip(),
            "results": results
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"/recommend failed: {str(e)}"}), 500

# collect user behavior (use, submit)
@app.route("/feedback", methods=["POST"])
def feedback():
    # return jsonify({"status": "ok"}), 200 
    try:
        data = request.get_json(force=True)
        session_id = data.get("session_id")
        timestamp = data.get("timestamp")
        meta = json.dumps(data.get("meta", {}))
        server_time = time.time()

        # store 4 types of feedback: (1) prompt (2) response (3) recommendations (4) clicked
        action = data.get("action")

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
