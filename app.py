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
def get_completion(prompt):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content

# get small embedding with OpenAI API
def embed_query(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    emb = np.array(resp.data[0].embedding, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    return emb.tolist()

# fetch top k entries from Postgres DB based on embedding
def fetch_top_k(embedding, k):

    # connect to Postgres and run query
    with psycopg.connect(PG_URL) as conn:
        with conn.cursor() as cur:

            # retrieve items based on 1) cosine similarity, 2) upvote ratio, 3) sum upvotes, 4) randomness
            cur.execute("""
                WITH scored AS (
                    SELECT
                        doc_id, meta, upvote_ratio, sum_votes,
                        -- cosine similarity in [-1,1]; rescale to [0,1] to mix cleanly
                        (1 - (embedding <=> %s::vector))            AS cos_raw,
                        ((1 - (embedding <=> %s::vector)) + 1) / 2  AS cos01,

                        -- gentle vote & ratio features
                        LOG(1 + COALESCE(sum_votes,0))              AS votes_log,
                        COALESCE(upvote_ratio, 0.5)                 AS ratio_raw
                    FROM docs
                    )
                SELECT
                    doc_id, meta, upvote_ratio, sum_votes,
                    cos01,
                    -- weights you can tune: w_cos dominates; votes/ratio are small nudges
                    (0.9 * cos01)
                    + (0.05 * (votes_log /  LOG(1 + 1000)))   -- # of votes
                    + (0.05 * ratio_raw)                       -- # of 
                    + (RANDOM() * 0.1) 
                    AS final_score
                    FROM scored
                    ORDER BY final_score DESC
                    LIMIT %s;
                """,
                (embedding, embedding, k))
            rows = cur.fetchall()

    # process results
    results = []
    for doc_id, meta, upvote_ratio, sum_votes, cosine_sim, final_score in rows:

        results.append({
            "doc_id": doc_id,
            "similarity": round(float(final_score), 3),
            "meta": meta
        })

    return results

# root route
@app.route("/")
def home():    
    return render_template("index.html")

# get generic response route
@app.route("/get")
def get_bot_response():    
    userText = request.args.get('msg')  
    response = get_completion(userText)  
    
    return response

# get top recommendations route
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    query = data.get("query")
    k = int(data.get("k", 5))

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    # embed + top-k fetch
    try:
        emb = embed_query(query)
        results = fetch_top_k(emb, k)
        return jsonify({"query": query, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

if __name__ == "__main__":
    app.run(debug=True)