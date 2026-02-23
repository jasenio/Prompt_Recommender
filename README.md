# PromptHelper: A prompt recommendation system

## About
PromptHelper is a complementary interface for chatbot workflows. After a user submits a prompt and a chatbot responds, PromptHelper will generate follow-up prompt recommendations. A user may call upon and view these recommendations on their own command. Currently, these follow-ups are related to writing tasks. Paper can be found (here)[https://arxiv.org/abs/2601.15575].

---

## Basic Features
This code consists of two main components:

- **WritingBot** - A standard chatbot that responds to a user input
- **PromptHelper** - A prompt recommender system that takes in both (a) the user's last input and (b) WritingBot's last response to recommend follow-up prompts

---

## Modules

- **app.py** - Flask backend that handles user requests, prompt generation, and other logs
- **index.html** - HTML/Javascript frontend-interface of a chatbot + PromptHelper

---

## Installation / Usage
In order to run this code, please replace environmental variables of an OpenAI API (KEY, ORG, PROJ, MODEL).

Inside **app.py** is a system prompt under llm_recommendations(). Though we tailored our system to writing, 
this prompt may be modified to other types of tasks. Explicit algorithms may be able to replace certain prompt instructions.

---
