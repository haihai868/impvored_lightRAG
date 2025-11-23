from optim_agent import graph
from lightrag.prompt import PROMPTS

with open("optim_prompt/data/questions.txt", "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]


res = graph.invoke({
    "queries": questions,
    "best_score": 0.0,
    "cur_system_prompt": PROMPTS['naive_rag_response'],
    "best_system_prompt": PROMPTS['naive_rag_response'],
    "cur_iter": 0,
    "max_iters": 3,
})