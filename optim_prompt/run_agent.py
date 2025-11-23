from optim_agent import graph
from lightrag.prompt import PROMPTS


res = graph.invoke({
    "queries": ["What is the capital of France?", "What is the population of France?"],
    "best_score": 0.0,
    "cur_system_prompt": PROMPTS['naive_rag_response'],
    "best_system_prompt": PROMPTS['naive_rag_response'],
    "cur_iter": 0,
    "max_iters": 5,
})