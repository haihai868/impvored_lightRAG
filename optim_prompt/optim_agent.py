from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from pydantic import BaseModel
import json

from utils import llm
from tools import call_sysprompt_api
from prompts import eval_prompt, generate_prompt_prompt


class State(TypedDict):
    queries: list[str]
    cur_score: float
    best_score: float
    cur_system_prompt: str
    best_system_prompt: str
    query_results: list[str]
    improvement_suggestions: list[str]
    cur_iter: int
    max_iters: int

class EvaluationOutput(BaseModel):
    answer_relevance_score: float
    faithfulness_score: float
    context_utility_score: float
    improvement_suggestions: str

def get_answer_node(state: State):
    results = []
    for query in state.queries:
        result = call_sysprompt_api(
            system_prompt=state.cur_system_prompt,
            query=query
        )
        query_result = result['response']
        results.append(query_result)

    return {"query_results": results, "cur_iter": state["cur_iter"] + 1}

def eval_node(state: State):
    queries_len = len(state.queries)
    query_results = []
    if queries_len > 0:
        with open("eval/data/generation.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                query_results.append(json.loads(line))
    else:
        raise ValueError("No queries provided for evaluation.")

    chain = eval_prompt | llm.with_structured_output(EvaluationOutput)

    answer_relevance_scores = []
    faithfulness_scores = []
    context_utility_scores = []
    improvement_suggestions = []
    for query_result in query_results:
        score = chain.invoke({
            "query": query_result['query'],
            "answer": query_result['answer'],
            "context": query_result['context'],
        })
        answer_relevance_scores.append(score.answer_relevance_score)
        faithfulness_scores.append(score.faithfulness_score)
        context_utility_scores.append(score.context_utility_score)

        improvement_suggestions.append(score.improvement_suggestions)

    answer_relevance_score = sum(answer_relevance_scores) / len(answer_relevance_scores) 
    faithfulness_score = sum(faithfulness_scores) / len(faithfulness_scores) 
    context_utility_score = sum(context_utility_scores) / len(context_utility_scores) 
    scores = 0.4 * answer_relevance_score + 0.4 * faithfulness_score + 0.2 * context_utility_score

    # Save the current prompt and its scores to a JSON file
    saved_prompt = {
        "prompt": state.cur_system_prompt,
        "answer_relevance_score": answer_relevance_score,
        "faithfulness_score": faithfulness_score,
        "context_utility_score": context_utility_score
    }

    try:
        with open(f"optim_prompt/data/generated_prompts.json", "r", encoding="utf-8") as f:
            log_data = json.load(f)
    except FileNotFoundError:
        log_data = []
    except json.JSONDecodeError:
        print(f"File is empty or corrupted. Initializing log_data as an empty list.")
        log_data = []

    log_data.append(saved_prompt)

    with open(f"optim_prompt/data/generated_prompts.json", "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4, ensure_ascii=False)

    return {"cur_score": scores, "improvement_suggestions": improvement_suggestions}

def update_best_prompt_node(state: State):
    if state.cur_score > state.best_score:
        return {
            "best_score": state.cur_score,
            "best_system_prompt": state.cur_system_prompt
        }
    else:
        return {
            "best_score": state.best_score,
            "best_system_prompt": state.best_system_prompt
        }

def generate_prompt_node(state: State):
    chain = generate_prompt_prompt | llm
    result = chain.invoke({
        "best_system_prompt": state.best_system_prompt,
        "queries": state.queries,
        "generated_answers": state.query_results,
        "cur_system_prompt": state.cur_system_prompt,
        "improvement_reasons": " ".join(state.improvement_suggestions)
    })
    return {"cur_system_prompt": result}

def should_continue(state: State) -> bool:
    return {"should_continue": state["cur_iter"] + 1 < state["max_iters"]}

graph_builder = StateGraph(State)
graph_builder.add_node("get_answer_node", get_answer_node)
graph_builder.add_node("eval_node", eval_node)
graph_builder.add_node("update_best_prompt_node", update_best_prompt_node)
graph_builder.add_node("generate_prompt_node", generate_prompt_node)
graph_builder.add_conditional_node("should_continue", should_continue)


graph_builder.add_edge(START, "get_answer_node")
graph_builder.add_edge("get_answer_node", "eval_node")
graph_builder.add_edge("eval_node", "update_best_prompt_node")
graph_builder.add_edge("update_best_prompt_node", "should_continue")
graph_builder.add_conditional_edges("should_continue", 
                                    lambda state: state["should_continue"], 
                                    {
                                        True: "generate_prompt_node", 
                                        False: END
                                    })
graph_builder.add_edge("generate_prompt_node", "get_answer_node")

graph = graph_builder.compile()