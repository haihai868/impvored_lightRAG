from langchain_core.prompts import ChatPromptTemplate

eval_template = """
You are a highly analytical and objective RAG (Retrieval-Augmented Generation) evaluator. Your task is to strictly score the quality of a generated answer based *only* on the provided Query, the Generated Answer, and the Retrieved Context.

You must be strict and assign a single, composite score between **0.0** (Completely inadequate) and **1.0** (Perfectly accurate and complete).

---
# Evaluation Criteria

1.  **Answer Relevance:** Does the Generated Answer directly and comprehensively address the user's Query? (Score 0.0 to 1.0)
2.  **Faithfulness:** Is the Generated Answer factually consistent with and supported by the information found *only* within the Retrieved Context? (Score 0.0 to 1.0)
3.  **Context Utility:** Was the Retrieved Context highly relevant and necessary for formulating the answer? A high score indicates the retrieved information was precisely what was needed. (Score 0.0 to 1.0)
4.  **Improvement Suggestion (Not for Scoring):** Provide a short reason why the answer could be improved, including any missing points or weaknesses.

---
# Data to Analyze:

- **User Query:** {query}
  
- **Generated Answer (RAG Output):** {answer}
  
- **Retrieved Context (Source Chunks):**
  {context}
  
---
# Output Instruction:
Calculate the composite score based on criteria above. You **MUST** output only the final three numerical scores and an improvement suggestion for the three criteria above.

**Final Score:**
"""

eval_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', eval_template),
        ("human", "Evaluate the answer strictly following the instructions above.")
    ]
)

generate_prompt_template = """
You are an advanced Prompt Engineer tasked with optimizing a system prompt for a RAG application. Your goal is to generate a new, highly refined version of the previous best system prompt to achieve a higher score in the next iteration.

The primary focus for this refinement must be to **enhance the objectivity, accuracy, and overall utility** of the generated answer, specifically addressing the issues observed in the last run.

---
# Context for Refinement:

- **Previous Best System Prompt:** {best_system_prompt}

- **Current System Prompt (Used in Last Run):** {cur_system_prompt}
  
- **User Queries from Last Run:** {queries}
  
- **Generated Answers from Last Run:** {generated_answers}
  
- **Improvement Reasons for the Current System Prompt: {improvement_reasons}

---
# Key Constraints for the NEW System Prompt:

1.  **Mandatory Placeholders:** The new system prompt **MUST** retain and utilize the original placeholders from the template prompt. If the original prompt included variables like "content_data", "response_type", or others, the new prompt **must** include them.
2.  **Focus:** The new instructions should guide the model to produce answers that are more **relevant to the query** and more **faithful to the retrieved context**, specifically based on the `Reason for Improvement`.
3.  **Output Format:** Provide only the text of the new system prompt.

---
# Output:
"""

generate_prompt_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', generate_prompt_template),
        ("human", "Generate a new better system prompt following the instructions above.")
    ]
)

