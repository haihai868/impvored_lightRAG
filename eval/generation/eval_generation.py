import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall

data = []
with open("eval/data/generation.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

generation_dataset = pd.read_csv("eval/data/generation_dataset.csv")

data = {
    'question': [item['question'] for item in data],
    'answer': [item['answer'] for item in data],
    'contexts': [[item['contexts']] for item in data],
    'ground_truth': generation_dataset['ground_truth'].tolist()
}

dataset = Dataset.from_dict(data)


gemini_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 
gemini_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

ragas_gemini_llm = LangchainLLMWrapper(gemini_llm)
ragas_gemini_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

metrics = [answer_relevancy, faithfulness, context_recall, context_precision] 

for metric in metrics:
    if hasattr(metric, 'llm'):
        metric.__setattr__("llm", ragas_gemini_llm)
        
    if hasattr(metric, 'embeddings'):
        metric.__setattr__("embeddings", ragas_gemini_embeddings)

    if metric.name == 'context_recall':
         metric.__setattr__("llm", ragas_gemini_llm)

# score = evaluate(dataset, llm=ragas_gemini_llm, metrics=metrics)
# df = score.to_pandas()
# df.to_csv('eval/data/score.csv', index=False)

score = pd.read_csv('eval/data/score.csv')
score[['answer_relevancy','faithfulness','context_recall','context_precision']].to_csv('eval/data/score.csv')