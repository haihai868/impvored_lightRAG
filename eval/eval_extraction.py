import requests
import json

def normalize(text):
    if not text:
        return ""
    text = text.strip().lower()
    return text

def get_graph(label: str, max_depth: int = 5, max_nodes: int = 1000):
    url = "http://localhost:9621/graphs"

    params = {
        "label": label,
        "max_depth": max_depth,
        "max_nodes": max_nodes
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()  
        return data
    else:
        print("Error:", response.status_code, response.text)

def get_nodes_edges_from_graph(graph_data):
    model_entities = {}
    for node in graph_data["nodes"]:
        eid = normalize(node["properties"]["entity_id"])
        etype = normalize(node["properties"]["entity_type"])
        model_entities[eid] = etype

    model_relations = set()
    for edge in graph_data["edges"]:
        src = normalize(edge["source"])
        tgt = normalize(edge["target"])
        desc = normalize(edge["properties"].get("description", ""))
        model_relations.add((src, desc, tgt))

    return model_entities, model_relations

def get_nodes_edges_from_evaldata(eval_data):
    entities = {}
    for ent in eval_data["entities"]:
        entities[normalize(ent["id"])] = normalize(ent["type"])

    relations = set()
    for rel in eval_data["relations"]:
        relations.add((normalize(rel["source"]),
                          normalize(rel["relation"]),
                          normalize(rel["target"])))

    return entities, relations

def eval_entities(model_entities, gt_entities):
    tp = fp = fn = wrong_type = 0

    missing_entities = []
    extra_entities = []
    wrong_type_entities = []

    for id, type in gt_entities.items():
        if id not in model_entities:
            missing_entities.append(id)
            fn += 1
        else:
            if model_entities[id] != type:
                wrong_type_entities.append((id, type, model_entities[id]))
                wrong_type += 1
            else:
                tp += 1

    for id in model_entities.keys():
        if id not in gt_entities:
            extra_entities.append(id)
            fp += 1

    precision = tp / (tp + fp + wrong_type) if (tp + fp + wrong_type) > 0 else 0
    recall = tp / (tp + fn + wrong_type) if (tp + fn + wrong_type) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    type_accuracy = tp / (tp + wrong_type) if (tp + wrong_type) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "type_accuracy": type_accuracy,
        "missing_entities": missing_entities,
        "extra_entities": extra_entities,
        "wrong_type_entities": wrong_type_entities,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "wrong_type": wrong_type,
    }

def eval_relations(model_relations, gt_relations):
    tp = fp = fn = 0

    matched = []
    missing = []
    extra = []

    used_model = set()
    for gt_source, gt_relation, gt_target in gt_relations:
        found = False
        for model_source, model_desc, model_target in model_relations:
            if gt_source == model_source and gt_target == model_target:
                found = True
                matched.append((gt_source, gt_relation, gt_target))
                used_model.add((model_source, model_desc, model_target))
                tp += 1
                break

        if not found:
            missing.append((gt_source, gt_relation, gt_target))
            fn += 1

    
    for rel in model_relations:
        if rel not in used_model:
            fp += 1
            extra.append(rel)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1       = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_relations": matched,
        "missing_relations": missing,
        "extra_relations": extra,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }

if __name__ == "__main__":
    labels = ["Species", "Theory of Relativity", "NASA", "GPT-4", "MapReduce Method"]
    all_model_nodes = {}
    all_model_edges = set()

    for label in labels:
        graph_data = get_graph(label)
        nodes, edges = get_nodes_edges_from_graph(graph_data)
        all_model_nodes.update(nodes)
        all_model_edges.update(edges)

    with open('eval/eval_extraction_data.json', 'r') as f:
        eval_extraction_data = json.load(f)

    all_gt_nodes = {}
    all_gt_edges = set()
    for doc in eval_extraction_data['documents']:
        gt_nodes, gt_edges = get_nodes_edges_from_evaldata(doc)
        all_gt_nodes.update(gt_nodes)
        all_gt_edges.update(gt_edges)

    # entity_eval = eval_entities(all_model_nodes, all_gt_nodes)
    relation_eval = eval_relations(all_model_edges, all_gt_edges)

    # with open('eval/entities_extraction_eval_results.json', 'w') as f:
    #     json.dump(entity_eval, f)

    with open('eval/relations_extraction_eval_results.json', 'w') as f:
        json.dump(relation_eval, f)