import json
from kg_gen.kg_gen import KGGen
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import os


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def gpt_evaluate_response(correct_answer: str, context: str) -> int:
    prompt = f"""
    Context:
    {context}

    Correct Answer:
    {correct_answer}

    Task:
    Determine whether the context contains the information stated in the correct answer. \\
    Respond with "1" if yes, and "0" if no. Do not provide any explanation, just the number.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an evaluator that checks if the correct answer can be deduced from the information in the context.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=1,
        temperature=0.0,
    )
    return int(response.choices[0].message.content.strip())


def evaluate_accuracy(
    kggen: KGGen,
    questions_answers: list[dict],
    node_embeddings: dict[str, np.ndarray],
    model: SentenceTransformer,
    graph: nx.DiGraph,
    output_file: str,
):
    print(
        f"Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
    )
    correct = 0
    results = []

    for qa in questions_answers:
        correct_answer = qa["answer"]
        print(f"\nEvaluating answer: {correct_answer}")
        top_nodes = kggen.retrieve_relevant_nodes(
            correct_answer, node_embeddings, model
        )
        print(f"Top nodes: {top_nodes}")
        context = set()
        for node, _ in top_nodes:
            node_context = kggen.retrieve_context(node, graph)
            print(f"Context for node {node}: {node_context}")
            context.update(node_context)
        context_text = " ".join(context)
        print(f"Combined context: '{context_text}'\n---")

        evaluation = gpt_evaluate_response(correct_answer, context_text)
        results.append(
            {
                "correct_answer": correct_answer,
                "retrieved_context": context_text,
                "evaluation": evaluation,
            }
        )
        correct += evaluation
        breakpoint()
        break

    accuracy = correct / len(questions_answers)
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


# Main function to process multiple files
def main():
    json_files = [f"MINE/results/kggen/{i}.json" for i in range(1, 107)]

    with open("MINE/answers.json", "r") as f:
        all_questions_answers = json.load(f)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    valid_pairs = [
        (json_file, qa)
        for json_file, qa in zip(json_files, all_questions_answers)
        if os.path.exists(json_file)
    ]
    kggen = KGGen()
    for json_file, questions_answers in valid_pairs:
        output_file = json_file.replace(".json", "_results.json")
        print(f"Processing file: {json_file}")
        try:
            nxGraph = kggen.to_nx(kggen.from_file(json_file))
            node_embeddings, _ = kggen.generate_embeddings(nxGraph, embedding_model)
            evaluate_accuracy(
                kggen,
                questions_answers,
                node_embeddings,
                embedding_model,
                nxGraph,
                output_file,
            )
            break

        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}, skipping...")
            continue


if __name__ == "__main__":
    main()
