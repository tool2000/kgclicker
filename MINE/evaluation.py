import json
from kg_gen.kg_gen import KGGen
import networkx as nx
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
        *_, context_text = kggen.retrieve(correct_answer, node_embeddings, graph)
        evaluation = gpt_evaluate_response(correct_answer, context_text)
        result = {
            "correct_answer": correct_answer,
            "retrieved_context": context_text,
            "evaluation": evaluation,
        }
        results.append(result)
        correct += evaluation

    accuracy = correct / len(questions_answers)
    results.append({"accuracy": f"{accuracy * 100:.2f}%"})

    # Save results to file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def main():
    json_files = [f"MINE/results/kggen/{i}.json" for i in range(1, 107)]

    with open("MINE/answers.json", "r") as f:
        all_questions_answers = json.load(f)

    kggen = KGGen(retrieval_model="all-MiniLM-L6-v2")
    valid_pairs = [
        (json_file, qa)
        for json_file, qa in zip(json_files, all_questions_answers)
        if os.path.exists(json_file)
    ]
    for json_file, questions_answers in valid_pairs:
        output_file = json_file.replace(".json", "_results.json")
        print(f"Processing file: {json_file}")
        try:
            nxGraph = kggen.to_nx(kggen.from_file(json_file))
            node_embeddings, _ = kggen.generate_embeddings(nxGraph)
            evaluate_accuracy(
                kggen,
                questions_answers,
                node_embeddings,
                nxGraph,
                output_file,
            )
        except Exception as e:
            print(f"Error processing file {json_file}: {str(e)}, skipping...")


if __name__ == "__main__":
    main()
