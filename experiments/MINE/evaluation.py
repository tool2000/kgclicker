import json
import sys
import os

# Add the src directory to Python path to import from source code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kg_gen.kg_gen import KGGen
import networkx as nx
import numpy as np
from openai import OpenAI
from datasets import load_dataset

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


def load_evaluation_data(answers_repo: str = "kyssen/kg-gen-evaluation-answers"):
    """Load evaluation data from Hugging Face Hub"""
    try:
        # Load answers dataset from Hugging Face
        answers_dataset = load_dataset(answers_repo)

        # Extract answers
        answers = [item["answers"] for item in answers_dataset["train"].to_list()]

        print(f"Loaded {len(answers)} answer sets from Hugging Face")
        return answers

    except Exception as e:
        print(f"Failed to load from Hugging Face: {e}")
        print("Falling back to local files...")

        # Fallback to local files
        with open("MINE/answers.json", "r") as f:
            answers = json.load(f)

        return answers


def main():
    # Load data from Hugging Face (with local fallback)
    all_questions_answers = load_evaluation_data()

    json_files = [f"MINE/results/kggen/{i}.json" for i in range(1, 107)]

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
    # main()
    with open("experiments/MINE/answers.json", "r") as f:
        answers = json.load(f)
        print(len(answers))
