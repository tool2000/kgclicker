from dotenv import load_dotenv
from src.kg_gen import KGGen
import os
import pytest

load_dotenv()


def match_subset(set1, set2):
    return set1.issubset(set2) or set2.issubset(set1)


@pytest.fixture
def kg():
    return KGGen(api_key=os.getenv("OPENAI_API_KEY"))


def test_basic(kg: KGGen):
    # Generate a simple graph
    text = "Harry has two parents - his dad James Potter and his mom Lily Potter. Harry and his wife Ginny have three kids together: their oldest son James Sirius, their other son Albus, and their daughter Lily Luna."

    graph = kg.generate(input_data=text, model="openai/gpt-4o")

    expected_entities = {
        "Harry",
        "James Potter",
        "Lily Potter",
        "Ginny",
        "James Sirius",
        "Albus",
        "Lily Luna",
    }
    print(graph)
    assert match_subset(graph.entities, expected_entities)


def test_clustered(kg: KGGen):
    # Test texts
    text1 = "Linda is Joshua's mother. Ben is Josh's brother. Andrew is Josh's father."
    text2 = "Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt. Josh also goes by Joshua."

    # Generate individual graphs
    graph1 = kg.generate(
        input_data=text1, model="openai/gpt-4o", context="Family relationships"
    )

    graph2 = kg.generate(
        input_data=text2, model="openai/gpt-4o", context="Family relationships"
    )

    # # Aggregate the graphs
    combined_graph = kg.aggregate([graph1, graph2])

    # Cluster the combined graph
    clustered_graph = kg.cluster(
        combined_graph,
        context="Family relationships",
        model="openai/gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    expected_entities = {"Linda", "Joshua", "Josh", "Ben", "Andrew", "Judy"}
    expected_edges = {
        "is mother of",
        "is brother of",
        "is father of",
        "is sister of",
        "is nephew of",
        "is aunt of",
    }
    print(clustered_graph)
    assert clustered_graph.entities.issubset(
        expected_entities
    ) or expected_entities.issubset(clustered_graph.entities)
    assert clustered_graph.edges.issubset(expected_edges) or expected_edges.issubset(
        clustered_graph.edges
    )

    print("\nGraph 1:")
    print("Entities:", graph1.entities)
    print("Relations:", graph1.relations)
    print("Edges:", graph1.edges)

    print("\nGraph 2:")
    print("Entities:", graph2.entities)
    print("Relations:", graph2.relations)
    print("Edges:", graph2.edges)

    print("\nCombined Graph:")
    print("Entities:", combined_graph.entities)
    print("Relations:", combined_graph.relations)
    print("Edges:", combined_graph.edges)

    print("\nClustered Combined Graph:")
    print("Entities:", clustered_graph.entities)
    print("Relations:", clustered_graph.relations)
    print("Edges:", clustered_graph.edges)
    print("Entity Clusters:", clustered_graph.entity_clusters)
    print("Edge Clusters:", clustered_graph.edge_clusters)


def test_conversation(kg: KGGen):
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ]

    graph = kg.generate(
        input_data=messages, model="openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY")
    )
    expected_entities = {"France", "Paris"}
    assert match_subset(graph.entities, expected_entities)
    print(graph)
