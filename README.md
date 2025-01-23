# kg-gen: Knowledge Graph Generation from Unstructured Text

Welcome! `kg-gen` helps you generate knowledge graphs from some source text using AI. It can process both small and large text inputs, and it can also handle messages in a conversation format.

Why generate knowledge graphs? `kg-gen` is great if you want to:
- Create a graph to assist with RAG (Retrieval-Augmented Generation)
- Create graph synthetic data for model training and testing
- Structure any text into a graph 
- Analyze the relationships between concepts in your source text

<!-- TODO: [Read the paper](link) to see how it works under the hood. -->

## Quick start

Install the module:
```bash
pip install kg-gen
```

Then import and use `kg-gen`. You can provide your text input in one of two formats:
1. A single string  
2. A list of Message objects (each with a role and content)

You can also optionally pass:
- A `context` string describing the type of data you’re handling
- A list of `node_labels` to specify node types
- A list of `edge_labels` to specify permissible relationships
- An `ontology` describing valid node-edge-node tuples

Below are some example snippets:
```python
from kg_gen import KGGen

# Initialize the KGGen with your OpenAI API key
kg = KGGen(openai_api_key="OPENAI_API_KEY")

# EXAMPLE 1: Single string with model
text_input = "Hello world. This is a simple text."
graph_1 = kg.generate(
  input_data=text_input,
  model="OPENAI_MODEL"
)

# EXAMPLE 2: Large text with chunk size or delimiter
large_text = """Your very large text content goes here..."""
graph_2 = kg.generate(
  input_data=large_text,
  model="OPENAI_MODEL",
  chunk_size=1000
)

# EXAMPLE 3: Messages array
messages = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I’m fine, thank you!"}
]
graph_3 = kg.generate(
  input_data=messages,
  model="OPENAI_MODEL"
)

# EXAMPLE 4: Single string with labels/ontology specified
fourth_text_input = "Advil (ibuprofen) reduces hormones that cause inflammation and pain."
graph_4 = kg.generate(
  input_data=fourth_text_input,
  model="OPENAI_MODEL",
  context="medical data",
  ontology=[(("Advil", "drug"), "reduces", ("inflammatory_hormones", "symptom"))],
  node_labels=["drug", "symptom"],
  edge_labels=["reduces", "treats"]
)
```

## Documentation

### `generate` method

The main function you will call is `generate`, which has the following parameters:
- input_data: Union[str, List[Dict[Message]]] - Can be a string or a list of message objects that adhere to OpenAI's message format.
- model: str - A string representing the model name (e.g. "gpt-4o").
- chunk_size: Optional[int] - Integer specifying maximum chunk size (default: 1000).
- context: Optional[str] - A string describing the data context (e.g. "medical data").
- example_relations: Optional[ Union[ List[Tuple[str, str, str]], List[Tuple[(str, str), str, (str, str)]] ]] - A list of tuples specifying example relationships. If you want to specify node labels, use the second format.
- node_labels: Optional[List[str]] - A list of strings specifying labels for nodes in the graph.
- edge_labels: Optional[List[str]] - A list of strings specifying permissible edge/relationship labels.
- ontology: Optional[List[Tuple[str, str, str]]] - A list of node-edge-node tuples defining allowable structures. Order matters.
- output_folder: Optional[str] - A string specifying a folder path where partial progress can be saved.

Example call that uses all parameters:
```python
graph = kg.generate(
  input_data="Some text...",
  model="OPENAI_MODEL",
  chunk_size=500,
  context="generic",
  example_relations=[(("node1", "label1"), "relation", ("node2", "label2"))],
  node_labels=["label1", "label2"],
  edge_labels=["relation", "another_relation"],
  ontology=[("label1", "relation", "label2")],
  output_folder="temp_folder"
)
```

## For saving midway progress for massive datasets

You may specify an output file to save halfway progress for a generated graph:
```python
big_graph = kg.generate(
  input_data=large_text,
  model="OPENAI_MODEL",
  output_folder="graph_output",
)
```
This output folder tracks the progress of graph generation. You will be able to see which chunks have been processed and which are still pending.

Note that if you pass in `output_folder`, `input_data` will be duplicated to the output folder as a text file. If future calls to the same `output_folder` have a different `input_data`, then this will error out.

## For step by step use

You may also use substeps from the `KGGen` class directly to generate graphs step by step:
```python
from kg_gen import KGGen

# Example of step-by-step usage (a simple placeholder):
kg = KGGen(openai_api_key="YOUR_API_KEY")
# TODO
```

## Examples

Check out `tests/data` for all example inputs!  
- Medical textbook passage
- Wikipedia question answer pairs
- Wikipedia articles
- Logical reasoning over natural language passage

## Validating graph quality
<!-- TODO -->

## Todos
- [ ] chunk size
- [ ] node_labels, edge_labels, ontology
- [ ] example_relations
- [ ] cluster
- [ ] aggregate

## Other things to do 
- [ ] Strict mode where only exact matches from the text are allowed as nodes or edges
- [ ] Add multi-threading to speed up generation
- [ ] Add verifier that checks for graph quality after generation
- [ ] Convert output graph to PyTorch Geometric graph format
- [ ] Track cost of generation
- [ ] Given a messages array, specify which roles to use for node or edge generation, or both
- [ ] Generate a recommended ontology given text

## License
The MIT License.