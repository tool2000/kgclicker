# kg-gen: Knowledge Graph Generation from Any Text

Welcome! `kg-gen` helps you generate knowledge graphs from any source text using AI. It can process both small and large text inputs, and it can also handle messages in a conversation format.

Why generate knowledge graphs? `kg-gen` is great if you want to:
- Create a graph to assist with RAG (Retrieval-Augmented Generation)
- Create graph synthetic data for model training and testing
- Structure any text into a graph 
- Analyze the relationships between concepts in your source text

<!-- TODO: [Read the paper](link) to see how it works under the hood. -->

We support all model providers supported by [LiteLLM](https://docs.litellm.ai/docs/providers). We also use [DSPy](https://dspy.ai/) for structured output generation. 

## Quick start

Install the module:
```bash
pip install kg-gen
```

Then import and use `kg-gen`. You can provide your text input in one of two formats:
1. A single string  
2. A list of Message objects (each with a role and content)

Below are some example snippets:
```python
from kg_gen import KGGen

# Initialize the KGGen
kg = KGGen()

# EXAMPLE 1: Single string with model
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father. Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt."
graph_1 = kg.generate(
  input_data=text_input,
  model="openai/gpt-4o"
  api_key="<OPENAI_API_KEY>" # Optional if this is set in your environment
)
# Output: 
# entities={'Linda', 'Judy', 'Ben', 'Andrew', 'Josh'} 
# edges={'is sister of', 'is father of', 'is aunt of', 'is brother of', 
# 'is mother of', 'is nephew of'} 
# relations={('Judy', 'is aunt of', 'Josh'), ('Josh', 'is nephew of', 'Judy'), 
# ('Andrew', 'is father of', 'Josh'), ('Ben', 'is brother of', 'Josh'), 
# ('Judy', 'is sister of', 'Andrew'), ('Linda', 'is mother of', 'Josh')}

# EXAMPLE 2: Messages array with role filtering
messages = [
  {"role": "user", "content": "What is the capital of France?"}, 
  {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(
  input_data=messages,
  model="openai/gpt-4o-mini"
)
# Output: 
# entities={'Paris', 'France'} 
# edges={'has capital'} 
# relations={('France', 'has capital', 'Paris')}
```

## Message Array Processing
When processing message arrays, kg-gen:
1. Preserves the role information from each message
2. Maintains message order and boundaries
3. Can extract entities and relationships:
   - Between concepts mentioned in messages
   - Between speakers (roles) and concepts
   - Across multiple messages in a conversation

For example, given this conversation:
```python
messages = [
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "The capital of France is Paris."}
]
```

The generated graph might include entities like:
- "user"
- "assistant" 
- "France"
- "Paris"

And relations like:
- (user, "asks about", "France")
- (assistant, "states", "Paris")
- (Paris, "is capital of", "France")

## License
The MIT License.