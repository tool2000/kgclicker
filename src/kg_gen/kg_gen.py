from typing import Union, List, Dict, Tuple, Optional
from openai import OpenAI
from pydantic import BaseModel, model_validator, Field
from .steps._1_get_entities import get_entities
from .steps._2_get_relations import get_relations
import dspy
import json
import os

# ~~~ DATA STRUCTURES ~~~

class Graph(BaseModel):
  entities: set[str] = Field(..., description="All entities including additional ones from response")
  edges: set[str] = Field(..., description="All edges")
  relations: set[Tuple[str, str, str]] = Field(..., description="List of (subject, predicate, object) triples")

  @model_validator(mode='after')
  def validate_entities_in_relations(self) -> 'Graph':
    entities = set(self.entities)
    edges = set(self.edges)
    for subj, pred, obj in self.relations:
      if subj not in entities:
        raise ValueError(f"Relation subject '{subj}' not in entities")
      if obj not in entities:
        raise ValueError(f"Relation object '{obj}' not in entities")
      if pred not in edges:
        raise ValueError(f"Relation pred '{pred}' not edges")
    return self

  
# ~~~ KGGEN ~~~

class KGGen:
  def __init__(self, api_key: str):
    """Initialize KGGen with OpenAI API key.
    
    Args:
        api_key (str): OpenAI API key for making model calls
    """
    self.api_key = api_key
    self.dspy = dspy
    
  def generate(
    self,
    input_data: Union[str, List[Dict]],
    model: str,
    context: Optional[str] = None,
    example_relations: Optional[Union[
      List[Tuple[str, str, str]],
      List[Tuple[Tuple[str, str], str, Tuple[str, str]]]
    ]] = None,
    chunk_size: Optional[int] = None,
    node_labels: Optional[List[str]] = None,
    edge_labels: Optional[List[str]] = None,
    ontology: Optional[List[Tuple[str, str, str]]] = None,
    output_folder: Optional[str] = None
  ) -> Graph:
    """Generate a knowledge graph from input text or messages.
    
    Args:
        input_data: Text string or list of message dicts
        model: Name of OpenAI model to use
        chunk_size: Max size of text chunks to process
        context: Description of data context
        example_relations: Example relationship tuples
        node_labels: Valid node label strings
        edge_labels: Valid edge label strings
        ontology: Valid node-edge-node structure tuples
        output_folder: Path to save partial progress
        
    Returns:
        Generated knowledge graph
    """
    # TODO
    if chunk_size  or node_labels or edge_labels or ontology or example_relations or context:
      raise ValueError("chunk_size, node_labels, edge_labels, ontology, or example_relations are not supported parameters yet")
    
    # TODO
    if isinstance(input_data, list):
      raise ValueError("Message array input_data is not supported yet")
    
    self.lm = dspy.LM(f'openai/{model}', api_key=self.api_key)
    self.dspy.configure(lm=self.lm)
    
    # relations = get_relations(entities)
    entities = get_entities(self.dspy, input_data)
    relations = get_relations(self.dspy, input_data, entities)
    
    
    graph = Graph(
      entities = entities,
      relations = relations,
      edges = {relation[1] for relation in relations}
    )
    
    if output_folder:
      os.makedirs(output_folder, exist_ok=True)
      output_path = os.path.join(output_folder, 'graph.json')
      
      graph_dict = {
        'entities': list(entities),
        'relations': list(relations),
        'edges': list(graph.edges)
      }
      
      with open(output_path, 'w') as f:
        json.dump(graph_dict, f, indent=2)
      
    return graph
    
  def cluster():
    pass
    
  def aggregate():
    pass
    
