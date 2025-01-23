from typing import List
import dspy

class Relations(dspy.Signature):
  """Extract relations between entities from the source text. Relations are (subject, predicate, object) tuples. This is for an extraction task, please be thorough and accurate to the reference text."""
  
  source_text: str = dspy.InputField()
  entities: str = dspy.InputField()
  relations: list[tuple[str, str, str]] = dspy.OutputField()

def get_relations(dspy: dspy.dspy, input_data: str, entities: list[str]) -> List[str]:
  extract = dspy.Predict(Relations)
  result = extract(source_text = input_data, entities = entities)
  filtered_relations = [
    (s, p, o) for s, p, o in result.relations 
    if s in entities and o in entities
  ]
  return filtered_relations