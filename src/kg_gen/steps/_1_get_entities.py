from typing import List
import dspy 

class Entities(dspy.Signature):
  """Extract key entities from the source text. Extracted entities are subjects or objects. This is for an extraction task, please be thorough and accurate to the reference text."""
  
  source_text: str = dspy.InputField()  
  entities: list[str] = dspy.OutputField()
  

def get_entities(dspy: dspy.dspy, input_data: str, context: str = None) -> List[str]:
  extract = dspy.Predict(Entities)
  result = extract(source_text = input_data)
  return result.entities

