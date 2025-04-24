from typing import List, Literal
import dspy
from pydantic import BaseModel, create_model


def get_relations(dspy: dspy.dspy, input_data: str, entities: list[str], is_conversation: bool = False) -> List[str]:
  
  class Relation(BaseModel):
    "Knowledge graph subject-predicate-object tuple"
    subject: Literal[tuple(entities)]
    predicate: str
    object: Literal[tuple(entities)]
    
  class ExtractTextRelations(dspy.Signature):
    """Extract subject-predicate-object triples from the source text. Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
    This is for an extraction task, please be thorough, accurate, and faithful to the reference text."""
    
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.InputField()
    relations: list[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples. Be thorough.")

  class ExtractConversationRelations(dspy.Signature):
    """Extract subject-predicate-object triples from the conversation, including:
    1. Relations between concepts discussed
    2. Relations between speakers and concepts (e.g. user asks about X)
    3. Relations between speakers (e.g. assistant responds to user)
    Subject and object must be from entities list. Entities provided were previously extracted from the same source text.
    This is for an extraction task, please be thorough, accurate, and faithful to the reference text.
    """
    
    source_text: str = dspy.InputField()
    entities: list[str] = dspy.InputField()
    relations: list[Relation] = dspy.OutputField(desc="List of subject-predicate-object tuples where subject and object are exact matches to items in entities list. Be thorough")
  if is_conversation:
    extract = dspy.Predict(ExtractConversationRelations)
  else:
    extract = dspy.Predict(ExtractTextRelations)
    
  result = extract(source_text=input_data, entities=entities)
  # Assuming result.relations is a list of Relation objects
  filtered_relations = [
    (rel.subject, rel.predicate, rel.object) for rel in result.relations 
    if rel.subject in entities and rel.object in entities
  ]
  return filtered_relations