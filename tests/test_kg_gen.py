from src.kg_gen import KGGen
import os
from dotenv import load_dotenv

if __name__ == "__main__":
  # Load environment variables
  load_dotenv()
  
  # Example usage
  kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"))
  
  # Generate a simple graph from text
  text = "Advil (ibuprofen) reduces inflammation and pain."
  graph = kg.generate(
    input_data=text,
    model="gpt-4o"
  )
  print(graph)