from src.kg_gen import KGGen
import os
from dotenv import load_dotenv

if __name__ == "__main__":
  # Load environment variables
  load_dotenv()
  
  # Example usage
  kg = KGGen()
  
  # Generate a simple graph
  text = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father. Judy is Andrew's sister. Josh is Judy's nephew. Judy is Josh's aunt." # text option
  # text = [
  #   {"role": "user", "content": "Who is in Josh's family?"},
  #   {"role": "assistant", "content": "Josh's family includes his mother Linda, father Andrew, brother Ben, and aunt Judy who is Andrew's sister."}
  # ] # messages array
  graph = kg.generate(
    input_data=text,
    model="openai/gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
  )
  print(graph)