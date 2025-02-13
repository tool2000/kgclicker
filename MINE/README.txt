To run MINE please follow these steps:
 1. change "YOUR_OPENAI_KEY" to your actual key in evaluation.py
 2. use your KG generator to generate a KG from each of the essays found in essays.json. The KGs should be json files structured as example.json is. 
 3. name these KGs 1.json, 2.jsoon, ..., 106.json in order and place them in the KGs folder.
 4. run evaluation.py
 5. look for the files 1_result.json,..., 106_result in the KGs folder.