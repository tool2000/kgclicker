# Running MINE

To run MINE:
1. Set your OpenAI API key as an environment variable:
   - **Windows PowerShell:** `$env:OPENAI_API_KEY="your_actual_key_here"`
   - **Linux/Mac:** `export OPENAI_API_KEY="your_actual_key_here"`
2. Use your KG generator to generate a KG from each of the essays found in [`essays.json`](essays.json). The KGs should be JSON files structured in the same way as [`example.json`](example.json).
3. Name these KGs `1.json`, `2.json`, ..., `106.json` in order and place them in the `results/kggen/` folder in this directory.
4. Ensure [`answers.json`](answers.json) exists with the evaluation questions and answers.
5. Run `python evaluation.py`.
6. Look for the files `1_results.json`,..., `106_results.json` in the `results/kggen/` folder.