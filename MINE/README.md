# Running MINE

MINE now automatically loads evaluation data from Hugging Face, making it easier to run evaluations without managing local files.

## Quick Start

1. Set your OpenAI API key as an environment variable:
   - **Windows PowerShell:** `$env:OPENAI_API_KEY="your_actual_key_here"`
   - **Linux/Mac:** `export OPENAI_API_KEY="your_actual_key_here"`

2. Use your KG generator to generate a KG from each of the essays. The essays are available on Hugging Face at [kg-gen-evaluation-essays](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-essays), and the evaluation data (questions and answers) will be automatically downloaded from Hugging Face.

3. Name the generated KGs `1.json`, `2.json`, ..., `106.json` in order and place them in the `results/kggen/` folder in this directory.

4. Run `python evaluation.py`.

5. Look for the results in `1_results.json`, ..., `106_results.json` in the `results/kggen/` folder.

## Data Loading

The evaluation script automatically:
- ✅ **Downloads evaluation data from Hugging Face** ([kg-gen-evaluation-answers](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-answers))
- ✅ **Falls back to local files** if Hugging Face is unavailable
- ✅ **Shows clear status messages** about data source

**Source Essays:** Available at [kg-gen-evaluation-essays](https://huggingface.co/datasets/kyssen/kg-gen-evaluation-essays) - use these to generate your knowledge graphs.

## Local Development

If you prefer to use local files or Hugging Face is unavailable:
- Ensure [`answers.json`](answers.json) exists with the evaluation questions and answers
- The script will automatically detect and use local files as fallback