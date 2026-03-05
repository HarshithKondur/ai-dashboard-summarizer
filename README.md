# 📊 AI Dashboard Summarizer

Automatically generate charts from any insurance dataset and get a plain-English AI-powered executive summary — straight in your terminal.

## What It Does
- Loads car insurance claims CSV data
- Generates 4 matplotlib visualizations saved as PNG files
- Sends key statistics to Claude API (claude-sonnet-4-6)
- Prints a plain English executive summary to the terminal

## Requirements
- Python 3.9+
- Anthropic API key

## How to Run
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
python3 dashboard_summarizer.py
```
