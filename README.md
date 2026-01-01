# agreg-agent

A draft AI agent capable of answering agrégation questions on the phonetic evolution of French.

## Installation

Clone the repository:

```bash
git clone https://github.com/phtryll/agreg-agent.git
cd agreg-agent
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Make sure you have Ollama installed and configured on your system. You can find installation instructions at [https://ollama.com/](https://ollama.com/).

## Usage

Run the agent by executing:

```bash
python main.py eval.txt
```

You can change ollama models by passing another one with the `--model` flag.
