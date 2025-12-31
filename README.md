# agreg-agent

Un brouillon d'agent IA capable de répondre à des questions d’agrégation sur l'évolution phonétique du français.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/agreg-agent.git
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
