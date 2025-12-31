import argparse
from source.runner import run_agent
from source.chunks import load_pdf_chunks
from source.nodes.retriever import EmbeddingIndex


def read_question(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def main() -> None:

    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question_file",
        type=str,
        help="Path to txt file containing the question"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1",
        help="Ollama model name")
    args = parser.parse_args()

    # Read question
    question: str = read_question(args.question_file)

    # Load and index the book, also select embedding model
    chunks = load_pdf_chunks("data/laborderie.pdf")
    index = EmbeddingIndex(model_name="cross-encoder/ms-marco-MiniLM-L12-v2", chunks=chunks)

    # Run the agent
    state = run_agent(question=question, model=args.model, index=index)

    # Print final answer
    print("\n[ANSWER]")
    print(state["answer"])

    # Print issues
    if state["issues"]:
        print("\n[WARNINGS]")
        for issue in state["issues"]:
            print(f"\t- {issue}")


if __name__ == "__main__":
    main()