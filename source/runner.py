from source.nodes.retriever import retrieve_passages
from source.state import AgentState
from source.nodes.classify import classify
from source.nodes.transcribe import transcribe
from source.nodes.triggers import infer_triggers
from source.nodes.queries import infer_phonological_questions
from source.nodes.planner import plan
from source.nodes.generate import generate
from source.nodes.render import render_answer
from source.nodes.verify import verify_derivation
from .state import AgentState


def _print_step(step: str) -> None:
    print(f"\n[{step.upper()}]")


def run_agent(question: str, model: str, index) -> AgentState:

    # Initialize the agent state incrementally
    state: AgentState = {
        "question": question,
        "lexical_items": [],
        "ipa": {},
        "triggers": [],
        "queries": [],
        "retrieved": {},
        "plan": [],
        "structured_answer": None,
        "answer": "",
        "issues": [],
    }

    # Step 0: pass the question
    _print_step("load question")
    print(state["question"])

    # Step 1: Classify node
    _print_step("classify")
    state = classify(state, model)
    print("Lexical items:", state.get("lexical_items"))

    # Step 2: Transcribe node
    _print_step("transcribe")
    state = transcribe(state, model)
    print("IPA:", state.get("ipa"))

    # Step 3: Infer hypotheses node
    _print_step("triggers")
    state = infer_triggers(state, model)
    for trigger in state.get("triggers", []):
        print(f"- {trigger}")

    # Step 4: Generate queries node
    _print_step("planner questions")
    state = infer_phonological_questions(state, model)
    for index_, question in enumerate(state.get("queries", []), start=1):
        print(f"{index_}. {question}")

    # Step 5: Retrieve passages node
    _print_step("retriever")
    state = retrieve_passages(state, index)
    for question, passages in state.get("retrieved", {}).items():
        print("Q:", question)
        print(f"Retrieved passages: {len(passages)}")
        for i, passage in enumerate(passages, start=1):
            print(f"[{i}]\t{passage}")

    # Step 6: Planner node
    _print_step("planner synthesis")
    state = plan(state, model)
    for stage in state.get("plan", []):
        print(stage)

    if not state.get("plan"):
        state["issues"].append("runner: planning failed")
        return state

    # Step 7: Generate node
    _print_step("generate structured derivation")
    state = generate(state, model)

    if state.get("structured_answer") is None:
        state["issues"].append("runner: structured generation failed")
        return state

    # Step 8: Verify node
    _print_step("verify")
    structured = state.get("structured_answer")
    if structured is None:
        state["issues"].append("runner: no structured answer to verify")
        return state

    verification_issues = verify_derivation(structured)
    state["issues"].extend(verification_issues)

    if verification_issues:
        print("Verification issues detected:")
        for issue in verification_issues:
            print(f"- {issue}")

    # Step 9: Render final answer
    _print_step("render")
    state["answer"] = render_answer(structured)

    return state