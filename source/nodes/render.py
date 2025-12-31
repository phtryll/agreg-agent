from typing import List
from source.nodes.generate import Derivation


def render_answer(derivation: Derivation) -> str:
    # Collect all output lines in order
    lines: List[str] = []

    # Latin lemma and IPA > modern French lemma and IPA
    header: str = (
        f"{derivation.latin_lemma} {derivation.latin_ipa} "
        f"> {derivation.modern_lemma} {derivation.modern_ipa}"
    )
    lines.append(header)
    lines.append("")

    # Chronological stages of evolution
    for stage in derivation.stages:
        lines.append(stage.period)

        # IPA forms for this period
        for form in stage.forms:
            lines.append(f"  {form}")

        # Phonetic changes (one bullet per phenomenon)
        for change in stage.changes:
            lines.append(f"\t- {change}")

        lines.append("")

    # Remarks: this is more often than not empty... so much for the graphie...
    if derivation.remarks:
        lines.append("Remarques :")
        for remark in derivation.remarks:
            lines.append(f"- {remark}")

    # Join everything into the final rendered answer
    return "\n".join(lines).strip()