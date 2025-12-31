from typing import List
from .generate import Derivation


def verify_derivation(derivation: Derivation) -> List[str]:
    # Collect all structural issues detected
    issues: List[str] = []

    # Global checks
    if not derivation.latin_lemma or not derivation.latin_ipa:
        issues.append("verify: missing Latin lemma or IPA")

    if not derivation.modern_lemma or not derivation.modern_ipa:
        issues.append("verify: missing modern French lemma or IPA")

    if not derivation.stages:
        issues.append("verify: no chronological stages defined")
        return issues

    # Checks for each stage
    for index, stage in enumerate(derivation.stages, start=1):
        if not stage.period.strip():
            issues.append(f"verify: empty period label at stage {index}")

        if not stage.forms:
            issues.append(f"verify: no IPA forms at stage {index}")
        else:
            for form in stage.forms:
                if not (form.startswith("[") and form.endswith("]")):
                    issues.append(
                        f"verify: IPA form not bracketed at stage {index}: {form}"
                    )

        if not stage.changes:
            issues.append(f"verify: no phonetic changes at stage {index}")
        else:
            for change in stage.changes:
                if not change.strip():
                    issues.append(
                        f"verify: empty phonetic change at stage {index}"
                    )

    # Also verify the remarks
    if derivation.remarks is not None:
        for remark in derivation.remarks:
            if not remark.strip():
                issues.append("verify: empty remark")

    return issues