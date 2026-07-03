"""Generate the code reference pages."""

from pathlib import Path

MODULE_ROOT = Path("deep_river")
REFERENCE_ROOT = Path("docs/reference")


def iter_reference_targets():
    for path in sorted(MODULE_ROOT.rglob("*.py")):
        module_path = path.relative_to(MODULE_ROOT).with_suffix("")
        parts = list(module_path.parts)

        if parts[-1] in {"__init__", "__version__", "__main__", " "}:
            continue

        doc_path = Path(*parts).with_suffix(".md")

        if parts[0] == "utils":
            allowed_utils = {"tensor_conversion", "params"}
            if len(parts) >= 2 and parts[1] in allowed_utils:
                doc_path = Path(f"{parts[1]}.md")
            else:
                continue
            if len(parts) > 2:
                continue

        yield path, parts, doc_path


def main() -> None:
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)

    for path, parts, doc_path in iter_reference_targets():
        full_doc_path = REFERENCE_ROOT / doc_path
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)

        identifier = ".".join(parts)
        content = f"::: deep_river.{identifier}\n"
        full_doc_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
