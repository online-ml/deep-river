"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("deep_river").rglob("*.py")):
    module_path = path.relative_to("deep_river").with_suffix("")
    parts = list(module_path.parts)

    # Skip dunder and special modules
    if parts[-1] in {"__init__", "__version__", "__main__", " "}:
        continue

    # Default doc path (mirrors package structure)
    doc_path = path.relative_to("deep_river").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)
    nav_key_parts = parts[:]  # copy for potential modification

    # Handle utils filtering
    if parts[0] == "utils":
        allowed_utils = {"tensor_conversion", "params"}
        if len(parts) >= 2 and parts[1] in allowed_utils:
            # Flatten: expose as top-level pages (no utils section)
            top_name = parts[1]
            nav_key_parts = [top_name]
            doc_path = Path(f"{top_name}.md")
            full_doc_path = Path("reference", doc_path)
        else:
            # Skip any other utils module
            continue
        # Skip deeper nesting even for allowed modules (not currently used)
        if len(parts) > 2:
            continue

    nav[nav_key_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w+") as fd:
        identifier = ".".join(parts)
        print(f"::: deep_river.{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w+") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
