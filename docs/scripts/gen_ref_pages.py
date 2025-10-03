"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("deep_river").rglob("*.py")):
    module_path = path.relative_to("deep_river").with_suffix("")
    doc_path = path.relative_to("deep_river").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = list(module_path.parts)

    # Skip dunder and special modules
    if parts[-1] in {"__init__", "__version__", "__main__", " "}:
        continue

    # Exclude almost all utils.* modules except the explicitly whitelisted ones
    if parts[0] == "utils":
        allowed_utils = {"tensor_conversion", "params"}
        # parts example: ["utils", "params"]
        if len(parts) >= 2 and parts[1] not in allowed_utils:
            continue
        # Also skip deeper nested utils modules unless first submodule is allowed
        if len(parts) >= 2 and parts[1] in allowed_utils and len(parts) > 2:
            # currently no deeper structure needed in docs; keep only the top-level module page
            continue

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w+") as fd:
        identifier = ".".join(parts)
        print(f"::: deep_river.{identifier}", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w+") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
