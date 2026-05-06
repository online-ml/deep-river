from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover
    Version = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a built docs site into a versioned site tree."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--alias", action="append", default=[])
    parser.add_argument("--default", dest="default_target")
    return parser.parse_args()


def copy_site(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def write_redirect(path: Path, target: str) -> None:
    path.write_text(
        """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta http-equiv=\"refresh\" content=\"0; url={target}\">
    <link rel=\"canonical\" href=\"{target}\">
    <script>location.replace({target_json})</script>
    <title>Redirecting...</title>
  </head>
  <body>
    <p>Redirecting to <a href=\"{target}\">{target}</a>.</p>
  </body>
</html>
""".format(target=target, target_json=json.dumps(target)),
        encoding="utf-8",
    )


def load_versions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def version_sort_key(entry: dict) -> tuple:
    version = entry["version"]
    if version == "dev":
        return (0, 0, version)

    normalized = version[1:] if version.startswith(("v", "V")) else version
    if Version is None:  # pragma: no cover
        return (1, normalized, version)

    try:
        parsed = Version(normalized)
    except Exception:
        return (2, 0, version)
    return (1, parsed, version)


def sort_versions(entries: list[dict]) -> list[dict]:
    development = [entry for entry in entries if entry["version"] == "dev"]
    releases = [entry for entry in entries if entry["version"] != "dev"]
    releases.sort(key=version_sort_key, reverse=True)
    return development + releases


def update_versions(
    versions_path: Path,
    version: str,
    title: str,
    aliases: list[str],
) -> None:
    entries = load_versions(versions_path)
    current = None

    for entry in entries:
        entry_aliases = entry.get("aliases", [])
        entry["aliases"] = [alias for alias in entry_aliases if alias not in aliases]
        if entry["version"] == version:
            current = entry

    if current is None:
        current = {"version": version}
        entries.append(current)

    current["title"] = title
    current["aliases"] = aliases

    versions_path.write_text(
        json.dumps(sort_versions(entries), indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    target_dir = Path(args.target_dir).resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")
    if not target_dir.is_dir():
        raise SystemExit(f"Target directory does not exist: {target_dir}")

    copy_site(source_dir, target_dir / args.version)
    for alias in args.alias:
        copy_site(source_dir, target_dir / alias)

    update_versions(
        target_dir / "versions.json",
        version=args.version,
        title=args.title,
        aliases=args.alias,
    )

    if args.default_target:
        write_redirect(target_dir / "index.html", f"./{args.default_target}/")
        write_redirect(target_dir / "404.html", f"./{args.default_target}/")


if __name__ == "__main__":
    main()
