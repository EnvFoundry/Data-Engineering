"""Package the maintained source and agent entry points together."""

from pathlib import Path
import tomllib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    destination = root / "dist"
    destination.mkdir(exist_ok=True)
    package = destination / f"data-engineering-{version}-plugin.zip"
    entries = ["src", "skills", "docs", "examples", ".agents", ".claude-plugin", ".codex-plugin",
               ".mcp.json", "plugin.json", "mcp.json", "mcp.codex.json", "server.py", "pyproject.toml",
               "README.md", "README.zh-CN.md", "LICENSE"]
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            path = root / entry
            for file in sorted(path.rglob("*")) if path.is_dir() else [path]:
                if file.is_file() and not any(part == "__pycache__" or part.endswith(".egg-info") for part in file.parts):
                    archive.write(file, "data-engineering/" + file.relative_to(root).as_posix())
    print(package)


if __name__ == "__main__":
    main()
