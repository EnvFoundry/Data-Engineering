"""Exercise the installed wheel and source launcher outside the checkout."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import venv
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    wheel = next((root / "dist").glob(f"ai_data_engineering-{version}-*.whl"))
    with tempfile.TemporaryDirectory(prefix="data installation ") as directory:
        workspace = Path(directory)
        environment = workspace / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                        "--ignore-installed", str(wheel)], cwd=workspace, env=env, check=True,
                       stdout=subprocess.PIPE)
        installed = subprocess.check_output([str(python), "-c", "import ai_data; print(ai_data.__file__)"],
                                            cwd=workspace, env=env, text=True).strip()
        assert Path(installed).resolve().is_relative_to(environment.resolve())
        operations = json.loads(subprocess.check_output([str(python), "-m", "ai_data", "capabilities"],
                                                        cwd=workspace, env=env, text=True))
        assert {"build", "compose", "pack", "samples", "guide"} <= set(operations["operations"])
        subprocess.run([str(python), str(root / "examples/pack_training.py"), "--output", str(workspace / "packed")],
                       cwd=workspace, env=env, check=True, stdout=subprocess.PIPE)
        with zipfile.ZipFile(root / "dist" / f"data-engineering-{version}-plugin.zip") as archive:
            archive.extractall(workspace / "plugin")
        package = workspace / "plugin" / "data-engineering"
        relocated = json.loads(subprocess.check_output([str(python), str(package / "server.py"), "capabilities"],
                                                       cwd=workspace, env=env, text=True))
        assert operations["operations"] == relocated["operations"]
        for filename in ("plugin.json", ".claude-plugin/plugin.json", ".codex-plugin/plugin.json"):
            assert json.loads((package / filename).read_text())["version"] == version
    print(json.dumps({"wheel": wheel.name, "training_path": "passed", "relocated_plugin": "passed"}))


if __name__ == "__main__":
    main()
