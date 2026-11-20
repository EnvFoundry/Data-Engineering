import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class CLITests(unittest.TestCase):
    def test_installed_cli_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run([sys.executable, "-m", "ai_data", "build", str(EXAMPLES / "local.recipe.json"),
                                     "--output", str(root / "发布")], cwd=root, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))
            manifest = json.loads(result.stdout)
            self.assertEqual(manifest["counts"]["splits"], {"train": 3, "validation": 1, "test": 1})
            verify = subprocess.run([sys.executable, "-m", "ai_data", "verify", str(root / "发布")],
                                    cwd=root, capture_output=True, timeout=30)
            self.assertEqual(verify.returncode, 0, verify.stderr.decode("utf-8"))
            self.assertTrue(json.loads(verify.stdout)["ok"])
            config = root / "server.json"
            connect = subprocess.run([sys.executable, "-m", "ai_data", "connect", "--root", str(root), "--output", str(config)],
                                     cwd=root, capture_output=True, timeout=30)
            self.assertEqual(connect.returncode, 0, connect.stderr.decode("utf-8"))
            value = json.loads(config.read_text(encoding="utf-8"))["mcpServers"]["data-engineering"]
            self.assertEqual(value["command"], sys.executable)
            self.assertEqual(value["args"][-1], str(root.resolve()))

    def test_bad_recipe_returns_json_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bad.json").write_text('{"schema_version":"unsupported"}')
            result = subprocess.run([sys.executable, "-m", "ai_data", "build", str(root / "bad.json"),
                                     "--output", str(root / "output")], capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 2)
            self.assertIn("error", json.loads(result.stderr))
            self.assertFalse((root / "output").exists())


@unittest.skipUnless(importlib.util.find_spec("mcp"), "optional MCP SDK not installed")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_tools_and_workspace(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("records.jsonl", "local.recipe.json"):
                shutil.copyfile(EXAMPLES / name, root / name)
            records_path = root / "records.jsonl"
            rows = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                row["payload"].update(input_ids=[1, 3, 2], loss_mask=[0, 0, 1])
            records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            env = dict(os.environ)
            params = StdioServerParameters(command=sys.executable,
                                           args=["-m", "ai_data", "serve", "--root", str(root)], env=env)
            async with asyncio.timeout(30):
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        names = {t.name for t in (await session.list_tools()).tools}
                        self.assertEqual(names, {"data_capabilities", "data_profile", "data_build", "data_inspect", "data_verify",
                                                 "data_compose", "data_verify_mixture", "data_pack", "data_packed", "data_samples"})
                        resources = {str(resource.uri) for resource in (await session.list_resources()).resources}
                        self.assertEqual(resources, {"data://guides/data-design", "data://guides/mixture-design",
                                                     "data://guides/causal-packing", "data://guides/consumption-restart"})
                        prompts = {prompt.name for prompt in (await session.list_prompts()).prompts}
                        self.assertEqual(prompts, {"design_data_product", "prepare_training_data", "compare_data_recipes"})
                        guide = await session.read_resource("data://guides/causal-packing")
                        self.assertIn("target_ids", guide.contents[0].text)
                        prompt = await session.get_prompt("prepare_training_data", {
                            "mixture": "mixture", "encoding": "authored fixture", "objective": "assistant response tokens"})
                        self.assertTrue(prompt.messages)
                        result = await session.call_tool("data_build", {"recipe": "local.recipe.json", "output": "release"})
                        self.assertFalse(result.isError, result.content)
                        result = await session.call_tool("data_verify", {"path": "release"})
                        self.assertFalse(result.isError, result.content)
                        self.assertTrue(json.loads(result.content[0].text)["ok"])
                        manifest = json.loads((root / "release/manifest.json").read_text(encoding="utf-8"))
                        recipe = {"schema_version": "1.0", "name": "mcp-mixture",
                                  "sources": [{"name": "examples", "path": "release", "release_id": manifest["release_id"],
                                               "split": "train", "weight": 1}], "seed": "mcp-test",
                                  "budget": {"unit": "records", "value": 12}, "shard_records": 5, "exhaustion": "repeat"}
                        (root / "mixture.json").write_text(json.dumps(recipe), encoding="utf-8")
                        mixed = await session.call_tool("data_compose", {"recipe": "mixture.json", "output": "mixture"})
                        self.assertFalse(mixed.isError, mixed.content)
                        checked = await session.call_tool("data_verify_mixture", {"path": "mixture"})
                        self.assertFalse(checked.isError, checked.content)
                        self.assertEqual(json.loads(checked.content[0].text)["records"], 12)
                        packed = await session.call_tool("data_pack", {"mixture": "mixture", "output": "packed",
                            "sequence_length": 4, "tokenizer_name": "authored-fixture", "tokenizer_revision": "1",
                            "loss_mask": "/payload/loss_mask"})
                        self.assertFalse(packed.isError, packed.content)
                        self.assertEqual(json.loads(packed.content[0].text)["counts"]["trainable_tokens"], 12)
                        page = await session.call_tool("data_samples", {"path": "packed", "rank": 1,
                                                                        "world_size": 2, "limit": 1,
                                                                        "include_payload": True})
                        self.assertFalse(page.isError, page.content)
                        block = json.loads(page.content[0].text)["items"][0]
                        self.assertEqual(block["sample_index"], 1)
                        self.assertEqual(block["target_ids"], [3, 2, 3, 2])
                        self.assertEqual(block["loss_mask"], [0, 1, 0, 1])
                        forbidden = await session.call_tool("data_profile", {"path": "../outside.jsonl"})
                        self.assertTrue(forbidden.isError)
                        overwrite = await session.call_tool("data_build", {"recipe": "local.recipe.json", "output": "release"})
                        self.assertTrue(overwrite.isError)


if __name__ == "__main__":
    unittest.main()
