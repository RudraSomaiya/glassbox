"""Write data/perplexity.txt: WikiText-2 (raw, test split) when Hugging Face is
reachable, otherwise English prose from the docstrings of Python's standard
library (prints which one it used).

    uv run python perplexity_text.py
"""

import ast
import glob
import io
import os
import sysconfig
import urllib.request

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "perplexity.txt")
WIKITEXT = ("https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
            "wikitext-2-raw-v1/test-00000-of-00001.parquet")


def wikitext():
    import pyarrow.parquet as pq  # only needed on this path
    data = urllib.request.urlopen(WIKITEXT, timeout=30).read()
    return "".join(pq.read_table(io.BytesIO(data)).column("text").to_pylist())


def docstrings():
    out = []
    for f in sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))):
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node)
                if doc and len(doc) > 200:
                    out.append(doc)
    return "\n\n".join(out)


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        text, source = wikitext(), "wikitext-2-raw-v1 test"
    except Exception as e:  # network blocked, or pyarrow missing
        print("wikitext unavailable:", e)
        text, source = docstrings(), "python stdlib docstrings (fallback)"
    with open(OUT, "w") as f:
        f.write(text)
    print(source, len(text), "chars ->", OUT)
