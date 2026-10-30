"""Constrained output: the server only lets the model write JSON that matches
the schema, so the reply always parses and validates.

    cargo run --release -p glassbox-server -- serve --model models/SmolLM2-360M-Instruct --dtype q8_0
    uv run --with openai python examples/json_schema.py
"""

import json

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 40},
        "year": {"type": "integer"},
        "genres": {"type": "array", "items": {"enum": ["drama", "comedy", "sci-fi", "thriller"]}, "maxItems": 3},
        "recommended": {"type": "boolean"},
    },
    "required": ["name", "year", "genres", "recommended"],
}

reply = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "Describe the film Interstellar as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "film", "schema": schema}},
    max_tokens=120,
    temperature=0,
)
film = json.loads(reply.choices[0].message.content)
print(json.dumps(film, indent=2))
