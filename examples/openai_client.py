"""Talk to a Glassbox server with the official OpenAI Python client.

    cargo run --release -p glassbox-server -- serve --model models/SmolLM2-135M-Instruct --dtype q8_0
    uv run --with openai python examples/openai_client.py
"""

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

model = client.models.list().data[0].id
print("model:", model)

reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Name three planets."}],
    max_tokens=40,
    temperature=0,
)
print("reply:", reply.choices[0].message.content)
print("usage:", reply.usage)

print("streamed: ", end="", flush=True)
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count from one to five."}],
    max_tokens=30,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
