"""Tokenizer test strings: English, code, emoji, Hindi, Hinglish, whitespace edge cases."""

import glob
import os
import random
import sysconfig

ENGLISH = [
    "The quick brown fox jumps over the lazy dog.",
    "I can't believe it's already 2024; we'll see what happens.",
    "Dr. Smith's lab measured 3.14159 radians at 10:45 PM.",
    "She said, \"Don't touch that!\" and left.",
    "The KV cache stores keys and values for every past token.",
    "Hello, world! How are you doing today?",
    "It costs $1,299.99 (tax included) - roughly 15% off.",
    "E-mail me at someone@example.com or visit https://example.org/a?b=c&d=e.",
    "They're here, you've seen it, he'd know, I'm sure, we'll go.",
    "ALL CAPS SENTENCE WITH NUMBERS 12345 AND symbols #@!",
]
CODE = [
    "def forward(self, x):\n    return self.w @ x + self.b\n",
    "fn main() {\n    println!(\"{}\", 1 + 2);\n}\n",
    "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }",
    "const x = await fetch(`/api/${id}`).then(r => r.json());",
    "SELECT name, COUNT(*) FROM users WHERE age >= 18 GROUP BY name;",
    "if __name__ == \"__main__\":\n\tmain()\n",
    "<div class=\"card\"><span>{{ title }}</span></div>",
    "x = [i**2 for i in range(10) if i % 2 == 0]",
    "#include <stdio.h>\nint main(void) { return 0; }",
    "    // indented comment\n        deeper();\n",
]
EMOJI = [
    "I love this 😀😃😄!", "🚀 launch at T-10", "👨‍👩‍👧‍👦 family", "flags 🇮🇳🇺🇸",
    "✨ sparkle ✨", "thumbs 👍🏽 up", "🙂🙃🫠", "café ☕ naïve résumé", "Ω ≈ ∑ x² ≥ √2", "日本語のテキスト",
]
HINDI = [
    "नमस्ते, आप कैसे हैं?", "मुझे हिंदी बोलना पसंद है।", "भारत एक विशाल देश है।",
    "आज मौसम बहुत अच्छा है", "मेरा नाम रुद्र है और मैं इंजीनियर हूँ।",
    "क्या आप मेरी मदद कर सकते हैं?", "१२३ ४५६ संख्याएँ", "दिल्ली से मुंबई की दूरी",
]
HINGLISH = [
    "yaar kal movie dekhne chalein?", "bhai ye code chal hi nahi raha 😅",
    "aaj office mein bahut kaam tha", "kya scene hai bro, sab theek?",
    "mujhe ek chai aur do samosa chahiye", "kal meeting 10 baje hai, late mat hona",
    "ye model bahut chhota hai but kaam karta hai", "arre wah! kya baat hai",
]
WHITESPACE = [
    "", " ", "  ", "\n", "\n\n", "\t", " \t \n ", "a  b", "a   b", "a\n\nb", "trailing   ",
    "   leading", "\r\n", "x \n y", "  \n\n  hi", "tabs\t\tand spaces  ", "mixed  nbsp",
    "zero​width", "end.\n", "'s 're 've 'll 'd 't 'm", "''quoted''", "123456789",
    "1.5e-10", "a1b2c3", "٣٤٥ arabic digits", "<|im_start|>user\nhi<|im_end|>\n",
    "<|endoftext|>", "text<|im_end|>more", "<|im_start|><|im_start|>",
]


def stdlib_lines(n, seed):
    lib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(lib, "*.py")))
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        try:
            lines = open(rng.choice(files), encoding="utf-8").read().splitlines()
        except UnicodeDecodeError:
            continue
        if not lines:
            continue
        i = rng.randrange(len(lines))
        out.append("\n".join(lines[i:i + rng.randint(1, 4)]))
    return out


def strings():
    """Deterministic list of at least 2,000 test strings."""
    rng = random.Random(1234)
    base = ENGLISH + CODE + EMOJI + HINDI + HINGLISH + WHITESPACE
    out = list(base)
    pools = [ENGLISH, CODE, EMOJI, HINDI, HINGLISH, WHITESPACE]
    for _ in range(1200):
        parts = [rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 3))]
        sep = rng.choice([" ", "\n", "", "  ", "\t", ". "])
        out.append(sep.join(parts))
    out += stdlib_lines(900, 7)
    return out


if __name__ == "__main__":
    s = strings()
    print(len(s), sum(len(x) for x in s))
