//! Byte-level BPE, read from a Hugging Face `tokenizer.json`.
//!
//! Supports the pieces SmolLM2 (and GPT-2 style models) use: added/special
//! tokens, `Sequence` of `Digits` / `Split` / `ByteLevel` pre-tokenizers, and a
//! BPE model with ranked merges.

use fancy_regex::Regex;
use serde_json::Value;
use std::collections::HashMap;

const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

enum PreStep {
    /// Every digit becomes its own piece.
    Digits,
    /// Regex matches and the gaps between them both become pieces.
    Split(Regex),
}

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<String>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    added: Vec<(String, u32)>,
    special: Vec<bool>,
    steps: Vec<PreStep>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    ignore_merges: bool,
    pub eos_id: u32,
}

#[derive(Debug)]
pub struct TokenizerError(pub String);

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "tokenizer: {}", self.0)
    }
}
impl std::error::Error for TokenizerError {}

fn err(msg: impl Into<String>) -> TokenizerError {
    TokenizerError(msg.into())
}

/// GPT-2's reversible map from bytes to printable unicode characters.
fn bytes_to_unicode() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        out[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    out
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TokenizerError> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| err(e.to_string()))?;
        Self::from_json(&text)
    }

    pub fn from_json(text: &str) -> Result<Self, TokenizerError> {
        let j: Value = serde_json::from_str(text).map_err(|e| err(e.to_string()))?;
        let model = &j["model"];
        if model["type"] != "BPE" {
            return Err(err("only BPE models are supported"));
        }
        if !j["normalizer"].is_null() {
            return Err(err("normalizers are not supported"));
        }
        let mut vocab: HashMap<String, u32> = model["vocab"]
            .as_object()
            .ok_or_else(|| err("missing vocab"))?
            .iter()
            .map(|(k, v)| (k.clone(), v.as_u64().unwrap() as u32))
            .collect();
        let mut added = Vec::new();
        for t in j["added_tokens"].as_array().into_iter().flatten() {
            let content = t["content"].as_str().unwrap().to_string();
            let id = t["id"].as_u64().unwrap() as u32;
            vocab.insert(content.clone(), id);
            added.push((content, id));
        }
        // Longest first, so the scan in `split_added` prefers the longest match.
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        let size = vocab.values().max().map_or(0, |m| m + 1) as usize;
        let mut tokens = vec![String::new(); size];
        for (k, &v) in &vocab {
            tokens[v as usize] = k.clone();
        }
        let mut special = vec![false; size];
        for (_, id) in &added {
            special[*id as usize] = true;
        }

        let mut merges = HashMap::new();
        for (rank, m) in model["merges"].as_array().into_iter().flatten().enumerate() {
            let (a, b) = match m {
                Value::String(s) => {
                    let (a, b) = s.split_once(' ').ok_or_else(|| err("bad merge"))?;
                    (a.to_string(), b.to_string())
                }
                Value::Array(p) => (p[0].as_str().unwrap().to_string(), p[1].as_str().unwrap().to_string()),
                _ => return Err(err("bad merge")),
            };
            let merged = format!("{a}{b}");
            let (Some(&ia), Some(&ib), Some(&im)) = (vocab.get(&a), vocab.get(&b), vocab.get(&merged)) else {
                return Err(err(format!("merge {a} {b} not in vocab")));
            };
            merges.entry((ia, ib)).or_insert((rank as u32, im));
        }

        let mut steps = Vec::new();
        let pre = &j["pre_tokenizer"];
        let list: Vec<&Value> = match pre["type"].as_str() {
            Some("Sequence") => pre["pretokenizers"].as_array().unwrap().iter().collect(),
            Some(_) => vec![pre],
            None => vec![],
        };
        for p in list {
            match p["type"].as_str().unwrap_or("") {
                "Digits" if p["individual_digits"] == true => steps.push(PreStep::Digits),
                "ByteLevel" => {
                    if p["add_prefix_space"] == true {
                        return Err(err("ByteLevel add_prefix_space is not supported"));
                    }
                    if p["use_regex"] != false {
                        steps.push(PreStep::Split(Regex::new(GPT2_PATTERN).unwrap()));
                    }
                }
                "Split" if p["behavior"] == "Isolated" && p["invert"] != true => {
                    let pat = p["pattern"]["Regex"]
                        .as_str()
                        .ok_or_else(|| err("Split needs a Regex pattern"))?;
                    steps.push(PreStep::Split(Regex::new(pat).map_err(|e| err(e.to_string()))?));
                }
                other => return Err(err(format!("unsupported pre-tokenizer {other}"))),
            }
        }

        let byte_to_char = bytes_to_unicode();
        let char_to_byte = byte_to_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let eos_id = ["<|im_end|>", "<|endoftext|>", "</s>"]
            .iter()
            .find_map(|t| vocab.get(*t).copied())
            .unwrap_or(0);
        Ok(Tokenizer {
            vocab,
            tokens,
            merges,
            added,
            special,
            steps,
            byte_to_char,
            char_to_byte,
            ignore_merges: model["ignore_merges"] == true,
            eos_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Find the earliest added token, then encode the plain text before it.
            let hit = (0..rest.len()).filter(|&i| rest.is_char_boundary(i)).find_map(|i| {
                self.added
                    .iter()
                    .find(|(s, _)| rest[i..].starts_with(s.as_str()))
                    .map(|a| (i, a))
            });
            match hit {
                Some((i, (s, id))) => {
                    self.encode_plain(&rest[..i], &mut out);
                    out.push(*id);
                    rest = &rest[i + s.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let mut pieces = vec![text];
        for step in &self.steps {
            let mut next = Vec::with_capacity(pieces.len());
            for p in pieces {
                match step {
                    PreStep::Digits => {
                        let mut start = 0;
                        for (i, c) in p.char_indices() {
                            if c.is_numeric() {
                                if start < i {
                                    next.push(&p[start..i]);
                                }
                                next.push(&p[i..i + c.len_utf8()]);
                                start = i + c.len_utf8();
                            }
                        }
                        if start < p.len() {
                            next.push(&p[start..]);
                        }
                    }
                    PreStep::Split(re) => {
                        let mut last = 0;
                        for m in re.find_iter(p) {
                            let m = m.expect("regex backtrack limit");
                            if m.start() > last {
                                next.push(&p[last..m.start()]);
                            }
                            next.push(m.as_str());
                            last = m.end();
                        }
                        if last < p.len() {
                            next.push(&p[last..]);
                        }
                    }
                }
            }
            pieces = next;
        }
        for p in pieces {
            self.bpe(p, out);
        }
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mapped: String = piece.bytes().map(|b| self.byte_to_char[b as usize]).collect();
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(&mapped) {
                out.push(id);
                return;
            }
        }
        let mut word: Vec<u32> = mapped
            .chars()
            .map(|c| self.vocab[c.encode_utf8(&mut [0; 4]) as &str])
            .collect();
        // ponytail: O(n^2) per word; words are short. Add a priority queue if long inputs get slow.
        loop {
            let best = word
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(rank, id)| (rank, i, id)))
                .min();
            let Some((_, i, id)) = best else { break };
            word[i] = id;
            word.remove(i + 1);
        }
        out.extend(word);
    }

    /// Raw bytes of one token. Byte-level tokens can split a UTF-8 character,
    /// so streaming callers join bytes before decoding.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let s = &self.tokens[id as usize];
        if self.is_special(id) {
            return s.as_bytes().to_vec();
        }
        s.chars()
            .map(|c| self.char_to_byte.get(&c).copied().unwrap_or(b'?'))
            .collect()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.token_bytes(id)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// SmolLM2's ChatML template, including its default system prompt.
    pub fn chat_prompt(messages: &[(&str, &str)]) -> String {
        let mut s = String::new();
        if messages.first().is_none_or(|m| m.0 != "system") {
            s.push_str(
                "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n",
            );
        }
        for (role, content) in messages {
            s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        s.push_str("<|im_start|>assistant\n");
        s
    }
}
