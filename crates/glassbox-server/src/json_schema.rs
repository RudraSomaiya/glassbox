//! Constrained decoding for `response_format: {type: "json_schema"}`.
//!
//! The schema compiles to a byte-level pushdown machine: a stack of frames
//! (fixed bytes, a string body, a number, a literal choice, array items, a
//! value still to start). Each step, the machine is run over a trie of every
//! token's bytes; a token is allowed when all of its bytes are accepted, so
//! the output always parses and validates.
//!
//! Supported: object (properties, required), array (items, minItems,
//! maxItems), string (maxLength), number, integer, boolean, null, enum, and
//! nesting. Simplifications, chosen so output is always valid and always ends:
//! objects emit every listed property in schema order (required or not, so
//! `required` is always satisfied), output is compact (no whitespace), strings
//! are capped at maxLength or 256 bytes, arrays at maxItems or 8 items,
//! numbers at 12 integer and 6 fraction digits.

use crate::scheduler::Constraint;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug)]
enum Node {
    /// `{"a":` value `,"b":` value `}`, as alternating fixed bytes and values.
    Obj(Vec<(Vec<u8>, Arc<Node>)>),
    Arr {
        item: Arc<Node>,
        min: usize,
        max: usize,
    },
    Str {
        max: usize,
    },
    Num {
        integer: bool,
    },
    /// One of a fixed set of JSON texts (enum, boolean, null).
    Lit(Arc<Vec<Vec<u8>>>),
}

fn compile(schema: &Value) -> Result<Arc<Node>, String> {
    if let Some(e) = schema.get("enum") {
        let opts = e.as_array().ok_or("enum must be an array")?;
        if opts.is_empty() || opts.len() > 64 {
            return Err("enum needs between 1 and 64 values".into());
        }
        return Ok(Arc::new(Node::Lit(Arc::new(
            opts.iter().map(|v| v.to_string().into_bytes()).collect(),
        ))));
    }
    let ty = schema
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or("every schema needs a type or an enum")?;
    Ok(Arc::new(match ty {
        "object" => {
            let props = schema.get("properties").and_then(|p| p.as_object());
            let mut fields = Vec::new();
            for (i, (name, sub)) in props.into_iter().flatten().enumerate() {
                let key = serde_json::to_string(name).unwrap();
                let lead = if i == 0 {
                    format!("{{{key}:")
                } else {
                    format!(",{key}:")
                };
                fields.push((lead.into_bytes(), compile(sub)?));
            }
            for r in schema.get("required").and_then(|r| r.as_array()).into_iter().flatten() {
                let r = r.as_str().ok_or("required names must be strings")?;
                if !props.is_some_and(|p| p.contains_key(r)) {
                    return Err(format!("required property {r} is not in properties"));
                }
            }
            Node::Obj(fields)
        }
        "array" => {
            let item = compile(schema.get("items").ok_or("array needs items")?)?;
            let min = schema.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let max = schema
                .get("maxItems")
                .and_then(|v| v.as_u64())
                .unwrap_or(8)
                .max(min as u64) as usize;
            Node::Arr { item, min, max }
        }
        "string" => Node::Str {
            max: schema.get("maxLength").and_then(|v| v.as_u64()).unwrap_or(256) as usize,
        },
        "number" => Node::Num { integer: false },
        "integer" => Node::Num { integer: true },
        "boolean" => Node::Lit(Arc::new(vec![b"true".to_vec(), b"false".to_vec()])),
        "null" => Node::Lit(Arc::new(vec![b"null".to_vec()])),
        t => return Err(format!("type {t} is not supported")),
    }))
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum NumPhase {
    Start,
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
}

#[derive(Clone, Debug)]
enum Frame {
    Value(Arc<Node>),
    Bytes(Arc<[u8]>, usize),
    /// Inside a string: bytes so far, cap, whether the last byte was a
    /// backslash, and continuation bytes still owed to a UTF-8 character
    /// (so the cap never cuts one in half).
    Str {
        len: usize,
        max: usize,
        escape: bool,
        cont: u8,
    },
    Num {
        integer: bool,
        phase: NumPhase,
        digits: usize,
    },
    Lit {
        opts: Arc<Vec<Vec<u8>>>,
        pos: usize,
        alive: u64,
    },
    /// Between array items, `count` done so far.
    Items {
        item: Arc<Node>,
        count: usize,
        min: usize,
        max: usize,
    },
}

#[derive(Clone, Debug)]
pub struct State(Vec<Frame>);

enum Step {
    Took,
    /// This frame is complete; offer the byte to the frame below.
    Done,
    No,
}

impl State {
    fn new(root: Arc<Node>) -> Self {
        State(vec![Frame::Value(root)])
    }

    pub fn finished(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether the document could end here (only when the stack is empty or
    /// the top is a number/literal that may stop, with nothing below it).
    fn can_end(&self) -> bool {
        match self.0.as_slice() {
            [] => true,
            [Frame::Num { phase, .. }] => matches!(phase, NumPhase::Zero | NumPhase::Int | NumPhase::Frac),
            [Frame::Lit { opts, pos, alive }] => lit_complete(opts, *pos, *alive),
            _ => false,
        }
    }

    fn advance(&mut self, b: u8) -> bool {
        loop {
            let Some(top) = self.0.last_mut() else { return false };
            let step = match top {
                Frame::Value(node) => {
                    let node = node.clone();
                    self.0.pop();
                    self.expand(&node);
                    continue;
                }
                Frame::Bytes(s, i) => {
                    if s[*i] != b {
                        Step::No
                    } else {
                        *i += 1;
                        if *i == s.len() {
                            self.0.pop();
                        }
                        return true;
                    }
                }
                Frame::Str { len, max, escape, cont } => {
                    if *cont > 0 {
                        if b & 0xC0 == 0x80 {
                            *cont -= 1;
                            *len += 1;
                            Step::Took
                        } else {
                            Step::No
                        }
                    } else if *escape {
                        if b"\"\\/bfnrt".contains(&b) {
                            *escape = false;
                            *len += 1;
                            Step::Took
                        } else {
                            Step::No
                        }
                    } else if b == b'"' {
                        self.0.pop();
                        return true;
                    } else if b == b'\\' && *len + 2 <= *max {
                        *escape = true;
                        *len += 1;
                        Step::Took
                    } else if b >= 0x80 {
                        let need = match b {
                            0xC2..=0xDF => 1,
                            0xE0..=0xEF => 2,
                            0xF0..=0xF4 => 3,
                            _ => return false,
                        };
                        if *len + (need as usize) < *max {
                            *cont = need;
                            *len += 1;
                            Step::Took
                        } else {
                            Step::No
                        }
                    } else if b >= 0x20 && b != b'\\' && *len < *max {
                        *len += 1;
                        Step::Took
                    } else {
                        Step::No
                    }
                }
                Frame::Num { integer, phase, digits } => {
                    let digit = b.is_ascii_digit();
                    let next = match (*phase, b) {
                        (NumPhase::Start, b'-') => Some((NumPhase::Minus, 0)),
                        (NumPhase::Start | NumPhase::Minus, b'0') => Some((NumPhase::Zero, 1)),
                        (NumPhase::Start | NumPhase::Minus, _) if digit => Some((NumPhase::Int, 1)),
                        (NumPhase::Int, _) if digit && *digits < 12 => Some((NumPhase::Int, *digits + 1)),
                        (NumPhase::Zero | NumPhase::Int, b'.') if !*integer => Some((NumPhase::Dot, 0)),
                        (NumPhase::Dot, _) if digit => Some((NumPhase::Frac, 1)),
                        (NumPhase::Frac, _) if digit && *digits < 6 => Some((NumPhase::Frac, *digits + 1)),
                        _ => None,
                    };
                    match next {
                        Some((p, d)) => {
                            *phase = p;
                            *digits = d;
                            Step::Took
                        }
                        None if matches!(*phase, NumPhase::Zero | NumPhase::Int | NumPhase::Frac) => Step::Done,
                        None => Step::No,
                    }
                }
                Frame::Lit { opts, pos, alive } => {
                    let mut next = 0u64;
                    for (i, o) in opts.iter().enumerate() {
                        if *alive & (1 << i) != 0 && o.get(*pos) == Some(&b) {
                            next |= 1 << i;
                        }
                    }
                    if next != 0 {
                        *alive = next;
                        *pos += 1;
                        // Done as soon as the only survivors are complete.
                        if opts
                            .iter()
                            .enumerate()
                            .all(|(i, o)| next & (1 << i) == 0 || o.len() == *pos)
                        {
                            self.0.pop();
                        }
                        return true;
                    } else if lit_complete(opts, *pos, *alive) {
                        Step::Done
                    } else {
                        Step::No
                    }
                }
                Frame::Items { item, count, min, max } => {
                    let (item, c, mn, mx) = (item.clone(), *count, *min, *max);
                    if b == b']' && c >= mn {
                        self.0.pop();
                        return true;
                    }
                    if c > 0 {
                        if b == b',' && c < mx {
                            *count += 1;
                            self.0.push(Frame::Value(item));
                            return true;
                        }
                        return false;
                    }
                    if mx == 0 {
                        return false;
                    }
                    *count = 1;
                    self.0.push(Frame::Value(item));
                    continue;
                }
            };
            match step {
                Step::Took => return true,
                Step::No => return false,
                Step::Done => {
                    self.0.pop();
                }
            }
        }
    }

    fn expand(&mut self, node: &Arc<Node>) {
        match &**node {
            Node::Obj(fields) => {
                self.0.push(Frame::Bytes(Arc::from(&b"}"[..]), 0));
                for (lead, value) in fields.iter().rev() {
                    self.0.push(Frame::Value(value.clone()));
                    self.0.push(Frame::Bytes(Arc::from(&lead[..]), 0));
                }
                if fields.is_empty() {
                    self.0.push(Frame::Bytes(Arc::from(&b"{"[..]), 0));
                }
            }
            Node::Arr { item, min, max } => {
                self.0.push(Frame::Items {
                    item: item.clone(),
                    count: 0,
                    min: *min,
                    max: *max,
                });
                self.0.push(Frame::Bytes(Arc::from(&b"["[..]), 0));
            }
            Node::Str { max } => {
                self.0.push(Frame::Str {
                    len: 0,
                    max: *max,
                    escape: false,
                    cont: 0,
                });
                self.0.push(Frame::Bytes(Arc::from(&b"\""[..]), 0));
            }
            Node::Num { integer } => self.0.push(Frame::Num {
                integer: *integer,
                phase: NumPhase::Start,
                digits: 0,
            }),
            Node::Lit(opts) => {
                let alive = if opts.len() >= 64 {
                    u64::MAX
                } else {
                    (1u64 << opts.len()) - 1
                };
                self.0.push(Frame::Lit {
                    opts: opts.clone(),
                    pos: 0,
                    alive,
                })
            }
        }
    }

    /// In a string body with room to spare: any "plain" token is fine.
    fn plain_string(&self, room: usize) -> bool {
        matches!(self.0.last(), Some(Frame::Str { len, max, escape: false, cont: 0 }) if len + room <= *max)
    }
}

fn lit_complete(opts: &[Vec<u8>], pos: usize, alive: u64) -> bool {
    opts.iter()
        .enumerate()
        .any(|(i, o)| alive & (1 << i) != 0 && o.len() == pos)
}

/// Every token's bytes as a trie, built once per vocabulary.
pub struct Trie {
    /// Per node: children (byte, node) and tokens ending here.
    children: Vec<Vec<(u8, u32)>>,
    ends: Vec<Vec<u32>>,
    /// Tokens made only of bytes that can sit inside a JSON string unescaped.
    plain: Vec<u32>,
    longest_plain: usize,
    /// Tokens containing a quote, backslash or control byte (checked one by one in strings).
    other: Vec<u32>,
    vocab: Arc<Vec<Vec<u8>>>,
    eos: u32,
}

impl Trie {
    pub fn new(vocab: Arc<Vec<Vec<u8>>>, special: &[bool], eos: u32) -> Self {
        let mut children: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
        let mut ends: Vec<Vec<u32>> = vec![Vec::new()];
        let (mut plain, mut other, mut longest_plain) = (Vec::new(), Vec::new(), 0);
        for (id, bytes) in vocab.iter().enumerate() {
            if bytes.is_empty() || special.get(id).copied().unwrap_or(false) {
                continue;
            }
            let mut n = 0usize;
            for &b in bytes {
                n = match children[n].iter().find(|c| c.0 == b) {
                    Some(&(_, c)) => c as usize,
                    None => {
                        children.push(Vec::new());
                        ends.push(Vec::new());
                        let c = children.len() - 1;
                        children[n].push((b, c as u32));
                        c
                    }
                };
            }
            ends[n].push(id as u32);
            // Plain: complete UTF-8 with nothing that needs escaping. Tokens
            // that split a character are checked byte by byte instead.
            if std::str::from_utf8(bytes).is_ok() && bytes.iter().all(|&b| b >= 0x20 && b != b'"' && b != b'\\') {
                plain.push(id as u32);
                longest_plain = longest_plain.max(bytes.len());
            } else {
                other.push(id as u32);
            }
        }
        Trie {
            children,
            ends,
            plain,
            longest_plain,
            other,
            vocab,
            eos,
        }
    }

    /// Tokens that keep the output a valid prefix of a schema-valid document.
    pub fn allowed(&self, state: &State, out: &mut Vec<u32>) {
        out.clear();
        if state.can_end() {
            out.push(self.eos);
        }
        if state.finished() {
            return;
        }
        if state.plain_string(self.longest_plain) {
            out.extend(&self.plain);
            for &t in &self.other {
                let mut s = state.clone();
                if self.vocab[t as usize].iter().all(|&b| s.advance(b)) {
                    out.push(t);
                }
            }
            return;
        }
        // Depth-first over the trie, carrying the machine state down each edge.
        let mut stack = vec![(0u32, state.clone())];
        while let Some((n, s)) = stack.pop() {
            out.extend(&self.ends[n as usize]);
            for &(b, c) in &self.children[n as usize] {
                let mut s2 = s.clone();
                if s2.advance(b) {
                    stack.push((c, s2));
                }
            }
        }
    }
}

pub struct JsonConstraint {
    trie: Arc<Trie>,
    state: State,
    allowed: Vec<u32>,
}

impl JsonConstraint {
    pub fn new(schema: &Value, trie: Arc<Trie>) -> Result<Self, String> {
        Ok(JsonConstraint {
            state: State::new(compile(schema)?),
            trie,
            allowed: Vec::new(),
        })
    }
}

impl Constraint for JsonConstraint {
    fn mask(&mut self, logits: &mut [f32]) {
        self.trie.allowed(&self.state, &mut self.allowed);
        let mut keep = vec![false; logits.len()];
        for &t in &self.allowed {
            keep[t as usize] = true;
        }
        for (l, k) in logits.iter_mut().zip(keep) {
            if !k {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    fn accept(&mut self, token: u32) {
        if token == self.trie.eos {
            self.state.0.clear();
            return;
        }
        for &b in &self.trie.vocab[token as usize] {
            let ok = self.state.advance(b);
            debug_assert!(ok, "masked token was sampled");
        }
    }

    fn done(&self) -> bool {
        self.state.finished()
    }
}

/// Minimal validator for the supported subset (used by tests and the example).
pub fn validate(schema: &Value, v: &Value) -> bool {
    if let Some(e) = schema.get("enum").and_then(|e| e.as_array()) {
        return e.contains(v);
    }
    match schema.get("type").and_then(|t| t.as_str()) {
        Some("object") => {
            let Some(o) = v.as_object() else { return false };
            let props = schema.get("properties").and_then(|p| p.as_object());
            let required_ok = schema
                .get("required")
                .and_then(|r| r.as_array())
                .into_iter()
                .flatten()
                .all(|r| o.contains_key(r.as_str().unwrap_or("")));
            required_ok
                && o.iter()
                    .all(|(k, val)| props.and_then(|p| p.get(k)).is_some_and(|s| validate(s, val)))
        }
        Some("array") => {
            let Some(a) = v.as_array() else { return false };
            let min = schema.get("minItems").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let max = schema.get("maxItems").and_then(|x| x.as_u64()).unwrap_or(u64::MAX) as usize;
            a.len() >= min && a.len() <= max && a.iter().all(|x| validate(&schema["items"], x))
        }
        Some("string") => v.as_str().is_some_and(|s| {
            schema
                .get("maxLength")
                .and_then(|m| m.as_u64())
                .is_none_or(|m| s.chars().count() as u64 <= m)
        }),
        Some("number") => v.is_number(),
        // JSON Schema: an integer is any number with no fractional part (-0 included).
        Some("integer") => v.as_f64().is_some_and(|f| f.fract() == 0.0),
        Some("boolean") => v.is_boolean(),
        Some("null") => v.is_null(),
        _ => false,
    }
}
