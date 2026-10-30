//! Weight loading: Hugging Face safetensors (f32/f16/bf16) and GGUF (F32, F16, Q8_0, Q4_0).

use crate::kernels::{Matrix, Weights};
use crate::model::{Config, Layer, Model};
use crate::quant::{BlockQ4_0, BlockQ8_0, QK};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

fn to_f32(dtype: Dtype, data: &[u8]) -> Result<Vec<f32>, Error> {
    Ok(match dtype {
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect(),
        other => return Err(format!("unsupported dtype {other:?}").into()),
    })
}

impl Model {
    /// Build a model from a `config.json` string and safetensors bytes.
    pub fn from_safetensors(config: &str, bytes: &[u8]) -> Result<Model, Error> {
        let config = Config::from_json(config)?;
        let st = SafeTensors::deserialize(bytes)?;
        let get = |name: &str| -> Result<(Vec<usize>, Vec<f32>), Error> {
            let t = st.tensor(name).map_err(|e| format!("{name}: {e}"))?;
            Ok((t.shape().to_vec(), to_f32(t.dtype(), t.data())?))
        };
        let mat = |name: &str| -> Result<Matrix, Error> {
            let (shape, w) = get(name)?;
            Ok(Matrix::f32(shape[0], shape[1], w))
        };
        let vec = |name: &str| -> Result<Vec<f32>, Error> { Ok(get(name)?.1) };
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                Ok(Layer {
                    attn_norm: vec(&format!("{p}.input_layernorm.weight"))?,
                    wq: mat(&format!("{p}.self_attn.q_proj.weight"))?,
                    wk: mat(&format!("{p}.self_attn.k_proj.weight"))?,
                    wv: mat(&format!("{p}.self_attn.v_proj.weight"))?,
                    wo: mat(&format!("{p}.self_attn.o_proj.weight"))?,
                    mlp_norm: vec(&format!("{p}.post_attention_layernorm.weight"))?,
                    w_gate: mat(&format!("{p}.mlp.gate_proj.weight"))?,
                    w_up: mat(&format!("{p}.mlp.up_proj.weight"))?,
                    w_down: mat(&format!("{p}.mlp.down_proj.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let has_head = st.names().iter().any(|n| *n == "lm_head.weight");
        let lm_head = if has_head && !config.tie_word_embeddings {
            Some(mat("lm_head.weight")?)
        } else {
            None
        };
        Ok(Model {
            embed: mat("model.embed_tokens.weight")?,
            norm: vec("model.norm.weight")?,
            layers,
            lm_head,
            config,
            rope_interleaved: false,
        })
    }

    /// Load `config.json` + `model.safetensors` from a model directory.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_dir(dir: impl AsRef<std::path::Path>) -> Result<Model, Error> {
        let dir = dir.as_ref();
        let config = std::fs::read_to_string(dir.join("config.json"))?;
        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        // SAFETY: the file is opened read-only and not modified while mapped;
        // the map is dropped before this function returns.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Model::from_safetensors(&config, &map)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_gguf(path: impl AsRef<std::path::Path>) -> Result<Model, Error> {
        let file = std::fs::File::open(path)?;
        // SAFETY: as in `load_dir`, read-only and not modified while mapped.
        let map = unsafe { memmap2::Mmap::map(&file)? };
        Model::from_gguf(&map)
    }

    pub fn from_gguf(bytes: &[u8]) -> Result<Model, Error> {
        let g = Gguf::parse(bytes)?;
        let arch = g.str("general.architecture").unwrap_or("llama").to_string();
        let u = |k: &str| {
            g.num(&format!("{arch}.{k}"))
                .ok_or_else(|| format!("missing {arch}.{k}"))
        };
        let embed = g.matrix("token_embd.weight")?;
        let config = Config {
            hidden_size: u("embedding_length")? as usize,
            intermediate_size: u("feed_forward_length")? as usize,
            num_hidden_layers: u("block_count")? as usize,
            num_attention_heads: u("attention.head_count")? as usize,
            num_key_value_heads: u("attention.head_count_kv")? as usize,
            vocab_size: embed.rows,
            rms_norm_eps: u("attention.layer_norm_rms_epsilon")? as f32,
            rope_theta: g.num(&format!("{arch}.rope.freq_base")).unwrap_or(10000.0) as f32,
            tie_word_embeddings: !g.tensors.contains_key("output.weight"),
            head_dim: None,
            max_position_embeddings: g.num(&format!("{arch}.context_length")).unwrap_or(2048.0) as usize,
        };
        let layers = (0..config.num_hidden_layers)
            .map(|i| {
                let m = |n: &str| g.matrix(&format!("blk.{i}.{n}.weight"));
                let v = |n: &str| g.vector(&format!("blk.{i}.{n}.weight"));
                Ok(Layer {
                    attn_norm: v("attn_norm")?,
                    wq: m("attn_q")?,
                    wk: m("attn_k")?,
                    wv: m("attn_v")?,
                    wo: m("attn_output")?,
                    mlp_norm: v("ffn_norm")?,
                    w_gate: m("ffn_gate")?,
                    w_up: m("ffn_up")?,
                    w_down: m("ffn_down")?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(g.matrix("output.weight")?)
        };
        Ok(Model {
            norm: g.vector("output_norm.weight")?,
            embed,
            layers,
            lm_head,
            config,
            rope_interleaved: true,
        })
    }
}

enum Meta {
    Num(f64),
    Str(String),
    Other,
}

struct Gguf<'a> {
    meta: HashMap<String, Meta>,
    /// name -> (dims, ggml type, absolute byte offset)
    tensors: HashMap<String, (Vec<usize>, u32, usize)>,
    bytes: &'a [u8],
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let s = self.b.get(self.at..self.at + n).ok_or("gguf: truncated")?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64, Error> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String, Error> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Meta, Error> {
        let b = |r: &mut Self, n| -> Result<[u8; 8], Error> {
            let mut a = [0u8; 8];
            a[..n].copy_from_slice(r.take(n)?);
            Ok(a)
        };
        Ok(match ty {
            0 | 7 => Meta::Num(self.take(1)?[0] as f64),
            1 => Meta::Num(self.take(1)?[0] as i8 as f64),
            2 => Meta::Num(u16::from_le_bytes(self.take(2)?.try_into()?) as f64),
            3 => Meta::Num(i16::from_le_bytes(self.take(2)?.try_into()?) as f64),
            4 => Meta::Num(self.u32()? as f64),
            5 => Meta::Num(self.u32()? as i32 as f64),
            6 => Meta::Num(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            8 => Meta::Str(self.string()?),
            9 => {
                let (ty, n) = (self.u32()?, self.u64()?);
                for _ in 0..n {
                    self.value(ty)?;
                }
                Meta::Other
            }
            10 => Meta::Num(u64::from_le_bytes(b(self, 8)?) as f64),
            11 => Meta::Num(i64::from_le_bytes(b(self, 8)?) as f64),
            12 => Meta::Num(f64::from_le_bytes(b(self, 8)?)),
            _ => return Err(format!("gguf: unknown value type {ty}").into()),
        })
    }
}

const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_Q4_0: u32 = 2;
const GGML_Q8_0: u32 = 8;

impl<'a> Gguf<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let mut r = Reader { b: bytes, at: 0 };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let version = r.u32()?;
        if version < 2 {
            return Err(format!("GGUF v{version} not supported").into());
        }
        let (nt, nk) = (r.u64()?, r.u64()?);
        let mut meta = HashMap::new();
        for _ in 0..nk {
            let k = r.string()?;
            let ty = r.u32()?;
            meta.insert(k, r.value(ty)?);
        }
        let mut infos = Vec::new();
        for _ in 0..nt {
            let name = r.string()?;
            let nd = r.u32()?;
            let dims = (0..nd)
                .map(|_| r.u64().map(|d| d as usize))
                .collect::<Result<Vec<_>, _>>()?;
            let (ty, off) = (r.u32()?, r.u64()? as usize);
            infos.push((name, dims, ty, off));
        }
        let align = match meta.get("general.alignment") {
            Some(Meta::Num(a)) => *a as usize,
            _ => 32,
        };
        let data = r.at.div_ceil(align) * align;
        let tensors = infos.into_iter().map(|(n, d, t, o)| (n, (d, t, data + o))).collect();
        Ok(Gguf { meta, tensors, bytes })
    }

    fn num(&self, k: &str) -> Option<f64> {
        match self.meta.get(k) {
            Some(Meta::Num(v)) => Some(*v),
            _ => None,
        }
    }

    fn str(&self, k: &str) -> Option<&str> {
        match self.meta.get(k) {
            Some(Meta::Str(s)) => Some(s),
            _ => None,
        }
    }

    fn matrix(&self, name: &str) -> Result<Matrix, Error> {
        let (dims, ty, off) = self.tensors.get(name).ok_or_else(|| format!("gguf: missing {name}"))?;
        let cols = dims[0];
        let rows = dims[1..].iter().product::<usize>();
        let n = rows * cols;
        let w = match *ty {
            GGML_F32 | GGML_F16 => {
                let (dt, size) = if *ty == GGML_F32 {
                    (Dtype::F32, 4)
                } else {
                    (Dtype::F16, 2)
                };
                Weights::F32(to_f32(dt, self.slice(*off, n * size)?)?)
            }
            GGML_Q8_0 => Weights::Q8_0(read_blocks(
                self.slice(*off, n / QK * 34)?,
                |b| {
                    let mut blk = BlockQ8_0 {
                        d: u16::from_le_bytes([b[0], b[1]]),
                        qs: [0; QK],
                    };
                    for (q, &v) in blk.qs.iter_mut().zip(&b[2..]) {
                        *q = v as i8;
                    }
                    blk
                },
                34,
            )),
            GGML_Q4_0 => Weights::Q4_0(read_blocks(
                self.slice(*off, n / QK * 18)?,
                |b| {
                    let mut blk = BlockQ4_0 {
                        d: u16::from_le_bytes([b[0], b[1]]),
                        qs: [0; QK / 2],
                    };
                    blk.qs.copy_from_slice(&b[2..18]);
                    blk
                },
                18,
            )),
            t => return Err(format!("gguf: {name} has unsupported type {t}").into()),
        };
        Ok(Matrix { rows, cols, w })
    }

    fn vector(&self, name: &str) -> Result<Vec<f32>, Error> {
        let m = self.matrix(name)?;
        let mut out = vec![0.0; m.rows * m.cols];
        for r in 0..m.rows {
            m.row(r, &mut out[r * m.cols..(r + 1) * m.cols]);
        }
        Ok(out)
    }

    fn slice(&self, off: usize, len: usize) -> Result<&'a [u8], Error> {
        Ok(self.bytes.get(off..off + len).ok_or("gguf: tensor out of bounds")?)
    }
}

fn read_blocks<T>(b: &[u8], f: impl Fn(&[u8]) -> T, size: usize) -> Vec<T> {
    b.chunks_exact(size).map(f).collect()
}
