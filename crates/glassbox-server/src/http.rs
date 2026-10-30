//! OpenAI-compatible HTTP API. Handlers tokenize and stream; one engine thread
//! owns the model and runs the scheduler.

use crate::scheduler::{Config, Constraint, Event, Finish, Metrics, Request, Scheduler, StepRecord};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use glassbox_core::{Model, SamplerConfig, Tokenizer};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc as tmpsc, oneshot};
use tokio_stream::StreamExt;

pub enum Cmd {
    Submit(Request, oneshot::Sender<usize>),
    StartTrace,
    TakeTrace(oneshot::Sender<Vec<StepRecord>>),
}

#[derive(Default, Clone)]
pub struct Snapshot {
    pub metrics: Metrics,
    pub queue_depth: usize,
    pub running: usize,
    pub kv_utilization: f64,
}

#[derive(Clone)]
pub struct AppState {
    pub tx: mpsc::Sender<Cmd>,
    pub tok: Arc<Tokenizer>,
    pub model_name: String,
    pub snapshot: Arc<Mutex<Snapshot>>,
    pub vocab: Arc<Vec<Vec<u8>>>,
}

/// The engine thread: waits for work when idle, otherwise steps continuously.
pub fn spawn_engine(model: &'static Model, config: Config, snapshot: Arc<Mutex<Snapshot>>) -> mpsc::Sender<Cmd> {
    let (tx, rx) = mpsc::channel::<Cmd>();
    std::thread::Builder::new()
        .name("glassbox-engine".into())
        .spawn(move || {
            let mut s = Scheduler::new(model, config);
            let handle = |s: &mut Scheduler, c: Cmd| match c {
                Cmd::Submit(r, id) => {
                    let _ = id.send(s.submit(r));
                }
                Cmd::StartTrace => s.trace = Some(Vec::new()),
                Cmd::TakeTrace(out) => {
                    let _ = out.send(s.trace.take().unwrap_or_default());
                }
            };
            loop {
                if s.is_idle() {
                    match rx.recv() {
                        Ok(c) => handle(&mut s, c),
                        Err(_) => return,
                    }
                }
                while let Ok(c) = rx.try_recv() {
                    handle(&mut s, c);
                }
                s.step();
                // ponytail: keeps every TTFT/ITL sample; a long-running server wants a histogram.
                let mut snap = snapshot.lock().unwrap();
                snap.metrics = s.metrics.clone();
                snap.queue_depth = s.queue_depth();
                snap.running = s.running();
                snap.kv_utilization = s.kv_utilization();
            }
        })
        .expect("spawn engine thread");
    tx
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .route("/v1/completions", post(completions))
        .route("/v1/trace", post(trace))
        .route("/metrics", get(metrics))
        .with_state(state)
}

#[derive(Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
pub struct GenParams {
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub response_format: Option<Value>,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    #[serde(flatten)]
    pub params: GenParams,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(flatten)]
    pub params: GenParams,
}

fn error(code: StatusCode, msg: impl Into<String>) -> Response {
    (
        code,
        Json(json!({"error": {"message": msg.into(), "type": "invalid_request_error"}})),
    )
        .into_response()
}

async fn models(State(st): State<AppState>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": st.model_name, "object": "model", "owned_by": "glassbox"}]}))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Turns token ids into text across token boundaries that split a UTF-8 character.
struct Detok {
    pending: Vec<u8>,
}

impl Detok {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_string();
                self.pending.clear();
                s
            }
            Err(e) if e.error_len().is_none() => {
                let ok = e.valid_up_to();
                let s = String::from_utf8_lossy(&self.pending[..ok]).into_owned();
                self.pending.drain(..ok);
                s
            }
            Err(_) => {
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                s
            }
        }
    }
}

type ConstraintFactory = fn(&Value, &Tokenizer) -> Result<Box<dyn Constraint>, String>;

/// Set by the constrained-decoding module; `None` means `response_format` is rejected.
pub static JSON_SCHEMA: std::sync::OnceLock<ConstraintFactory> = std::sync::OnceLock::new();

async fn generate(st: AppState, prompt: Vec<u32>, p: GenParams, chat: bool) -> Response {
    let constraint = match p.response_format.as_ref().filter(|f| f["type"] == "json_schema") {
        None => None,
        Some(f) => match JSON_SCHEMA.get() {
            None => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "response_format json_schema is not supported by this build",
                )
            }
            Some(make) => match make(&f["json_schema"]["schema"], &st.tok) {
                Ok(c) => Some(c),
                Err(e) => return error(StatusCode::BAD_REQUEST, format!("unsupported schema: {e}")),
            },
        },
    };
    let (sink, mut rx) = tmpsc::unbounded_channel();
    let sampler = SamplerConfig {
        temperature: p.temperature.unwrap_or(0.7),
        top_p: p.top_p.unwrap_or(1.0),
        top_k: p.top_k.unwrap_or(0),
        min_p: p.min_p.unwrap_or(0.0),
        repetition_penalty: 1.0,
        seed: p.seed.unwrap_or_else(|| now() ^ 0x5eed),
    };
    let req = Request {
        prompt,
        max_tokens: p.max_tokens.unwrap_or(256),
        sampler,
        stop: vec![st.tok.eos_id],
        sink,
        constraint,
    };
    let (id_tx, id_rx) = oneshot::channel();
    if st.tx.send(Cmd::Submit(req, id_tx)).is_err() {
        return error(StatusCode::SERVICE_UNAVAILABLE, "engine is not running");
    }
    let id = format!("glassbox-{}", id_rx.await.unwrap_or(0));
    let model = st.model_name.clone();
    let created = now();
    let object = if chat { "chat.completion" } else { "text_completion" };

    if p.stream {
        let vocab = st.vocab.clone();
        let mut detok = Detok { pending: Vec::new() };
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(move |ev| {
            let (delta, finish) = match ev {
                Event::Token(t) => (detok.push(&vocab[t as usize]), Value::Null),
                Event::Done { reason, .. } => (String::new(), json!(reason_str(reason))),
            };
            let choice = if chat {
                json!({"index": 0, "delta": {"content": delta}, "finish_reason": finish})
            } else {
                json!({"index": 0, "text": delta, "finish_reason": finish})
            };
            let chunk = json!({"id": id, "object": format!("{object}.chunk"), "created": created, "model": model,
                               "choices": [choice]});
            Ok::<_, std::convert::Infallible>(SseEvent::default().data(chunk.to_string()))
        });
        let done = tokio_stream::once(Ok(SseEvent::default().data("[DONE]")));
        return Sse::new(stream.chain(done)).into_response();
    }

    let mut ids = Vec::new();
    let (mut reason, mut usage) = (Finish::Stop, (0, 0));
    while let Some(ev) = rx.recv().await {
        match ev {
            Event::Token(t) => ids.push(t),
            Event::Done {
                reason: r,
                prompt_tokens,
                completion_tokens,
            } => {
                reason = r;
                usage = (prompt_tokens, completion_tokens);
            }
        }
    }
    let text = st.tok.decode(&ids);
    let choice = if chat {
        json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": reason_str(reason)})
    } else {
        json!({"index": 0, "text": text, "finish_reason": reason_str(reason)})
    };
    Json(json!({
        "id": id, "object": object, "created": created, "model": model, "choices": [choice],
        "usage": {"prompt_tokens": usage.0, "completion_tokens": usage.1, "total_tokens": usage.0 + usage.1},
    }))
    .into_response()
}

fn reason_str(r: Finish) -> &'static str {
    match r {
        Finish::Stop => "stop",
        Finish::Length => "length",
        Finish::Aborted => "abort",
    }
}

async fn chat(State(st): State<AppState>, Json(req): Json<ChatRequest>) -> Response {
    if req.messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }
    let msgs: Vec<(&str, &str)> = req
        .messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();
    let prompt = st.tok.encode(&Tokenizer::chat_prompt(&msgs));
    generate(st, prompt, req.params, true).await
}

async fn completions(State(st): State<AppState>, Json(req): Json<CompletionRequest>) -> Response {
    let prompt = st.tok.encode(&req.prompt);
    if prompt.is_empty() {
        return error(StatusCode::BAD_REQUEST, "prompt must not be empty");
    }
    generate(st, prompt, req.params, false).await
}

#[derive(Deserialize)]
struct TraceRequest {
    requests: Vec<TraceItem>,
}

#[derive(Deserialize)]
struct TraceItem {
    messages: Vec<Message>,
    #[serde(default)]
    delay_ms: u64,
    #[serde(default)]
    max_tokens: Option<usize>,
}

/// Run a set of chat requests (each after its delay) with step recording on,
/// and return what the scheduler did plus every request's tokens.
async fn trace(State(st): State<AppState>, Json(req): Json<TraceRequest>) -> Response {
    let _ = st.tx.send(Cmd::StartTrace);
    let mut tasks = Vec::new();
    for item in req.requests {
        let st = st.clone();
        tasks.push(tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(item.delay_ms)).await;
            let msgs: Vec<(&str, &str)> = item
                .messages
                .iter()
                .map(|m| (m.role.as_str(), m.content.as_str()))
                .collect();
            let prompt = st.tok.encode(&Tokenizer::chat_prompt(&msgs));
            let (sink, mut rx) = tmpsc::unbounded_channel();
            let n_prompt = prompt.len();
            let r = Request {
                prompt,
                max_tokens: item.max_tokens.unwrap_or(48),
                sampler: SamplerConfig::default(),
                stop: vec![st.tok.eos_id],
                sink,
                constraint: None,
            };
            let (id_tx, id_rx) = oneshot::channel();
            let arrival = crate::scheduler::unix_ms();
            let _ = st.tx.send(Cmd::Submit(r, id_tx));
            let id = id_rx.await.unwrap_or(0);
            let mut tokens = Vec::new();
            while let Some(Event::Token(t)) = rx.recv().await {
                tokens.push(json!({"id": t, "text": String::from_utf8_lossy(&st.vocab[t as usize]),
                                   "t_ms": crate::scheduler::unix_ms()}));
            }
            json!({"id": id, "arrival_ms": arrival, "prompt_tokens": n_prompt, "tokens": tokens})
        }));
    }
    let mut requests = Vec::new();
    for t in tasks {
        requests.push(t.await.unwrap_or(Value::Null));
    }
    let (out_tx, out_rx) = oneshot::channel();
    let _ = st.tx.send(Cmd::TakeTrace(out_tx));
    let steps = out_rx.await.unwrap_or_default();
    Json(json!({"model": st.model_name, "requests": requests, "steps": steps})).into_response()
}

fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[((s.len() - 1) as f64 * q).round() as usize]
}

async fn metrics(State(st): State<AppState>) -> impl IntoResponse {
    let s = st.snapshot.lock().unwrap().clone();
    let m = &s.metrics;
    let mut out = String::new();
    let mut line = |name: &str, kind: &str, help: &str, v: f64| {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name} {v}\n"));
    };
    line(
        "glassbox_requests_finished_total",
        "counter",
        "Requests completed.",
        m.requests_finished as f64,
    );
    line(
        "glassbox_prompt_tokens_total",
        "counter",
        "Prompt tokens received.",
        m.prompt_tokens as f64,
    );
    line(
        "glassbox_generated_tokens_total",
        "counter",
        "Tokens generated.",
        m.generated_tokens as f64,
    );
    line(
        "glassbox_prefix_hit_tokens_total",
        "counter",
        "Prompt tokens served from the prefix cache.",
        m.prefix_hit_tokens as f64,
    );
    line(
        "glassbox_preemptions_total",
        "counter",
        "Sequences preempted for lack of KV blocks.",
        m.preemptions as f64,
    );
    line("glassbox_steps_total", "counter", "Forward passes run.", m.steps as f64);
    line(
        "glassbox_batch_tokens_total",
        "counter",
        "Tokens processed across all forward passes.",
        m.batch_tokens as f64,
    );
    line(
        "glassbox_queue_depth",
        "gauge",
        "Requests waiting for admission.",
        s.queue_depth as f64,
    );
    line(
        "glassbox_running",
        "gauge",
        "Sequences in the running batch.",
        s.running as f64,
    );
    line(
        "glassbox_kv_utilization",
        "gauge",
        "Fraction of KV cache blocks in use.",
        s.kv_utilization,
    );
    for (name, help, v) in [
        ("glassbox_ttft_seconds", "Time to first token.", &m.ttft),
        ("glassbox_itl_seconds", "Time between tokens of one request.", &m.itl),
    ] {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} summary\n"));
        for q in [0.5, 0.9, 0.99] {
            out.push_str(&format!("{name}{{quantile=\"{q}\"}} {}\n", quantile(v, q)));
        }
        out.push_str(&format!(
            "{name}_sum {}\n{name}_count {}\n",
            v.iter().sum::<f64>(),
            v.len()
        ));
    }
    ([("content-type", "text/plain; version=0.0.4")], out)
}
