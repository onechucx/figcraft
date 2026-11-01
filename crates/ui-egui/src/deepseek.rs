//! The model side of the AI chat: one chat-completions call per model turn, and the worker thread
//! that runs a conversation. The models are DeepSeek's, from its API and from this computer through
//! Ollama, and Antigravity, which runs on this computer through Ollama. The worker never touches the
//! document. When the model asks for tools it hands the calls to the UI thread, which owns the
//! document, and waits for their results.
//! Native builds only: the browser build has no sockets.

use std::sync::mpsc::Receiver;
use std::time::Duration;

use serde_json::{Value, json};

use crate::web_page;

/// The chat-completions endpoint of the DeepSeek API.
pub const ENDPOINT: &str = "https://api.deepseek.com/chat/completions";
/// The chat-completions endpoint of Ollama on this computer, which serves the local model.
pub const LOCAL_ENDPOINT: &str = "http://127.0.0.1:11434/v1/chat/completions";
/// The model that reads the images sent with a message.
pub const VISION_MODEL: &str = "deepseek-flash";
/// Tool rounds one message may run before the turn is stopped (a guard against loops).
pub const MAX_TOOL_ROUNDS: usize = 16;
/// How long one call to the API may take.
const TIMEOUT: Duration = Duration::from_secs(180);
/// How long one call to the local model may take. A 7-billion-parameter model on a laptop CPU can
/// take minutes for one reply.
const LOCAL_TIMEOUT: Duration = Duration::from_secs(600);
/// Characters of an error reply kept for the message shown to the user.
const ERROR_CHARS: usize = 300;
/// What a tool round answers with when its results do not match its calls.
const MISMATCH: &str = "The tool results did not match the tool calls";

/// One model the panel offers: its provider, its name there, and where and how it runs.
pub struct Choice {
    /// The provider the model belongs to. The panel's first switch picks a provider, and its second
    /// lists that provider's models.
    pub route: &'static str,
    /// The name the panel shows.
    pub label: &'static str,
    /// The chat-completions endpoint the model is called at.
    pub endpoint: &'static str,
    /// The model name the endpoint takes.
    pub model: &'static str,
    /// Whether the model is called with the DeepSeek API key. Only the API takes one.
    pub needs_key: bool,
    /// Whether the model reads images.
    pub reads_images: bool,
    /// Whether the model has no native tool calls. Its tools are then described in the system
    /// prompt, and it asks for one by replying with a JSON object.
    pub text_tools: bool,
    /// How long one call may take.
    pub timeout: Duration,
}

/// The models the panel offers, in the order it shows them. A provider's models are listed together.
pub static CHOICES: [Choice; 4] = [
    Choice {
        route: "DeepSeek",
        label: "deepseek-v4-pro",
        endpoint: ENDPOINT,
        model: "deepseek-v4-pro",
        needs_key: true,
        reads_images: false,
        text_tools: false,
        timeout: TIMEOUT,
    },
    Choice {
        route: "DeepSeek",
        label: "deepseek-flash",
        endpoint: ENDPOINT,
        model: VISION_MODEL,
        needs_key: true,
        reads_images: true,
        text_tools: false,
        timeout: TIMEOUT,
    },
    Choice {
        route: "DeepSeek",
        label: "hermes (local)",
        endpoint: LOCAL_ENDPOINT,
        model: "deepseek-hermes:latest",
        needs_key: false,
        reads_images: false,
        text_tools: true,
        timeout: LOCAL_TIMEOUT,
    },
    Choice {
        route: "Antigravity",
        label: "antigravity",
        endpoint: LOCAL_ENDPOINT,
        model: "antigravity:latest",
        needs_key: false,
        reads_images: false,
        text_tools: true,
        timeout: LOCAL_TIMEOUT,
    },
];

/// The providers the panel offers, in the order the choices give them, each named once.
pub fn providers() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    for choice in CHOICES.iter() {
        if !names.contains(&choice.route) {
            names.push(choice.route);
        }
    }
    names
}

/// The index in [`CHOICES`] of the first model of a provider, if it has one.
pub fn first_model(provider: &str) -> Option<usize> {
    CHOICES.iter().position(|choice| choice.route == provider)
}

/// What one turn needs: where to call, with which key, model, system prompt and tools, and whether
/// the tools go through the reply text. The key is never printed or stored in a message.
#[derive(Clone)]
pub struct Setup {
    /// The provider's name, for the messages that report a failed call.
    pub provider: &'static str,
    pub endpoint: String,
    /// Empty when the endpoint takes no key.
    pub key: String,
    pub model: String,
    pub system: String,
    pub tools: Vec<Value>,
    /// Whether the tools go through the reply text (see [`Choice::text_tools`]).
    pub text_tools: bool,
    pub timeout: Duration,
}

/// One tool call the model made: the id its result must answer, the tool's name and its
/// arguments as JSON text.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// What the UI thread sends the worker.
pub enum Job {
    /// A message from the user, with the images it carries (as `data:` URLs) and the setup to
    /// answer it.
    User { text: String, images: Vec<String>, setup: Setup },
    /// The results of the tool calls the worker asked for, as (call id, result text).
    ToolResults(Vec<(String, String)>),
}

/// What the worker reports to the UI thread.
#[derive(Debug, PartialEq)]
pub enum Event {
    /// The model wants these tools run. Answer with [`Job::ToolResults`].
    Tools(Vec<ToolCall>),
    /// A page the model asked for is being read. The UI shows its address.
    Reading(String),
    /// The turn is over: the model's reply, or why it failed.
    Done(Result<String, String>),
}

/// Runs conversations until the job channel closes. `report` gets every event (the UI thread
/// wakes for them).
pub fn run(jobs: Receiver<Job>, mut report: impl FnMut(Event)) {
    let mut history: Vec<Value> = Vec::new();
    while let Ok(job) = jobs.recv() {
        // Tool results with no turn waiting for them are stale.
        let Job::User { text, images, setup } = job else { continue };
        let mark = history.len();
        history.push(user_message(&text, &images));
        let result = turn(&jobs, &setup, &mut history, &mut report);
        if result.is_err() {
            // Keep only complete exchanges: a half-run turn would fail the next request.
            history.truncate(mark);
        } else if let Some(message) = history.get_mut(mark).filter(|_| !images.is_empty()) {
            // Images are large to send again, so later turns keep the words and a note that images came with them.
            let note = format!("[{} image(s) were sent with this message; they are not kept.]", images.len());
            let words = if text.is_empty() { note } else { format!("{text}\n{note}") };
            *message = user_message(&words, &[]);
        }
        report(Event::Done(result));
    }
}

/// A user message. One with images is a list of parts: its words, then each image.
fn user_message(text: &str, images: &[String]) -> Value {
    if images.is_empty() {
        return json!({"role": "user", "content": text});
    }
    let mut parts = Vec::with_capacity(images.len() + 1);
    if !text.is_empty() {
        parts.push(json!({"type": "text", "text": text}));
    }
    parts.extend(images.iter().map(|url| json!({"type": "image_url", "image_url": {"url": url}})));
    json!({"role": "user", "content": parts})
}

/// One user message: call the model, run the tools it asks for, until it answers in text.
fn turn(jobs: &Receiver<Job>, setup: &Setup, history: &mut Vec<Value>, report: &mut impl FnMut(Event)) -> Result<String, String> {
    let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(setup.timeout)).build());
    let system = system_text(setup);
    let mut round = 0;
    loop {
        let mut messages = Vec::with_capacity(history.len() + 1);
        messages.push(json!({"role": "system", "content": system}));
        messages.extend(history.iter().cloned());
        let message = complete(&agent, setup, &messages)?;
        history.push(message.clone());
        let calls: Vec<ToolCall> = if setup.text_tools {
            text_tool_call(&text_of(&message), &setup.tools, format!("text_{round}")).into_iter().collect()
        } else {
            tool_calls(&message)
        };
        if calls.is_empty() {
            return Ok(text_of(&message));
        }
        if round == MAX_TOOL_ROUNDS {
            return Err(format!("The assistant stopped after {MAX_TOOL_ROUNDS} rounds of tool calls"));
        }
        round += 1;
        let answers = run_tools(jobs, &calls, report)?;
        if setup.text_tools {
            // A model without native tool calls reads each result as a message from the user.
            for (call, (_, content)) in calls.iter().zip(answers) {
                history.push(json!({"role": "user", "content": format!("Result of {}: {content}", call.name)}));
            }
        } else {
            for (id, content) in answers {
                history.push(json!({"role": "tool", "tool_call_id": id, "content": content}));
            }
        }
    }
}

/// The answers to one round of tool calls, in the order the model asked. Pages are read here, on the
/// worker, so the UI never waits on the network. The other calls go to the UI thread, which owns the
/// document, and their results come back in the same order.
fn run_tools(jobs: &Receiver<Job>, calls: &[ToolCall], report: &mut impl FnMut(Event)) -> Result<Vec<(String, String)>, String> {
    let app_calls: Vec<ToolCall> = calls.iter().filter(|call| call.name != web_page::TOOL_NAME).cloned().collect();
    let app_results = if app_calls.is_empty() {
        Vec::new()
    } else {
        report(Event::Tools(app_calls.clone()));
        wait_for_results(jobs, app_calls.len())?
    };
    let mut app_results = app_results.into_iter();
    let mut answers = Vec::with_capacity(calls.len());
    for call in calls {
        if call.name == web_page::TOOL_NAME {
            report(Event::Reading(web_page::url_argument(&call.arguments).unwrap_or_default()));
            answers.push((call.id.clone(), web_page::tool_result(&call.arguments)));
            continue;
        }
        let (id, content) = app_results.next().ok_or_else(|| MISMATCH.to_string())?;
        if id != call.id {
            return Err(MISMATCH.to_string());
        }
        answers.push((id, content));
    }
    Ok(answers)
}

/// The body of a chat-completions call. Thinking is off: its reasoning would have to be sent back
/// with every tool round, and the tool-choice modes that force a call are refused while it runs. The
/// tools are left out when there are none, since a model without native tool calls refuses the field.
pub fn request_body(model: &str, messages: &[Value], tools: &[Value]) -> Value {
    let mut body = json!({
        "model": model,
        "messages": messages,
        "thinking": {"type": "disabled"},
        "stream": false,
    });
    if !tools.is_empty() {
        body["tools"] = json!(tools);
    }
    body
}

/// The system prompt of a turn. A model without native tool calls is told its tools in the prompt,
/// and how to ask for one: a reply that is only a JSON object naming the tool and its arguments.
fn system_text(setup: &Setup) -> String {
    if !setup.text_tools {
        return setup.system.clone();
    }
    let mut text = format!(
        "{}\n\nYou can call these tools. To call one, reply with only a JSON object in this form: \
         {{\"tool\": \"<name>\", \"arguments\": {{...}}}}. Its result comes back to you as a message from the user. \
         When you have the answer, reply in plain text instead.\n\nTools:\n",
        setup.system
    );
    for tool in &setup.tools {
        let Some(function) = tool.get("function") else { continue };
        let name = function.get("name").and_then(Value::as_str).unwrap_or_default();
        let description = function.get("description").and_then(Value::as_str).unwrap_or_default();
        let parameters = function.get("parameters").cloned().unwrap_or_else(|| json!({}));
        text.push_str(&format!("- {name}: {description} Parameters: {parameters}\n"));
    }
    text
}

/// One chat-completions call: the assistant's message from the reply. The key goes only to the API;
/// a local endpoint is called without it.
fn complete(agent: &ureq::Agent, setup: &Setup, messages: &[Value]) -> Result<Value, String> {
    // A model without native tool calls gets its tools in the system prompt, not in the request.
    let tools: &[Value] = if setup.text_tools { &[] } else { &setup.tools };
    let body = request_body(&setup.model, messages, tools);
    let mut request = agent.post(&setup.endpoint);
    if !setup.key.is_empty() {
        request = request.header("Authorization", format!("Bearer {}", setup.key));
    }
    let provider = setup.provider;
    let reply = request.send_json(&body).map_err(|e| format!("Could not reach {provider}: {e}"))?;
    let status = reply.status();
    let text = reply.into_body().read_to_string().map_err(|e| format!("Could not read the reply from {provider}: {e}"))?;
    if !status.is_success() {
        return Err(format!("{provider} returned {status}: {}", error_message(&text)));
    }
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("{provider} sent a reply that is not JSON: {e}"))?;
    assistant_message(&value)
}

/// The tool call in a reply, for a model without native tool calls: a reply that is only a JSON
/// object naming one of the tools, optionally inside a code fence. Anything else is an answer.
fn text_tool_call(reply: &str, tools: &[Value], id: String) -> Option<ToolCall> {
    let call: Value = serde_json::from_str(strip_fence(reply.trim())).ok()?;
    let name = call.get("tool")?.as_str()?;
    let known = tools.iter().any(|tool| tool.pointer("/function/name").and_then(Value::as_str) == Some(name));
    if !known {
        return None;
    }
    let arguments = call.get("arguments").filter(|a| a.is_object()).cloned().unwrap_or_else(|| json!({}));
    Some(ToolCall { id, name: name.to_string(), arguments: arguments.to_string() })
}

/// The text inside a Markdown code fence, or the text itself when it is not fenced.
fn strip_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else { return text };
    let body = rest.split_once('\n').map_or(rest, |(_, body)| body);
    body.trim_end().strip_suffix("```").unwrap_or(body).trim()
}

/// The first choice's message of a chat-completions reply.
pub fn assistant_message(reply: &Value) -> Result<Value, String> {
    reply.pointer("/choices/0/message").filter(|m| m.is_object()).cloned().ok_or_else(|| "The reply had no message".to_string())
}

/// The tool calls in an assistant message (empty when it is a plain answer). A call missing its
/// id or name is skipped.
pub fn tool_calls(message: &Value) -> Vec<ToolCall> {
    message.get("tool_calls").and_then(Value::as_array).map(|calls| calls.iter().filter_map(tool_call).collect()).unwrap_or_default()
}

fn tool_call(call: &Value) -> Option<ToolCall> {
    Some(ToolCall {
        id: call.get("id")?.as_str()?.to_string(),
        name: call.pointer("/function/name")?.as_str()?.to_string(),
        arguments: call.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}").to_string(),
    })
}

/// The text of an assistant message (empty when it has none).
pub fn text_of(message: &Value) -> String {
    message.get("content").and_then(Value::as_str).unwrap_or_default().to_string()
}

/// The message in an error reply, or the start of the reply when it has none.
fn error_message(text: &str) -> String {
    serde_json::from_str::<Value>(text)
        .ok()
        .and_then(|v| v.pointer("/error/message").and_then(Value::as_str).map(str::to_string))
        .unwrap_or_else(|| text.chars().take(ERROR_CHARS).collect())
}

/// The results of the tool calls the worker asked for. Anything else that arrives meanwhile
/// means the turn can't go on.
fn wait_for_results(jobs: &Receiver<Job>, expected: usize) -> Result<Vec<(String, String)>, String> {
    match jobs.recv() {
        Ok(Job::ToolResults(results)) if results.len() == expected => Ok(results),
        Ok(Job::ToolResults(_)) => Err(MISMATCH.to_string()),
        Ok(Job::User { .. }) => Err("A new message arrived before the tools finished".to_string()),
        Err(_) => Err("The chat was closed".to_string()),
    }
}

#[cfg(test)]
#[path = "deepseek_tests.rs"]
mod tests;
