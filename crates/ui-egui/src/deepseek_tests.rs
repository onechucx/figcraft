//! The DeepSeek client: the request and reply shapes, and the worker run against a local stand-in
//! for the API.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::mpsc;
use std::thread::JoinHandle;

use serde_json::json;

use super::*;

const TOOL_REPLY: &str = r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"list_commands","arguments":"{}"}}]}}]}"#;
const TEXT_REPLY: &str = r#"{"choices":[{"message":{"role":"assistant","content":"Here are the commands."}}]}"#;

/// The address of a local stand-in for the API's chat endpoint, with the listener that answers there.
fn listen() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/chat/completions", listener.local_addr().unwrap());
    (listener, url)
}

/// A local stand-in for the API. It answers each connection with the next (status, body) and hands
/// back the request bodies it received (null for a request with no body, such as a page read).
fn serve(listener: TcpListener, replies: Vec<(&'static str, String)>) -> JoinHandle<Vec<Value>> {
    std::thread::spawn(move || {
        let mut requests = Vec::new();
        for (status, body) in replies {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream);
            let mut length = 0;
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    length = value.trim().parse().unwrap();
                }
            }
            let mut received = vec![0; length];
            reader.read_exact(&mut received).unwrap();
            requests.push(if length == 0 { Value::Null } else { serde_json::from_slice(&received).unwrap() });
            let mut stream = reader.into_inner();
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
        }
        requests
    })
}

/// A reply in which the model says `text`.
fn text_reply(text: &str) -> String {
    json!({"choices": [{"message": {"role": "assistant", "content": text}}]}).to_string()
}

/// A reply in which the model asks for these tool calls, each as (id, name, arguments).
fn asks_for(calls: &[(&str, &str, &str)]) -> String {
    let calls: Vec<Value> =
        calls.iter().map(|(id, name, arguments)| json!({"id": id, "type": "function", "function": {"name": name, "arguments": arguments}})).collect();
    json!({"choices": [{"message": {"role": "assistant", "content": null, "tool_calls": calls}}]}).to_string()
}

fn setup(endpoint: String) -> Setup {
    Setup {
        provider: "DeepSeek",
        endpoint,
        key: "test-key".into(),
        model: "deepseek-flash".into(),
        system: "Be brief.".into(),
        tools: vec![json!({"type": "function", "function": {"name": "list_commands"}})],
        text_tools: false,
        timeout: TIMEOUT,
    }
}

/// The setup of the local model: no key, and its tools go through the reply text.
fn local_setup(endpoint: String) -> Setup {
    Setup { key: String::new(), model: "deepseek-hermes:latest".into(), text_tools: true, timeout: LOCAL_TIMEOUT, ..setup(endpoint) }
}

#[test]
fn a_request_turns_thinking_off_and_is_not_streamed() {
    let messages = [json!({"role": "user", "content": "hi"})];
    let body = request_body("deepseek-flash", &messages, &[]);
    assert_eq!(body["model"], "deepseek-flash");
    assert_eq!(body["messages"][0]["content"], "hi");
    assert_eq!(body["thinking"]["type"], "disabled");
    assert_eq!(body["stream"], false);
}

#[test]
fn tool_calls_are_read_from_the_message_and_a_call_without_an_id_is_skipped() {
    let message = json!({"role": "assistant", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "list_commands", "arguments": "{\"filter\":\"rect\"}"}},
        {"type": "function", "function": {"name": "run_command", "arguments": "{}"}},
        {"id": "call_3", "type": "function", "function": {"name": "get_node"}}
    ]});
    assert_eq!(
        tool_calls(&message),
        vec![
            ToolCall { id: "call_1".into(), name: "list_commands".into(), arguments: "{\"filter\":\"rect\"}".into() },
            ToolCall { id: "call_3".into(), name: "get_node".into(), arguments: "{}".into() },
        ]
    );
}

#[test]
fn a_plain_answer_has_text_and_no_tool_calls() {
    let message = json!({"role": "assistant", "content": "Done."});
    assert!(tool_calls(&message).is_empty());
    assert_eq!(text_of(&message), "Done.");
    assert_eq!(text_of(&json!({"role": "assistant", "content": null})), "");
}

#[test]
fn a_reply_needs_a_first_choice_with_a_message() {
    let reply = json!({"choices": [{"message": {"role": "assistant", "content": "hi"}}]});
    assert_eq!(assistant_message(&reply).unwrap()["content"], "hi");
    assert!(assistant_message(&json!({"choices": []})).is_err());
}

#[test]
fn an_error_reply_shows_the_api_message_or_the_start_of_the_body() {
    assert_eq!(error_message(r#"{"error":{"message":"Invalid API key"}}"#), "Invalid API key");
    assert_eq!(error_message(&"x".repeat(1000)).chars().count(), ERROR_CHARS);
}

#[test]
fn a_tool_round_is_answered_before_the_model_gives_its_text() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", TOOL_REPLY.into()), ("200 OK", TEXT_REPLY.into())]);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "What can you do?".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    jobs.send(Job::ToolResults(vec![("call_1".into(), r#"{"count":0}"#.into())])).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(
        events,
        vec![
            Event::Tools(vec![ToolCall { id: "call_1".into(), name: "list_commands".into(), arguments: "{}".into() }]),
            Event::Done(Ok("Here are the commands.".into())),
        ]
    );
    assert_eq!(requests[0]["thinking"]["type"], "disabled");
    let sent = requests[1]["messages"].as_array().unwrap();
    let last = sent.last().unwrap();
    assert_eq!(last["role"], "tool");
    assert_eq!(last["tool_call_id"], "call_1");
    assert_eq!(last["content"], r#"{"count":0}"#);
}

#[test]
fn a_page_the_model_asks_for_is_read_by_the_worker_and_its_text_goes_back() {
    let (listener, url) = listen();
    let page = url.replace("/chat/completions", "/page");
    let server = serve(
        listener,
        vec![
            ("200 OK", asks_for(&[("call_2", "fetch_url", &json!({"url": page}).to_string())])),
            // The stand-in serves the page as JSON, so its text comes back as it is.
            ("200 OK", "Hello from the page".into()),
            ("200 OK", text_reply("It says hello.")),
        ],
    );
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "Read the page".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(events, vec![Event::Reading(page.clone()), Event::Done(Ok("It says hello.".into()))]);
    let sent = requests[2]["messages"].as_array().unwrap();
    let last = sent.last().unwrap();
    assert_eq!(last["role"], "tool");
    assert_eq!(last["tool_call_id"], "call_2");
    let answer: Value = serde_json::from_str(last["content"].as_str().unwrap()).unwrap();
    assert_eq!(answer, json!({"url": page, "text": "Hello from the page"}));
}

#[test]
fn a_page_that_cannot_be_read_goes_back_to_the_model_as_an_error() {
    let (listener, url) = listen();
    let server = serve(
        listener,
        vec![
            ("200 OK", asks_for(&[("call_2", "fetch_url", r#"{"url":"http://127.0.0.1:1/x"}"#)])),
            ("200 OK", text_reply("That page did not open.")),
        ],
    );
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "Read it".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(events, vec![Event::Reading("http://127.0.0.1:1/x".into()), Event::Done(Ok("That page did not open.".into()))]);
    let last = requests[1]["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "tool");
    assert!(last["content"].as_str().unwrap().contains("\"error\""));
}

#[test]
fn app_tools_and_page_reads_in_one_round_keep_the_order_the_model_asked_for() {
    let (listener, url) = listen();
    let server = serve(
        listener,
        vec![
            ("200 OK", asks_for(&[("call_1", "list_commands", "{}"), ("call_2", "fetch_url", r#"{"url":"http://127.0.0.1:1/x"}"#)])),
            ("200 OK", text_reply("Done.")),
        ],
    );
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "Go".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    jobs.send(Job::ToolResults(vec![("call_1".into(), r#"{"count":0}"#.into())])).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(
        events,
        vec![
            Event::Tools(vec![ToolCall { id: "call_1".into(), name: "list_commands".into(), arguments: "{}".into() }]),
            Event::Reading("http://127.0.0.1:1/x".into()),
            Event::Done(Ok("Done.".into())),
        ]
    );
    let sent = requests[1]["messages"].as_array().unwrap();
    let tools: Vec<&Value> = sent.iter().filter(|m| m["role"] == "tool").collect();
    assert_eq!(tools[0]["tool_call_id"], "call_1");
    assert_eq!(tools[1]["tool_call_id"], "call_2");
}

#[test]
fn a_refused_call_ends_the_turn_with_the_api_message() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("401 Unauthorized", r#"{"error":{"message":"Authentication Fails"}}"#.into())]);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "hi".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    server.join().unwrap();
    assert_eq!(events, vec![Event::Done(Err("DeepSeek returned 401 Unauthorized: Authentication Fails".into()))]);
}

#[test]
fn tool_results_for_other_calls_end_the_turn() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", asks_for(&[("call_1", "list_commands", "{}")]))]);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "hi".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    jobs.send(Job::ToolResults(vec![("call_9".into(), "{}".into())])).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    server.join().unwrap();
    assert_eq!(events.last(), Some(&Event::Done(Err(MISMATCH.into()))));
}

#[test]
fn tool_results_with_no_turn_waiting_for_them_are_ignored() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", TEXT_REPLY.into())]);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::ToolResults(vec![("call_9".into(), "stale".into())])).unwrap();
    jobs.send(Job::User { text: "hi".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    server.join().unwrap();
    assert_eq!(events, vec![Event::Done(Ok("Here are the commands.".into()))]);
}

#[test]
fn the_vision_model_is_one_of_the_models_the_panel_offers() {
    assert!(CHOICES.iter().any(|c| c.model == VISION_MODEL && c.reads_images));
}

#[test]
fn an_image_goes_with_its_message_as_an_image_part() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", text_reply("It is a cat."))]);
    let picture = "data:image/png;base64,iVBORw0KGgo=".to_string();
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "What is this?".into(), images: vec![picture.clone()], setup: setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(events, vec![Event::Done(Ok("It is a cat.".into()))]);
    let last = requests[0]["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last["role"], "user");
    assert_eq!(last["content"][0], json!({"type": "text", "text": "What is this?"}));
    assert_eq!(last["content"][1], json!({"type": "image_url", "image_url": {"url": picture}}));
}

#[test]
fn later_turns_keep_the_words_of_an_image_message_and_not_the_image() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", text_reply("It is a cat.")), ("200 OK", text_reply("Yes."))]);
    let (jobs, inbox) = mpsc::channel();
    let picture = "data:image/png;base64,AAAA".to_string();
    jobs.send(Job::User { text: "What is this?".into(), images: vec![picture], setup: setup(url.clone()) }).unwrap();
    jobs.send(Job::User { text: "Is it a cat?".into(), images: Vec::new(), setup: setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(events, vec![Event::Done(Ok("It is a cat.".into())), Event::Done(Ok("Yes.".into()))]);
    let earlier = requests[1]["messages"][1]["content"].as_str().unwrap();
    assert!(earlier.starts_with("What is this?"), "{earlier}");
    assert!(earlier.contains("1 image(s)"), "{earlier}");
    assert!(!requests[1].to_string().contains("image_url"));
}

#[test]
fn every_local_choice_needs_no_key_and_reads_no_images() {
    let local: Vec<&Choice> = CHOICES.iter().filter(|c| c.endpoint == LOCAL_ENDPOINT).collect();
    assert_eq!(local.len(), 5);
    for choice in local {
        assert!(!choice.needs_key, "{}", choice.label);
        assert!(choice.text_tools, "{}", choice.label);
        assert!(!choice.reads_images, "{}", choice.label);
    }
}

#[test]
fn an_unreachable_local_model_is_told_how_to_start_it() {
    let local = setup(LOCAL_ENDPOINT.to_string());
    let message = unreachable_message(&local, "connection refused");
    assert!(message.contains("ollama serve"), "{message}");
    assert!(message.contains("connection refused"), "{message}");
    let api = setup(ENDPOINT.to_string());
    assert_eq!(unreachable_message(&api, "connection refused"), "Could not reach DeepSeek: connection refused");
}

#[test]
fn the_panel_lists_two_providers_and_names_the_first_model_of_each() {
    assert_eq!(providers(), vec!["DeepSeek", "Antigravity"]);
    assert_eq!(first_model("DeepSeek"), Some(0));
    assert_eq!(first_model("Antigravity"), Some(3));
    assert_eq!(first_model("Nobody"), None);
}

#[test]
fn antigravity_offers_a_planner_and_coders_of_different_sizes() {
    let models: Vec<&str> = CHOICES.iter().filter(|c| c.route == "Antigravity").map(|c| c.model).collect();
    assert_eq!(models, ["antigravity:latest", "qwen2.5-coder:14b", "qwen3-coder:30b", "qwen3.5:35b"]);
}

#[test]
fn antigravity_is_sent_its_own_model_with_its_tools_in_the_prompt() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", text_reply("Here are the commands."))]);
    let choice = CHOICES.iter().find(|c| c.route == "Antigravity").unwrap();
    let antigravity = Setup { provider: choice.route, model: choice.model.into(), ..local_setup(url) };
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "What can you do?".into(), images: Vec::new(), setup: antigravity }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(events, vec![Event::Done(Ok("Here are the commands.".into()))]);
    assert_eq!(requests[0]["model"], "antigravity:latest");
    assert!(requests[0].get("tools").is_none());
    let system = requests[0]["messages"][0]["content"].as_str().unwrap();
    assert!(system.contains("- list_commands:"), "{system}");
}

#[test]
fn a_failed_call_names_the_provider_that_failed() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("500 Internal Server Error", r#"{"error":{"message":"model not found"}}"#.into())]);
    let mut failing = setup(url);
    failing.provider = "Antigravity";
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "hi".into(), images: Vec::new(), setup: failing }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    server.join().unwrap();
    assert_eq!(events, vec![Event::Done(Err("Antigravity returned 500 Internal Server Error: model not found".into()))]);
}

#[test]
fn a_local_model_gets_its_tools_in_the_prompt_and_no_tools_field() {
    let (listener, url) = listen();
    let server = serve(listener, vec![("200 OK", text_reply("Here are the commands."))]);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "What can you do?".into(), images: Vec::new(), setup: local_setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(events, vec![Event::Done(Ok("Here are the commands.".into()))]);
    assert!(requests[0].get("tools").is_none());
    let system = requests[0]["messages"][0]["content"].as_str().unwrap();
    assert!(system.starts_with("Be brief."), "{system}");
    assert!(system.contains("- list_commands:"), "{system}");
}

#[test]
fn a_tool_call_in_a_local_reply_runs_the_tool_and_its_result_goes_back_as_a_message() {
    let (listener, url) = listen();
    let fenced = "```json\n{\"tool\": \"list_commands\", \"arguments\": {\"filter\": \"rect\"}}\n```";
    let server = serve(listener, vec![("200 OK", text_reply(fenced)), ("200 OK", text_reply("There are two."))]);
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "Which commands?".into(), images: Vec::new(), setup: local_setup(url) }).unwrap();
    jobs.send(Job::ToolResults(vec![("text_0".into(), r#"{"count":2}"#.into())])).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    let requests = server.join().unwrap();

    assert_eq!(
        events,
        vec![
            Event::Tools(vec![ToolCall { id: "text_0".into(), name: "list_commands".into(), arguments: r#"{"filter":"rect"}"#.into() }]),
            Event::Done(Ok("There are two.".into())),
        ]
    );
    let last = requests[1]["messages"].as_array().unwrap().last().unwrap().clone();
    assert_eq!(last, json!({"role": "user", "content": r#"Result of list_commands: {"count":2}"#}));
}

#[test]
fn a_local_reply_that_names_no_known_tool_or_is_not_only_a_call_is_the_answer() {
    let (listener, url) = listen();
    let server = serve(
        listener,
        vec![
            ("200 OK", text_reply(r#"{"tool": "delete_everything", "arguments": {}}"#)),
            ("200 OK", text_reply(r#"I will call {"tool": "list_commands"} now."#)),
        ],
    );
    let (jobs, inbox) = mpsc::channel();
    jobs.send(Job::User { text: "Clean up".into(), images: Vec::new(), setup: local_setup(url.clone()) }).unwrap();
    jobs.send(Job::User { text: "Again".into(), images: Vec::new(), setup: local_setup(url) }).unwrap();
    drop(jobs);
    let mut events = Vec::new();
    run(inbox, |event| events.push(event));
    server.join().unwrap();

    assert_eq!(
        events,
        vec![
            Event::Done(Ok(r#"{"tool": "delete_everything", "arguments": {}}"#.into())),
            Event::Done(Ok(r#"I will call {"tool": "list_commands"} now."#.into())),
        ]
    );
}

#[test]
fn a_local_reply_is_a_call_only_when_it_is_just_the_call_object() {
    let tools = setup(String::new()).tools;
    let call = |reply: &str| text_tool_call(reply, &tools, "text_0".into());
    assert_eq!(call(r#"{"tool":"list_commands"}"#), Some(ToolCall { id: "text_0".into(), name: "list_commands".into(), arguments: "{}".into() }));
    assert_eq!(
        call("```json\n{\"tool\":\"list_commands\",\"arguments\":{\"filter\":\"x\"}}\n```"),
        Some(ToolCall { id: "text_0".into(), name: "list_commands".into(), arguments: r#"{"filter":"x"}"#.into() })
    );
    assert_eq!(call(r#"{"tool":"list_commands","arguments":"rect"}"#).unwrap().arguments, "{}");
    assert_eq!(call(r#"{"tool":"delete_everything"}"#), None);
    assert_eq!(call("Sure. {\"tool\":\"list_commands\"}"), None);
}

#[test]
fn a_fence_is_taken_off_only_when_it_wraps_the_text() {
    assert_eq!(strip_fence("```json\n{}\n```"), "{}");
    assert_eq!(strip_fence("```\n{}\n```"), "{}");
    assert_eq!(strip_fence("{}"), "{}");
}
