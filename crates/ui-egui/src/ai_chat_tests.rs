//! The chat's tools: what the model may run, how results are cut, and the errors it reads back.

use figcraft_engine::Session;
use serde_json::{Value, json};

use super::*;
use crate::{FigApp, NoServices};

fn app() -> FigApp {
    FigApp::new(Session::new(), Box::new(NoServices))
}

fn call(name: &str, arguments: &str) -> ToolCall {
    ToolCall { id: "call_1".into(), name: name.into(), arguments: arguments.into() }
}

#[test]
fn clip_leaves_short_text_alone_and_notes_what_it_cut() {
    assert_eq!(clip("abc", 3), "abc");
    assert_eq!(clip("abcdef", 3), "abc\n[3 more characters not shown]");
}

#[test]
fn clip_counts_characters_not_bytes() {
    assert_eq!(clip("ééé", 2), "éé\n[1 more characters not shown]");
}

#[test]
fn the_assistant_may_not_revert_or_force_close_a_file() {
    assert!(refused("file.revert", &json!({})).is_some());
    assert!(refused("file.close", &json!({"force": true})).is_some());
    assert!(refused("file.close", &json!({})).is_none());
    assert!(refused("file.close", &json!({"force": false})).is_none());
    assert!(refused("shape.rectangle", &json!({})).is_none());
}

#[test]
fn list_commands_lists_every_command_and_filters_by_id_or_label() {
    let all = list_commands(None);
    assert_eq!(all["count"].as_u64(), Some(figcraft_engine::all_commands().len() as u64));
    assert_eq!(list_commands(Some("no such command anywhere"))["count"], 0);
    let rect = list_commands(Some("rectangle"));
    let rows = rect["commands"].as_array().unwrap();
    assert!(rows.iter().any(|c| c["id"] == "shape.rectangle"));
    assert!(
        rows.iter().all(|c| c["id"].as_str().unwrap().contains("rectangle") || c["label"].as_str().unwrap().to_lowercase().contains("rectangle"))
    );
}

#[test]
fn the_tools_are_the_five_the_prompt_describes() {
    let tools = tool_definitions();
    let names: Vec<&str> = tools.iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["list_commands", "run_command", "inspect_document", "get_node", "fetch_url"]);
}

#[test]
fn an_unknown_tool_or_bad_arguments_come_back_as_errors_the_model_can_read() {
    let ctx = egui::Context::default();
    let mut app = app();
    assert_eq!(run_tool(&mut app, &ctx, &call("nope", "{}")), json!({"error": "There is no tool named nope."}).to_string());
    assert!(run_tool(&mut app, &ctx, &call("list_commands", "{")).contains("The arguments are not valid JSON"));
}

#[test]
fn run_command_refuses_a_revert_without_running_it() {
    let ctx = egui::Context::default();
    let mut app = app();
    let result: Value = serde_json::from_str(&run_tool(&mut app, &ctx, &call("run_command", r#"{"id":"file.revert"}"#))).unwrap();
    assert!(result["error"].as_str().unwrap().contains("file.revert"));
}

#[test]
fn run_command_runs_a_command_the_way_a_script_would() {
    let ctx = egui::Context::default();
    let mut app = app();
    let mut out = String::new();
    crate::windows::guard(&mut app, &ctx, |app| {
        out = run_tool(app, &ctx, &call("run_command", r#"{"id":"shape.rectangle","params":{"x":0,"y":0,"width":20,"height":10}}"#));
    });
    assert!(!out.contains("\"error\""), "{out}");
}

/// A file in the temp folder with `bytes`, its name kept apart from other tests' files.
fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("figcraft-ai-{}-{name}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[test]
fn an_image_is_recognised_by_its_first_bytes() {
    assert_eq!(image_type(b"\x89PNG\r\n\x1a\nrest"), Some("image/png"));
    assert_eq!(image_type(&[0xFF, 0xD8, 0xFF, 0xE0]), Some("image/jpeg"));
    assert_eq!(image_type(b"GIF89a."), Some("image/gif"));
    assert_eq!(image_type(b"RIFF\x24\x00\x00\x00WEBPVP8 "), Some("image/webp"));
    assert_eq!(image_type(b"RIFF\x24\x00\x00\x00WAVEfmt "), None);
    assert_eq!(image_type(b"%PDF-1.4"), None);
    assert_eq!(image_type(b""), None);
}

#[test]
fn an_attached_image_waits_as_a_data_url_and_switches_to_the_model_that_reads_images() {
    let mut app = app();
    let path = temp_file("cat.png", b"\x89PNG\r\n\x1a\nbody");
    attach(&mut app, path.clone());
    let _ = std::fs::remove_file(&path);
    assert_eq!(app.ai.pictures.len(), 1);
    assert!(app.ai.pictures[0].name.ends_with("cat.png"));
    assert!(app.ai.pictures[0].url.starts_with("data:image/png;base64,"));
    assert_eq!(app.ai.model, vision_index());
    assert!(app.ai.log.last().is_some_and(|l| l.text.contains("deepseek-flash")));
}

#[test]
fn a_file_that_is_not_an_image_is_refused() {
    let mut app = app();
    let path = temp_file("notes.txt", b"plain words");
    attach(&mut app, path.clone());
    let _ = std::fs::remove_file(&path);
    assert!(app.ai.pictures.is_empty());
    assert!(app.ai.log.last().is_some_and(|l| l.text.ends_with("is not a PNG, JPEG, GIF or WebP image.")));
}

#[test]
fn no_more_than_four_images_go_with_one_message() {
    let mut app = app();
    let path = temp_file("dot.gif", b"GIF89a.");
    for _ in 0..=MAX_IMAGES {
        attach(&mut app, path.clone());
    }
    let _ = std::fs::remove_file(&path);
    assert_eq!(app.ai.pictures.len(), MAX_IMAGES);
    assert!(app.ai.log.last().is_some_and(|l| l.text.contains("images can go with one message")));
}

#[test]
fn an_image_waits_for_the_model_that_reads_images() {
    assert_ne!(vision_index(), 0);
    let ctx = egui::Context::default();
    let mut app = app();
    app.ai.key = "test-key".into();
    app.ai.model = 0;
    app.ai.pictures.push(Picture { name: "cat.png".into(), url: "data:image/png;base64,AAAA".into() });
    app.ai.input = "What is this?".into();
    app.ai.submit(&ctx);
    assert!(!app.ai.busy);
    assert_eq!(app.ai.pictures.len(), 1);
    assert_eq!(app.ai.input, "What is this?");
    assert!(app.ai.log.last().is_some_and(|l| l.text.contains("deepseek-flash")));
}

#[test]
fn a_sub_agent_adds_its_instructions_after_the_shared_ones() {
    assert_eq!(system_prompt(0), SYSTEM_PROMPT);
    assert_eq!(system_prompt(99), SYSTEM_PROMPT);
    assert_eq!(system_prompt(1), format!("{SYSTEM_PROMPT}\n\n{}", SUB_AGENTS[1]));
    assert_ne!(system_prompt(2), system_prompt(3));
}

#[test]
fn the_panel_opens_on_the_local_model_so_no_key_is_asked_for() {
    let app = app();
    assert_eq!(app.ai.choice().route, "DeepSeek");
    assert_eq!(app.ai.choice().label, "hermes (local)");
    assert!(!app.ai.choice().needs_key);
    assert!(app.ai.setup().key.is_empty());
    assert_eq!(app.ai.setup().endpoint, deepseek::LOCAL_ENDPOINT);
}

#[test]
fn switching_provider_lands_on_its_first_model_and_keeping_the_provider_keeps_its_model() {
    let mut app = app();
    app.ai.model = deepseek::first_model("DeepSeek").unwrap_or(0) + 1;
    switch_provider(&mut app.ai, "DeepSeek");
    assert_eq!(app.ai.model, 1);
    switch_provider(&mut app.ai, "Antigravity");
    assert_eq!(app.ai.choice().label, "antigravity");
    switch_provider(&mut app.ai, "Nobody");
    assert_eq!(app.ai.choice().label, "antigravity");
}

#[test]
fn a_chosen_antigravity_model_is_sent_its_own_name_and_no_key() {
    let mut app = app();
    switch_provider(&mut app.ai, "Antigravity");
    app.ai.model = deepseek::CHOICES.iter().position(|c| c.model == "qwen3.5:35b").expect("the planner");
    let setup = app.ai.setup();
    assert_eq!(setup.provider, "Antigravity");
    assert_eq!(setup.model, "qwen3.5:35b");
    assert_eq!(setup.endpoint, deepseek::LOCAL_ENDPOINT);
    assert!(setup.key.is_empty());
}

#[test]
fn the_request_carries_the_provider_and_the_sub_agents_instructions() {
    let mut app = app();
    // The panel opens on the local model, which takes no key, so pick an API model to send one.
    app.ai.model = deepseek::CHOICES.iter().position(|c| c.needs_key).expect("an API model");
    app.ai.key = "test-key".into();
    app.ai.sub_agent = 2;
    let setup = app.ai.setup();
    assert_eq!(setup.provider, "DeepSeek");
    assert_eq!(setup.key, "test-key");
    assert!(setup.system.ends_with(SUB_AGENTS[2]));

    switch_provider(&mut app.ai, "Antigravity");
    let setup = app.ai.setup();
    assert_eq!(setup.provider, "Antigravity");
    assert_eq!(setup.model, "antigravity:latest");
    assert!(setup.key.is_empty(), "a local model is sent no key");
    assert!(setup.system.starts_with(SYSTEM_PROMPT));
}
