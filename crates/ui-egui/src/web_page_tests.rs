//! The web-page tool: HTML reduced to readable text, and fetches against a local stand-in server.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;

use serde_json::{Value, json};

use super::*;

/// A local stand-in for a web server: its address, which answers one request with the given status,
/// media type and body.
fn serve(status: &str, content_type: &str, body: String) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/page", listener.local_addr().unwrap());
    let head = format!("HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    std::thread::spawn(move || {
        let Ok((stream, _)) = listener.accept() else { return };
        let mut reader = BufReader::new(stream);
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                break;
            }
        }
        let mut stream = reader.into_inner();
        // The client may stop reading early (a page longer than the limit), so a failed write is fine.
        let _ = stream.write_all(head.as_bytes());
        let _ = stream.write_all(body.as_bytes());
    });
    url
}

#[test]
fn a_page_reads_as_its_title_then_its_text() {
    let html = "<html><head><title>Plans</title><style>p { color: red }</style></head>\
        <body><h1>Hello</h1><p>One &amp; two</p><script>alert(1)</script></body></html>";
    assert_eq!(html_to_text(html), "Plans\n\nHello\n\nOne & two");
}

#[test]
fn scripts_styles_and_comments_are_not_shown() {
    assert_eq!(html_to_text("<!-- hidden --><p>Shown</p><script>var a = '<p>';</script><noscript>Enable JS</noscript>"), "Shown");
}

#[test]
fn inline_words_stay_apart_and_block_elements_start_lines() {
    assert_eq!(html_to_text("<b>Hello</b> <i>world</i><br>next"), "Hello world\nnext");
}

#[test]
fn character_references_decode_and_unknown_ones_stay() {
    assert_eq!(html_to_text("&lt;b&gt; &#65; &#x42; &bogus; &amp;"), "<b> A B &bogus; &");
}

#[test]
fn a_less_than_sign_in_text_stays_and_an_unclosed_tag_is_dropped() {
    assert_eq!(html_to_text("1 < 2 and 3 > 2"), "1 < 2 and 3 > 2");
    assert_eq!(html_to_text("Text <b"), "Text");
}

#[test]
fn only_http_and_https_addresses_are_read() {
    assert!(fetch("file:///C:/Windows/win.ini").unwrap_err().contains("http"));
    assert!(fetch("ftp://example.com/a").unwrap_err().contains("http"));
}

#[test]
fn a_page_is_read_from_the_web_as_text() {
    let url = serve("200 OK", "text/html; charset=utf-8", "<title>Plans</title><p>Ship it</p>".into());
    assert_eq!(fetch(&url).unwrap(), "Plans\n\nShip it");
}

#[test]
fn a_json_page_is_read_as_it_is() {
    let url = serve("200 OK", "application/json", r#"{"ok": true}"#.into());
    assert_eq!(fetch(&url).unwrap(), r#"{"ok": true}"#);
}

#[test]
fn an_image_is_refused_as_not_text() {
    let url = serve("200 OK", "image/png", "not really a png".into());
    let why = fetch(&url).unwrap_err();
    assert!(why.contains("not text"), "{why}");
}

#[test]
fn a_page_that_failed_says_its_status() {
    let url = serve("404 Not Found", "text/html", "<p>Gone</p>".into());
    assert!(fetch(&url).unwrap_err().contains("404"));
}

#[test]
fn a_page_that_cannot_be_reached_says_so() {
    assert!(fetch("http://127.0.0.1:1/").unwrap_err().contains("Could not open"));
}

#[test]
fn a_page_longer_than_the_limit_is_cut_with_a_note() {
    let url = serve("200 OK", "text/plain", "a".repeat(MAX_BYTES as usize + 10));
    let text = fetch(&url).unwrap();
    assert!(text.ends_with("Only its start was read.]"), "{}", &text[text.len() - 60..]);
}

#[test]
fn the_tool_answers_with_the_text_and_cuts_long_text() {
    let url = serve("200 OK", "text/plain", "b".repeat(MAX_TEXT_CHARS + 5));
    let answer: Value = serde_json::from_str(&tool_result(&json!({"url": url}).to_string())).unwrap();
    assert_eq!(answer["url"], url);
    assert!(answer["text"].as_str().unwrap().ends_with("\n[5 more characters not shown]"));
}

#[test]
fn the_tool_says_when_it_was_given_no_address() {
    let answer: Value = serde_json::from_str(&tool_result("{}")).unwrap();
    assert_eq!(answer["error"], "fetch_url needs the page's url.");
}

#[test]
fn the_address_argument_is_trimmed_and_missing_when_absent() {
    assert_eq!(url_argument(r#"{"url":"  https://example.com  "}"#).as_deref(), Some("https://example.com"));
    assert_eq!(url_argument(r#"{"url":3}"#), None);
    assert_eq!(url_argument("not json"), None);
}
