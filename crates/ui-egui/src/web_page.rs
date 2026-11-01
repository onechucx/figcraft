//! The chat's web-page tool. The model gives an http or https address and gets back the page's
//! text, with the title first and the markup, scripts and styles left out. Pages are read on the
//! chat's worker thread, so the UI never waits on the network. Native builds only.

use std::io::Read;
use std::time::Duration;

use serde_json::{Value, json};

use crate::ai_chat::clip;

/// The tool's name, as the model calls it.
pub const TOOL_NAME: &str = "fetch_url";
/// Characters of a page's text the model gets. Longer text is cut, with a note.
pub const MAX_TEXT_CHARS: usize = 20_000;
/// Bytes of a page that are read at most. The rest is not read.
const MAX_BYTES: u64 = 2 * 1024 * 1024;
/// How long a page may take, from connecting to the last byte.
const TIMEOUT: Duration = Duration::from_secs(20);
/// Elements whose contents are never shown: code, styling and markup that is not content.
const SKIPPED: &[&str] = &["script", "style", "noscript", "svg", "template"];
/// Elements that start a new line in the text.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "html",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "title",
    "tr",
    "ul",
];

/// The address the model asked for, when its arguments name one.
pub fn url_argument(arguments: &str) -> Option<String> {
    let args: Value = serde_json::from_str(arguments).ok()?;
    args.get("url")?.as_str().map(|url| url.trim().to_string())
}

/// The tool's answer for the model: the page's text as JSON, or the reason it could not be read.
pub fn tool_result(arguments: &str) -> String {
    let outcome = match url_argument(arguments) {
        Some(url) => fetch(&url).map(|text| json!({"url": url, "text": clip(&text, MAX_TEXT_CHARS)})),
        None => Err("fetch_url needs the page's url.".to_string()),
    };
    match outcome {
        Ok(answer) => answer.to_string(),
        Err(error) => json!({ "error": error }).to_string(),
    }
}

/// The readable text of a web address. Only http and https are read, and a page that is not text
/// (an image, a file to download) is refused.
pub fn fetch(url: &str) -> Result<String, String> {
    let url = url.trim();
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return Err("Only http and https addresses can be read.".to_string());
    }
    let agent = ureq::Agent::new_with_config(ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(TIMEOUT)).build());
    let reply = agent.get(url).call().map_err(|e| format!("Could not open {url}: {e}"))?;
    let status = reply.status();
    if !status.is_success() {
        return Err(format!("The page answered with {status}."));
    }
    let kind = reply.body().mime_type().unwrap_or_default().to_ascii_lowercase();
    if !is_text(&kind) {
        return Err(format!("The page is {kind}, not text."));
    }
    let mut bytes = Vec::new();
    reply.into_body().into_reader().take(MAX_BYTES + 1).read_to_end(&mut bytes).map_err(|e| format!("Could not read the page: {e}"))?;
    let longer = bytes.len() as u64 > MAX_BYTES;
    bytes.truncate(MAX_BYTES as usize);
    let raw = String::from_utf8_lossy(&bytes);
    let mut text = if kind.is_empty() || kind.contains("html") { html_to_text(&raw) } else { tidy(&raw) };
    if longer {
        text.push_str("\n[The page is longer than 2 MiB. Only its start was read.]");
    }
    Ok(text)
}

/// Whether a media type is text the model can read.
fn is_text(kind: &str) -> bool {
    kind.is_empty() || kind.starts_with("text/") || ["html", "xml", "json"].iter().any(|word| kind.contains(word))
}

/// The readable text of an HTML page: markup, scripts and styles left out, block elements as line
/// breaks, character references decoded, whitespace tidied.
fn html_to_text(html: &str) -> String {
    let mut out = String::new();
    let mut rest = html;
    while let Some(at) = rest.find('<') {
        push_text(&mut out, &rest[..at]);
        rest = &rest[at..];
        if let Some(after) = rest.strip_prefix("<!--") {
            rest = after.find("-->").map_or("", |end| &after[end + 3..]);
            continue;
        }
        // A '<' that does not open a tag is text, as in "1 < 2".
        if !rest[1..].starts_with(|c: char| c.is_ascii_alphabetic() || c == '/' || c == '!') {
            out.push('<');
            rest = &rest[1..];
            continue;
        }
        let Some(end) = rest.find('>') else {
            // An unclosed tag at the end of the page has nothing readable after it.
            rest = "";
            break;
        };
        let tag = &rest[1..end];
        rest = &rest[end + 1..];
        let closing = tag.starts_with('/');
        let name = tag.trim_start_matches('/').chars().take_while(|c| c.is_ascii_alphanumeric()).collect::<String>().to_ascii_lowercase();
        if !closing && SKIPPED.contains(&name.as_str()) {
            rest = skip_element(rest, &name);
        }
        if BLOCKS.contains(&name.as_str()) {
            out.push('\n');
        }
    }
    push_text(&mut out, rest);
    tidy(&out)
}

/// `after` with the element `name` skipped up to and including its closing tag. Without a closing
/// tag the rest of the page is skipped, since such an element's contents are not text.
fn skip_element<'a>(after: &'a str, name: &str) -> &'a str {
    let close = format!("</{name}");
    match after.to_ascii_lowercase().find(&close) {
        Some(start) => after[start..].find('>').map_or("", |end| &after[start + end + 1..]),
        None => "",
    }
}

/// Add the words of `text` to `out`, with single spaces between them.
fn push_text(out: &mut String, text: &str) {
    let decoded = decode_entities(text);
    let words: Vec<&str> = decoded.split_whitespace().collect();
    if words.is_empty() {
        if !decoded.is_empty() {
            separate(out);
        }
        return;
    }
    if decoded.starts_with(char::is_whitespace) {
        separate(out);
    }
    out.push_str(&words.join(" "));
    if decoded.ends_with(char::is_whitespace) {
        separate(out);
    }
}

/// A single space at the end of `out`, unless it already ends in one or in a line break.
fn separate(out: &mut String) {
    if !out.is_empty() && !out.ends_with([' ', '\n']) {
        out.push(' ');
    }
}

/// `text` with its character references (`&amp;`, `&#39;`, `&#x41;`) decoded. Unknown ones stay.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        // A reference is short, so its ';' is looked for only near the '&'.
        let end = rest.as_bytes()[..rest.len().min(12)].iter().position(|&b| b == b';');
        match end.and_then(|end| decode_entity(&rest[1..end]).map(|c| (c, end))) {
            Some((c, end)) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The character a reference names (the part between '&' and ';').
fn decode_entity(name: &str) -> Option<char> {
    match name {
        "amp" => Some('&'),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "nbsp" => Some(' '),
        _ => {
            let number = name.strip_prefix('#')?;
            let code = match number.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok()?,
                None => number.parse().ok()?,
            };
            char::from_u32(code)
        }
    }
}

/// `text` with each line trimmed, and runs of blank lines (and blanks at the ends) removed.
fn tidy(text: &str) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for line in text.split('\n').map(str::trim) {
        if line.is_empty() && matches!(lines.last(), None | Some(&"")) {
            continue;
        }
        lines.push(line);
    }
    lines.join("\n").trim().to_string()
}

#[cfg(test)]
#[path = "web_page_tests.rs"]
mod tests;
