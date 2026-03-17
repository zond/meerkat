//! Formatting helpers for tool call and result display.

use serde_json::Value;

/// Tools that are internal plumbing — suppress their call/result output entirely.
const SILENT_TOOLS: &[&str] = &["peers", "wait", "list_meerkats"];

/// Whether a tool call should be shown to the user.
pub fn should_show_tool(name: &str) -> bool {
    !SILENT_TOOLS.contains(&name)
}

/// Extract a readable preview from tool call args.
pub fn tool_args_preview(name: &str, args: &Value) -> String {
    match name {
        "shell" | "bash" | "execute_command" => args
            .get("command")
            .and_then(|v| v.as_str())
            .map(|s| {
                // For multi-line commands, show just the first line.
                let first_line = s.lines().next().unwrap_or(s);
                let suffix = if s.contains('\n') { " ..." } else { "" };
                format!("\x1b[2m$ {}{suffix}\x1b[0m", truncate(first_line, 120))
            })
            .unwrap_or_default(),
        "read_file" | "read" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| format!("\x1b[2m{s}\x1b[0m"))
            .unwrap_or_default(),
        "write_file" | "write" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| format!("\x1b[2m{s}\x1b[0m"))
            .unwrap_or_default(),
        "list_directory" | "ls" => args
            .get("path")
            .and_then(|v| v.as_str())
            .map(|s| format!("\x1b[2m{s}\x1b[0m"))
            .unwrap_or_default(),
        "send" => {
            let target = args
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            let body = args
                .get("body")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let preview = truncate(body, 120);
            format!("\x1b[2m-> {target}: {preview}\x1b[0m")
        }
        "wire_peers" => {
            let a = args.get("a").and_then(|v| v.as_str()).unwrap_or("?");
            let b = args.get("b").and_then(|v| v.as_str()).unwrap_or("?");
            format!("\x1b[2m{a} <-> {b}\x1b[0m")
        }
        "task_create" | "mob_task_create" => {
            let desc = args
                .get("description")
                .and_then(|v| v.as_str())
                .or_else(|| args.get("subject").and_then(|v| v.as_str()))
                .unwrap_or("");
            format!("\x1b[2m{}\x1b[0m", truncate(desc, 120))
        }
        "task_update" | "mob_task_update" => {
            let id = args.get("task_id").and_then(|v| v.as_str()).unwrap_or("?");
            let status = args.get("status").and_then(|v| v.as_str()).unwrap_or("");
            let owner = args.get("owner").and_then(|v| v.as_str());
            if let Some(owner) = owner {
                format!("\x1b[2m{id} -> {status} (owner: {owner})\x1b[0m")
            } else if !status.is_empty() {
                format!("\x1b[2m{id} -> {status}\x1b[0m")
            } else {
                format!("\x1b[2m{id}\x1b[0m")
            }
        }
        _ => {
            let s = serde_json::to_string(args).unwrap_or_default();
            if s.len() > 120 {
                format!("\x1b[2m{}...\x1b[0m", &s[..117])
            } else if s != "{}" {
                format!("\x1b[2m{s}\x1b[0m")
            } else {
                String::new()
            }
        }
    }
}

/// Format a tool result for display. Shell results get special handling
/// to show exit code and the last few lines of stdout/stderr.
pub fn tool_result_preview(name: &str, result: &str, max_lines: usize) -> String {
    if matches!(name, "shell" | "bash" | "execute_command") {
        if let Ok(v) = serde_json::from_str::<Value>(result) {
            return format_shell_result(&v, max_lines);
        }
    }

    // Mob tool results: parse JSON and show a human-friendly summary.
    if matches!(name, "wire_peers") {
        return String::new(); // call preview already shows the info
    }
    if matches!(name, "task_create" | "mob_task_create") {
        if let Ok(v) = serde_json::from_str::<Value>(result) {
            let id = v.get("task_id").and_then(|v| v.as_str()).unwrap_or("");
            if !id.is_empty() {
                return format!("\x1b[2m-> {id}\x1b[0m");
            }
        }
        return String::new();
    }
    if matches!(name, "task_list" | "mob_task_list") {
        // Show count only.
        if let Ok(v) = serde_json::from_str::<Value>(result) {
            let count = v.get("tasks")
                .and_then(|v| v.as_array())
                .map_or(0, |a| a.len());
            return format!("\x1b[2m{count} tasks\x1b[0m");
        }
        return String::new();
    }
    if matches!(name, "send") {
        return String::new(); // call preview already shows the message
    }

    let line = result.lines().next().unwrap_or("");
    if line.len() > 200 {
        format!("\x1b[2m{}...\x1b[0m", &line[..197])
    } else if line.is_empty() {
        String::new()
    } else {
        format!("\x1b[2m{line}\x1b[0m")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max.saturating_sub(3)])
    } else {
        s.to_string()
    }
}

fn format_shell_result(v: &Value, max_lines: usize) -> String {
    let exit_code = v.get("exit_code").and_then(|e| e.as_i64());
    let stdout = v.get("stdout").and_then(|s| s.as_str()).unwrap_or("");
    let stderr = v.get("stderr").and_then(|s| s.as_str()).unwrap_or("");

    let mut parts = Vec::new();

    if let Some(code) = exit_code {
        if code != 0 {
            parts.push(format!("\x1b[31mexit {code}\x1b[0m"));
        }
    }

    let stderr_trimmed = stderr.trim();
    if !stderr_trimmed.is_empty() {
        let tail = tail_lines(stderr_trimmed, max_lines);
        parts.push(format!("\x1b[2m{tail}\x1b[0m"));
    }

    let stdout_trimmed = stdout.trim();
    if !stdout_trimmed.is_empty() {
        let tail = tail_lines(stdout_trimmed, max_lines);
        parts.push(format!("\x1b[2m{tail}\x1b[0m"));
    }

    parts.join(" ")
}

fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    let tail: Vec<&str> = lines[start..]
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let joined = tail.join(" | ");
    if joined.len() > 300 {
        format!("{}...", &joined[..297])
    } else {
        joined
    }
}
