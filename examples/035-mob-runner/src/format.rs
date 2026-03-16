//! Formatting helpers for tool call and result display.

use serde_json::Value;

/// Extract a readable preview from tool call args.
pub fn tool_args_preview(name: &str, args: &Value) -> String {
    match name {
        "shell" | "bash" | "execute_command" => args
            .get("command")
            .and_then(|v| v.as_str())
            .map(|s| format!("\x1b[2m$ {s}\x1b[0m"))
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
    // Try to parse shell-style JSON results.
    if matches!(name, "shell" | "bash" | "execute_command") {
        if let Ok(v) = serde_json::from_str::<Value>(result) {
            return format_shell_result(&v, max_lines);
        }
    }

    // Generic: show first line, truncated.
    let line = result.lines().next().unwrap_or("");
    if line.len() > 200 {
        format!("\x1b[2m{}...\x1b[0m", &line[..197])
    } else if line.is_empty() {
        String::new()
    } else {
        format!("\x1b[2m{line}\x1b[0m")
    }
}

/// Format a shell command result showing exit code and tail of output.
fn format_shell_result(v: &Value, max_lines: usize) -> String {
    let exit_code = v.get("exit_code").and_then(|e| e.as_i64());
    let stdout = v.get("stdout").and_then(|s| s.as_str()).unwrap_or("");
    let stderr = v.get("stderr").and_then(|s| s.as_str()).unwrap_or("");

    let mut parts = Vec::new();

    // Exit code (only show if non-zero).
    if let Some(code) = exit_code {
        if code != 0 {
            parts.push(format!("\x1b[31mexit {code}\x1b[0m"));
        }
    }

    // Show tail of stderr if present (often more useful than stdout for errors).
    let stderr_trimmed = stderr.trim();
    if !stderr_trimmed.is_empty() {
        let tail = tail_lines(stderr_trimmed, max_lines);
        parts.push(format!("\x1b[2m{tail}\x1b[0m"));
    }

    // Show tail of stdout.
    let stdout_trimmed = stdout.trim();
    if !stdout_trimmed.is_empty() {
        let tail = tail_lines(stdout_trimmed, max_lines);
        parts.push(format!("\x1b[2m{tail}\x1b[0m"));
    }

    parts.join(" ")
}

/// Get the last N lines of a string, joined with " | ".
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
