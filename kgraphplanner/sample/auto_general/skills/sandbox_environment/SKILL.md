# Sandbox Environment Skill

You have a secure sandbox for code execution, file editing, and report
generation.  Execute commands via the `execute` tool.

## File Editing

Use `edit` for precise, targeted file modifications:
- **Replace:** `edit <file> --old "exact old text" --new "replacement text"`
- **View:** `edit <file> --view` or `edit <file> --view --range 10:25`
- **Insert:** `edit <file> --insert <line> --text "new text"`
- **Create:** `edit <file> --create --content "file contents"`

For multi-file patches, use `apply_patch` with V4A format:
```
apply_patch << 'EOF'
*** Update File: /workspace/file.py
@@ anchor_line
 context line
-old line
+new line
*** End Patch
EOF
```

For writing new files with lots of content, use heredoc:
```
cat > /workspace/file.md << 'EOF'
content here
EOF
```

## Python

```
python3 -c "print('hello')"
pip install numpy matplotlib
python3 << 'EOF'
# multi-line scripts
EOF
```

## Reports (Markdown → PDF)

```
reportgen input.md -o output.pdf --title "Title" --author "Author" --toc
```

## Git

Pre-configured at `/workspace`.  No setup needed.
```
git add . && git commit -m "message"
git status / git log --oneline / git diff
git push
```

**IMPORTANT:** Do NOT use `git add -A` — the sandbox does not support the
`-A` flag.  Use `git add .` or `git add <specific-paths>` instead.

## Shell

Standard Linux commands available: `ls`, `cat`, `cp`, `mv`, `rm`, `mkdir`,
`find`, `grep`, `sed`, `awk`, `sort`, `head`, `tail`, `wc`, `tar`, `zip`,
`diff`, `cut`, `tr`, `base64`, `date`, `seq`.

## Environment

- **CWD:** `/workspace`
- **Dirs:** `/workspace`, `/data`, `/var`, `/etc`, `/tmp`
- **No network:** `curl`/`wget` are blocked — use the provided tools
  (web search, weather, place search) instead of direct HTTP requests.
- **File types:** `file <path>` — inspect file type
- **Find commands:** `which <cmd>` — locate any available command
