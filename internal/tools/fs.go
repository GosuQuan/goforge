package tools

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// ---------- read_file ----------

type ReadFile struct{ opts Options }

func (t *ReadFile) Name() string { return "read_file" }
func (t *ReadFile) Description() string {
	return "Read a text file from the workspace. Optionally pass start_line/end_line (1-based, inclusive) to read a range. Returns the file content."
}
func (t *ReadFile) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"Path relative to workspace root"},"start_line":{"type":"integer","description":"1-based first line"},"end_line":{"type":"integer","description":"1-based last line (inclusive)"}},"required":["path"]}`)
}

func (t *ReadFile) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Path      string `json:"path"`
		StartLine int    `json:"start_line"`
		EndLine   int    `json:"end_line"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	abs, err := t.opts.resolve(a.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	if info.IsDir() {
		return "", fmt.Errorf("read_file: %s is a directory (use list_dir)", a.Path)
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	start, end := 1, len(lines)
	if a.StartLine > 0 {
		start = a.StartLine
	}
	if a.EndLine > 0 {
		end = a.EndLine
	}
	if start < 1 {
		start = 1
	}
	if end > len(lines) {
		end = len(lines)
	}
	if start > end {
		return "", fmt.Errorf("read_file: invalid range %d..%d", start, end)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# %s (lines %d-%d of %d)\n", t.opts.rel(abs), start, end, len(lines))
	b.WriteString(strings.Join(lines[start-1:end], "\n"))
	return truncate(b.String(), t.opts.maxOut()), nil
}

// ---------- write_file ----------

type WriteFile struct{ opts Options }

func (t *WriteFile) Name() string { return "write_file" }
func (t *WriteFile) Description() string {
	return "Create or overwrite a text file with the given content. Parent directories are created. Use edit_file for targeted changes to existing files."
}
func (t *WriteFile) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`)
}

func (t *WriteFile) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	abs, err := t.opts.resolve(a.Path)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return "", fmt.Errorf("write_file: mkdir: %w", err)
	}
	if err := os.WriteFile(abs, []byte(a.Content), 0o644); err != nil {
		return "", fmt.Errorf("write_file: %w", err)
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(a.Content), t.opts.rel(abs)), nil
}

// ---------- edit_file ----------

type EditFile struct{ opts Options }

func (t *EditFile) Name() string { return "edit_file" }
func (t *EditFile) Description() string {
	return "Replace an exact string in an existing file. old_string must appear exactly once unless replace_all is true. Prefer this over write_file for small changes."
}
func (t *EditFile) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"},"replace_all":{"type":"boolean"}},"required":["path","old_string","new_string"]}`)
}

func (t *EditFile) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Path       string `json:"path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	if a.OldString == "" {
		return "", fmt.Errorf("edit_file: old_string must not be empty")
	}
	abs, err := t.opts.resolve(a.Path)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", fmt.Errorf("edit_file: %w", err)
	}
	content := string(data)
	count := strings.Count(content, a.OldString)
	if count == 0 {
		return "", fmt.Errorf("edit_file: old_string not found in %s", a.Path)
	}
	if count > 1 && !a.ReplaceAll {
		return "", fmt.Errorf("edit_file: old_string appears %d times in %s; pass replace_all=true or a more specific anchor", count, a.Path)
	}
	var updated string
	if a.ReplaceAll {
		updated = strings.ReplaceAll(content, a.OldString, a.NewString)
	} else {
		updated = strings.Replace(content, a.OldString, a.NewString, 1)
	}
	if err := os.WriteFile(abs, []byte(updated), 0o644); err != nil {
		return "", fmt.Errorf("edit_file: write: %w", err)
	}
	return fmt.Sprintf("edited %s (%d replacement(s))", t.opts.rel(abs), count), nil
}

// ---------- list_dir ----------

type ListDir struct{ opts Options }

func (t *ListDir) Name() string { return "list_dir" }
func (t *ListDir) Description() string {
	return "List the immediate children of a directory inside the workspace."
}
func (t *ListDir) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"Directory, default ."}}}`)
}

func (t *ListDir) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Path string `json:"path"`
	}
	if len(raw) > 0 {
		if err := decodeArgs(raw, &a); err != nil {
			return "", err
		}
	}
	if a.Path == "" {
		a.Path = "."
	}
	abs, err := t.opts.resolve(a.Path)
	if err != nil {
		return "", err
	}
	entries, err := os.ReadDir(abs)
	if err != nil {
		return "", fmt.Errorf("list_dir: %w", err)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "entries in %s:\n", t.opts.rel(abs))
	for _, e := range entries {
		suffix := ""
		if e.IsDir() {
			suffix = "/"
		}
		size := int64(0)
		if info, ierr := e.Info(); ierr == nil {
			size = info.Size()
		}
		fmt.Fprintf(&b, "%8d  %s%s\n", size, e.Name(), suffix)
	}
	return truncate(b.String(), t.opts.maxOut()), nil
}

// ---------- glob ----------

type Glob struct{ opts Options }

func (t *Glob) Name() string { return "glob" }
func (t *Glob) Description() string {
	return "Find files by glob pattern relative to the workspace root (e.g. **/*.go, pkg/**/*.ts). ** matches any number of directories."
}
func (t *Glob) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string"},"max_results":{"type":"integer"}},"required":["pattern"]}`)
}

func (t *Glob) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Pattern    string `json:"pattern"`
		MaxResults int    `json:"max_results"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	max := a.MaxResults
	if max <= 0 || max > 500 {
		max = 200
	}
	re, err := globToRegexp(a.Pattern)
	if err != nil {
		return "", err
	}
	var matches []string
	root := t.opts.root()
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			if d != nil && d.IsDir() && path != root && isSkippedDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		rel := filepath.ToSlash(t.opts.rel(path))
		if re.MatchString(rel) {
			matches = append(matches, rel)
			if len(matches) >= max {
				return fmt.Errorf("stop")
			}
		}
		return nil
	})
	if len(matches) == 0 {
		return fmt.Sprintf("no files match %q", a.Pattern), nil
	}
	return truncate(strings.Join(matches, "\n"), t.opts.maxOut()), nil
}

func globToRegexp(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		c := pattern[i]
		switch c {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				b.WriteString(".*")
				i++
				if i+1 < len(pattern) && pattern[i+1] == '/' {
					i++
					b.WriteString("")
				}
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		case '.', '+', '(', ')', '|', '^', '$', '[', ']', '{', '}', '\\':
			b.WriteByte('\\')
			b.WriteByte(c)
		default:
			b.WriteByte(c)
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}

// ---------- grep ----------

type Grep struct{ opts Options }

func (t *Grep) Name() string { return "grep" }
func (t *Grep) Description() string {
	return "Search file contents with a Go regular expression. Optionally restrict by subdirectory (path) and file glob (include, e.g. *.go)."
}
func (t *Grep) Schema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"include":{"type":"string"},"max_results":{"type":"integer"}},"required":["pattern"]}`)
}

func (t *Grep) Execute(_ context.Context, raw json.RawMessage) (string, error) {
	var a struct {
		Pattern    string `json:"pattern"`
		Path       string `json:"path"`
		Include    string `json:"include"`
		MaxResults int    `json:"max_results"`
	}
	if err := decodeArgs(raw, &a); err != nil {
		return "", err
	}
	re, err := regexp.Compile(a.Pattern)
	if err != nil {
		return "", fmt.Errorf("grep: bad pattern: %w", err)
	}
	base := t.opts.root()
	if a.Path != "" {
		base, err = t.opts.resolve(a.Path)
		if err != nil {
			return "", err
		}
	}
	max := a.MaxResults
	if max <= 0 || max > 500 {
		max = 100
	}
	var b strings.Builder
	count := 0
	_ = filepath.WalkDir(base, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if path != base && isSkippedDir(d.Name()) {
				return filepath.SkipDir
			}
			return nil
		}
		if a.Include != "" {
			if ok, _ := filepath.Match(a.Include, filepath.Base(path)); !ok {
				return nil
			}
		}
		info, ierr := d.Info()
		if ierr != nil || info.Size() > 2<<20 {
			return nil
		}
		f, oerr := os.Open(path)
		if oerr != nil {
			return nil
		}
		defer f.Close()
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
		lineNo := 0
		for sc.Scan() {
			lineNo++
			line := sc.Text()
			if re.MatchString(line) {
				fmt.Fprintf(&b, "%s:%d: %s\n", t.opts.rel(path), lineNo, strings.TrimSpace(line))
				count++
				if count >= max {
					return fmt.Errorf("stop")
				}
			}
		}
		return nil
	})
	if count == 0 {
		return fmt.Sprintf("no matches for %q", a.Pattern), nil
	}
	return truncate(b.String(), t.opts.maxOut()), nil
}

func isSkippedDir(name string) bool {
	switch name {
	case ".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build", "target", ".venv", "venv", "__pycache__", ".idea", ".vscode", ".tools", ".refs", ".goforge":
		return true
	}
	return false
}
