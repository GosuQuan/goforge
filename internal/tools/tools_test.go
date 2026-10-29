package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func newReg(t *testing.T) (*Registry, string) {
	t.Helper()
	dir := t.TempDir()
	return DefaultRegistry(Options{Root: dir, ShellTimeout: 10 * time.Second}), dir
}

func mustExec(t *testing.T, r *Registry, name string, args map[string]any) string {
	t.Helper()
	b, _ := json.Marshal(args)
	tool, ok := r.Get(name)
	if !ok {
		t.Fatalf("tool %s not registered", name)
	}
	out, err := tool.Execute(context.Background(), b)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return out
}

func TestFileToolsRoundtrip(t *testing.T) {
	r, dir := newReg(t)
	mustExec(t, r, "write_file", map[string]any{"path": "pkg/a.txt", "content": "hello world"})
	if _, err := os.Stat(filepath.Join(dir, "pkg/a.txt")); err != nil {
		t.Fatal(err)
	}
	out := mustExec(t, r, "read_file", map[string]any{"path": "pkg/a.txt"})
	if !strings.Contains(out, "hello world") {
		t.Fatalf("read: %s", out)
	}
	mustExec(t, r, "edit_file", map[string]any{"path": "pkg/a.txt", "old_string": "world", "new_string": "gopher"})
	out = mustExec(t, r, "read_file", map[string]any{"path": "pkg/a.txt"})
	if !strings.Contains(out, "hello gopher") {
		t.Fatalf("after edit: %s", out)
	}
}

func TestEditRequiresUniqueAnchor(t *testing.T) {
	r, _ := newReg(t)
	mustExec(t, r, "write_file", map[string]any{"path": "b.txt", "content": "x x"})
	b, _ := json.Marshal(map[string]any{"path": "b.txt", "old_string": "x", "new_string": "y"})
	tool, _ := r.Get("edit_file")
	if _, err := tool.Execute(context.Background(), b); err == nil {
		t.Fatal("expected ambiguity error")
	}
}

func TestPathEscapeRejected(t *testing.T) {
	r, _ := newReg(t)
	b, _ := json.Marshal(map[string]any{"path": "../../etc/passwd"})
	tool, _ := r.Get("read_file")
	if _, err := tool.Execute(context.Background(), b); err == nil {
		t.Fatal("expected escape rejection")
	}
}

func TestListGlobGrep(t *testing.T) {
	r, _ := newReg(t)
	mustExec(t, r, "write_file", map[string]any{"path": "a/one.go", "content": "package a\n// needle here\n"})
	mustExec(t, r, "write_file", map[string]any{"path": "b/two.txt", "content": "nothing\n"})

	list := mustExec(t, r, "list_dir", map[string]any{"path": "."})
	if !strings.Contains(list, "a/") {
		t.Fatalf("list: %s", list)
	}
	glob := mustExec(t, r, "glob", map[string]any{"pattern": "**/*.go"})
	if !strings.Contains(glob, "a/one.go") {
		t.Fatalf("glob: %s", glob)
	}
	grep := mustExec(t, r, "grep", map[string]any{"pattern": "needle", "include": "*.go"})
	if !strings.Contains(grep, "one.go") {
		t.Fatalf("grep: %s", grep)
	}
}

func TestShell(t *testing.T) {
	r, _ := newReg(t)
	out := mustExec(t, r, "run_shell", map[string]any{"command": "echo hello-shell"})
	if !strings.Contains(out, "hello-shell") {
		t.Fatalf("shell: %s", out)
	}
}
