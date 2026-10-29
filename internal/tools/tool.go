// Package tools implements the coding tools the agent can call. Every file
// tool is rooted at a workspace directory and refuses to escape it, so an agent
// (or a bad model output) cannot read/write outside the project.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/mima/goforge/internal/llm"
	"github.com/mima/goforge/internal/rag"
)

// Tool is a callable capability exposed to the model.
type Tool interface {
	Name() string
	Description() string
	Schema() json.RawMessage
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}

// Registry holds tools in registration order and exposes their specs.
type Registry struct {
	byName map[string]Tool
	order  []string
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{byName: map[string]Tool{}}
}

// Register adds a tool (later registrations with the same name replace).
func (r *Registry) Register(t Tool) *Registry {
	if _, ok := r.byName[t.Name()]; !ok {
		r.order = append(r.order, t.Name())
	}
	r.byName[t.Name()] = t
	return r
}

// Get returns a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	t, ok := r.byName[name]
	return t, ok
}

// Names returns registered tool names in order.
func (r *Registry) Names() []string { return append([]string{}, r.order...) }

// Specs converts tools to LLM tool specs.
func (r *Registry) Specs() []llm.ToolSpec {
	specs := make([]llm.ToolSpec, 0, len(r.order))
	for _, n := range r.order {
		t := r.byName[n]
		specs = append(specs, llm.ToolSpec{Name: t.Name(), Description: t.Description(), Parameters: t.Schema()})
	}
	return specs
}

// Execute runs a tool call. Tool errors are returned to the caller, which is
// expected to turn them into an observation message for the model.
func (r *Registry) Execute(ctx context.Context, call llm.ToolCall) (string, error) {
	t, ok := r.byName[call.Name]
	if !ok {
		return "", fmt.Errorf("unknown tool %q", call.Name)
	}
	return t.Execute(ctx, call.Arguments)
}

// Options configures the default tool set.
type Options struct {
	Root           string
	ShellTimeout   time.Duration
	MaxOutputBytes int
	Retriever      *rag.Retriever
}

func (o Options) root() string {
	if o.Root == "" {
		wd, _ := os.Getwd()
		return wd
	}
	abs, err := filepath.Abs(o.Root)
	if err != nil {
		return o.Root
	}
	return abs
}

func (o Options) maxOut() int {
	if o.MaxOutputBytes <= 0 {
		return 64 * 1024
	}
	return o.MaxOutputBytes
}

// resolve turns a possibly-relative path into an absolute path inside root and
// rejects anything that escapes the workspace.
func (o Options) resolve(p string) (string, error) {
	if strings.TrimSpace(p) == "" {
		return "", fmt.Errorf("path is required")
	}
	root := o.root()
	abs := p
	if !filepath.IsAbs(p) {
		abs = filepath.Join(root, p)
	}
	abs = filepath.Clean(abs)
	if abs != root && !strings.HasPrefix(abs, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("path %q escapes workspace root %q", p, root)
	}
	return abs, nil
}

func (o Options) rel(abs string) string {
	rel, err := filepath.Rel(o.root(), abs)
	if err != nil {
		return abs
	}
	return filepath.ToSlash(rel)
}

func decodeArgs(raw json.RawMessage, v any) error {
	if len(raw) == 0 {
		return fmt.Errorf("missing arguments")
	}
	dec := json.NewDecoder(strings.NewReader(string(raw)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("invalid arguments: %w", err)
	}
	return nil
}

func truncate(s string, max int) string {
	if max > 0 && len(s) > max {
		return s[:max] + fmt.Sprintf("\n... (truncated, %d bytes total)", len(s))
	}
	return s
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// DefaultRegistry wires the standard coding tool set.
func DefaultRegistry(opts Options) *Registry {
	r := NewRegistry()
	r.Register(&ReadFile{opts: opts})
	r.Register(&WriteFile{opts: opts})
	r.Register(&EditFile{opts: opts})
	r.Register(&ListDir{opts: opts})
	r.Register(&Glob{opts: opts})
	r.Register(&Grep{opts: opts})
	r.Register(&Shell{opts: opts})
	if opts.Retriever != nil {
		r.Register(&CodeSearch{opts: opts, retriever: opts.Retriever})
	}
	return r
}
