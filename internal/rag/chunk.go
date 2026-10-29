// Package rag implements a small but real Retrieval-Augmented Generation
// pipeline for source code: file walking, line-aware chunking, pluggable
// embeddings, a vector store (built-in cosine store, or chromem-go as a real
// embedded vector DB), and a retriever that the agent can call as a tool.
package rag

import (
	"fmt"
	"path/filepath"
	"strings"
)

// Chunk is one retrievable unit of source code.
type Chunk struct {
	ID        string `json:"id"`
	Path      string `json:"path"`
	Language  string `json:"language"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	Content   string `json:"content"`
}

// Result is a scored chunk returned by a query.
type Result struct {
	Chunk Chunk   `json:"chunk"`
	Score float32 `json:"score"`
}

// ChunkOptions controls splitting.
type ChunkOptions struct {
	MaxLines int // target window size in lines
	Overlap  int // lines shared between adjacent windows
	MaxChars int // hard cap on characters per chunk
}

// DefaultChunkOptions is tuned for source code: windows big enough to keep a
// function together, small enough to stay cheap in context.
var DefaultChunkOptions = ChunkOptions{MaxLines: 60, Overlap: 12, MaxChars: 2400}

func (o ChunkOptions) normalize() ChunkOptions {
	if o.MaxLines <= 0 {
		o.MaxLines = 60
	}
	if o.Overlap < 0 || o.Overlap >= o.MaxLines {
		o.Overlap = o.MaxLines / 5
	}
	if o.MaxChars <= 0 {
		o.MaxChars = 2400
	}
	return o
}

// LanguageFor maps a file extension to a language tag.
func LanguageFor(path string) string {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".js", ".jsx":
		return "javascript"
	case ".ts", ".tsx":
		return "typescript"
	case ".java":
		return "java"
	case ".rs":
		return "rust"
	case ".rb":
		return "ruby"
	case ".c", ".h":
		return "c"
	case ".cc", ".cpp", ".hpp":
		return "cpp"
	case ".md":
		return "markdown"
	case ".sql":
		return "sql"
	case ".sh":
		return "shell"
	case ".yaml", ".yml":
		return "yaml"
	case ".json":
		return "json"
	default:
		return "text"
	}
}

// Split splits one file's content into line-aware chunks. IDs are stable:
// path:startLine-endLine. Empty / whitespace-only chunks are dropped.
func Split(path string, content []byte, opts ChunkOptions) []Chunk {
	opts = opts.normalize()
	lang := LanguageFor(path)
	lines := strings.Split(strings.ReplaceAll(string(content), "\r\n", "\n"), "\n")
	// Trim a trailing empty line produced by a final newline.
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	if len(lines) == 0 {
		return nil
	}

	var out []Chunk
	step := opts.MaxLines - opts.Overlap
	for start := 0; start < len(lines); start += step {
		end := start + opts.MaxLines
		if end > len(lines) {
			end = len(lines)
		}
		body := strings.Join(lines[start:end], "\n")
		if strings.TrimSpace(body) == "" {
			if end == len(lines) {
				break
			}
			continue
		}
		for _, sub := range splitByChars(body, opts.MaxChars) {
			c := Chunk{
				Path:      path,
				Language:  lang,
				StartLine: start + 1,
				EndLine:   end,
				Content:   sub,
			}
			c.ID = fmt.Sprintf("%s:%d-%d", path, c.StartLine, c.EndLine)
			// disambiguate char-sub-chunks
			if len(splitByChars(body, opts.MaxChars)) > 1 {
				c.ID = fmt.Sprintf("%s:%d-%d#%d", path, c.StartLine, c.EndLine, len(out))
			}
			out = append(out, c)
		}
		if end == len(lines) {
			break
		}
	}
	return out
}

// splitByChars hard-wraps long text at MaxChars, preferring newline boundaries.
func splitByChars(s string, max int) []string {
	if len(s) <= max {
		return []string{s}
	}
	var parts []string
	for len(s) > max {
		cut := strings.LastIndexByte(s[:max], '\n')
		if cut <= 0 {
			cut = max
		}
		parts = append(parts, s[:cut])
		s = s[cut:]
		s = strings.TrimPrefix(s, "\n")
	}
	if strings.TrimSpace(s) != "" {
		parts = append(parts, s)
	}
	return parts
}
