package rag

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// IndexOptions configures a repository index run.
type IndexOptions struct {
	Root       string
	Extensions []string // e.g. ".go", ".py"; empty = DefaultExtensions
	SkipDirs   []string // directory names to prune
	MaxFileKB  int      // skip files larger than this (default 1024)
	Chunk      ChunkOptions
	Embedder   Embedder
	Store      VectorStore
	BatchSize  int // embedding batch size (default 32)
}

// IndexStats summarises an index run.
type IndexStats struct {
	Root      string
	Files     int
	Chunks    int
	Bytes     int64
	SkippedF  int
	Embedder  string
	StoreSize int
}

// DefaultExtensions is the set of source/doc files indexed by default.
var DefaultExtensions = []string{
	".go", ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".rs", ".rb",
	".c", ".h", ".cc", ".cpp", ".hpp", ".md", ".sql", ".sh", ".yaml", ".yml",
}

// DefaultSkipDirs keeps the index focused on source, not vendored/cache blobs.
var DefaultSkipDirs = []string{
	".git", ".hg", ".svn", "node_modules", "vendor", "dist", "build",
	"target", ".venv", "venv", "__pycache__", ".idea", ".vscode",
	".tools", ".refs", ".goforge", ".next", "coverage",
}

// Index walks Root, chunks every matching file, embeds the chunks and stores
// them. It is the offline, deterministic half of the RAG pipeline.
func Index(ctx context.Context, opts IndexOptions) (IndexStats, error) {
	if opts.Root == "" {
		return IndexStats{}, fmt.Errorf("rag: root is required")
	}
	if opts.Embedder == nil || opts.Store == nil {
		return IndexStats{}, fmt.Errorf("rag: embedder and store are required")
	}
	exts := opts.Extensions
	if len(exts) == 0 {
		exts = DefaultExtensions
	}
	extSet := make(map[string]bool, len(exts))
	for _, e := range exts {
		extSet[strings.ToLower(e)] = true
	}
	skip := opts.SkipDirs
	if len(skip) == 0 {
		skip = DefaultSkipDirs
	}
	skipSet := make(map[string]bool, len(skip))
	for _, d := range skip {
		skipSet[d] = true
	}
	maxBytes := int64(opts.MaxFileKB) * 1024
	if maxBytes <= 0 {
		maxBytes = 1024 * 1024
	}
	batch := opts.BatchSize
	if batch <= 0 {
		batch = 32
	}

	var chunks []Chunk
	stats := IndexStats{Root: opts.Root, Embedder: opts.Embedder.Name()}

	err := filepath.WalkDir(opts.Root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // skip unreadable entries
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if path != opts.Root && (skipSet[name] || strings.HasPrefix(name, ".") && name != ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !extSet[strings.ToLower(filepath.Ext(path))] {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		if info.Size() > maxBytes {
			stats.SkippedF++
			return nil
		}
		data, rerr := os.ReadFile(path)
		if rerr != nil {
			stats.SkippedF++
			return nil
		}
		rel, _ := filepath.Rel(opts.Root, path)
		if rel == "" {
			rel = path
		}
		fileChunks := Split(filepath.ToSlash(rel), data, opts.Chunk)
		if len(fileChunks) == 0 {
			return nil
		}
		stats.Files++
		stats.Bytes += info.Size()
		chunks = append(chunks, fileChunks...)
		return nil
	})
	if err != nil {
		return stats, err
	}

	// Embed in batches.
	for start := 0; start < len(chunks); start += batch {
		end := start + batch
		if end > len(chunks) {
			end = len(chunks)
		}
		texts := make([]string, 0, end-start)
		for _, c := range chunks[start:end] {
			// Prefix with path so the embedding carries file-level hints.
			texts = append(texts, c.Path+"\n"+c.Content)
		}
		vecs, verr := opts.Embedder.Embed(ctx, texts)
		if verr != nil {
			return stats, fmt.Errorf("rag: embed batch: %w", verr)
		}
		if aerr := opts.Store.Add(ctx, chunks[start:end], vecs); aerr != nil {
			return stats, fmt.Errorf("rag: store add: %w", aerr)
		}
	}
	stats.Chunks = len(chunks)
	stats.StoreSize = opts.Store.Count()
	return stats, nil
}
