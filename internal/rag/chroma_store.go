package rag

import (
	"context"
	"fmt"
	"strconv"

	chromem "github.com/philippgille/chromem-go"
)

// ChromaStore is a VectorStore backed by chromem-go, a pure-Go embedded vector
// database (https://github.com/philippgille/chromem-go). It gives us real
// persistent storage with HNSW-free exact search and a well-tested API, while
// keeping the project free of a server dependency.
//
// We feed pre-computed embeddings (Add + QueryEmbedding) so the same Embedder
// works for both the built-in MemoryStore and chromem.
type ChromaStore struct {
	db   *chromem.DB
	col  *chromem.Collection
	name string
}

// NewChromaStore opens (or creates) a persistent chromem DB under dir and gets
// the named collection.
func NewChromaStore(dir, collection string) (*ChromaStore, error) {
	db, err := chromem.NewPersistentDB(dir, false)
	if err != nil {
		return nil, fmt.Errorf("rag: open chromem db: %w", err)
	}
	// Embedding func is unused because we always pass explicit embeddings; it is
	// required by the API and kept for compatibility with text-based queries.
	dummy := func(ctx context.Context, text string) ([]float32, error) {
		return nil, fmt.Errorf("rag: chroma text embedding disabled; use QueryEmbedding")
	}
	col, err := db.GetOrCreateCollection(collection, nil, dummy)
	if err != nil {
		return nil, fmt.Errorf("rag: get collection: %w", err)
	}
	return &ChromaStore{db: db, col: col, name: collection}, nil
}

// Add implements VectorStore.
func (s *ChromaStore) Add(ctx context.Context, chunks []Chunk, vectors [][]float32) error {
	if len(chunks) != len(vectors) {
		return fmt.Errorf("rag: %d chunks but %d vectors", len(chunks), len(vectors))
	}
	ids := make([]string, len(chunks))
	metas := make([]map[string]string, len(chunks))
	contents := make([]string, len(chunks))
	for i, c := range chunks {
		ids[i] = c.ID
		metas[i] = map[string]string{
			"path":       c.Path,
			"language":   c.Language,
			"start_line": strconv.Itoa(c.StartLine),
			"end_line":   strconv.Itoa(c.EndLine),
		}
		contents[i] = c.Content
	}
	return s.col.Add(ctx, ids, vectors, metas, contents)
}

// Query implements VectorStore.
func (s *ChromaStore) Query(ctx context.Context, vector []float32, topK int) ([]Result, error) {
	// chromem rejects nResults larger than the collection size; clamp first.
	if n := s.col.Count(); topK > n {
		topK = n
	}
	if topK <= 0 {
		return nil, nil
	}
	raw, err := s.col.QueryEmbedding(ctx, vector, topK, nil, nil)
	if err != nil {
		return nil, err
	}
	out := make([]Result, 0, len(raw))
	for _, r := range raw {
		start, _ := strconv.Atoi(r.Metadata["start_line"])
		end, _ := strconv.Atoi(r.Metadata["end_line"])
		out = append(out, Result{
			Score: r.Similarity,
			Chunk: Chunk{
				ID:        r.ID,
				Path:      r.Metadata["path"],
				Language:  r.Metadata["language"],
				StartLine: start,
				EndLine:   end,
				Content:   r.Content,
			},
		})
	}
	return out, nil
}

// Count implements VectorStore.
func (s *ChromaStore) Count() int { return s.col.Count() }

// Close implements VectorStore. chromem persists on write, so nothing to flush.
func (s *ChromaStore) Close() error { return nil }
