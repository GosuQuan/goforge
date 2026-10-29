package rag

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
)

// VectorStore stores chunks with their vectors and answers nearest-neighbour
// queries.
type VectorStore interface {
	Add(ctx context.Context, chunks []Chunk, vectors [][]float32) error
	Query(ctx context.Context, vector []float32, topK int) ([]Result, error)
	Count() int
	Close() error
}

// PersistentStore is implemented by stores that can save/load themselves.
type PersistentStore interface {
	Save(path string) error
	Load(path string) error
}

type memEntry struct {
	Chunk  Chunk     `json:"chunk"`
	Vector []float32 `json:"vector"`
}

// MemoryStore is a brute-force cosine vector store. For repo-sized corpora
// (thousands of chunks) this is fast enough and completely dependency-free.
type MemoryStore struct {
	mu      sync.RWMutex
	entries []memEntry
}

// NewMemoryStore returns an empty store.
func NewMemoryStore() *MemoryStore { return &MemoryStore{} }

// Add implements VectorStore.
func (s *MemoryStore) Add(_ context.Context, chunks []Chunk, vectors [][]float32) error {
	if len(chunks) != len(vectors) {
		return fmt.Errorf("rag: %d chunks but %d vectors", len(chunks), len(vectors))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, c := range chunks {
		s.entries = append(s.entries, memEntry{Chunk: c, Vector: vectors[i]})
	}
	return nil
}

// Query implements VectorStore.
func (s *MemoryStore) Query(_ context.Context, vector []float32, topK int) ([]Result, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	results := make([]Result, 0, len(s.entries))
	for _, e := range s.entries {
		results = append(results, Result{Chunk: e.Chunk, Score: Cosine(vector, e.Vector)})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if topK > 0 && len(results) > topK {
		results = results[:topK]
	}
	return results, nil
}

// Count implements VectorStore.
func (s *MemoryStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// Close implements VectorStore.
func (s *MemoryStore) Close() error { return nil }

// Save writes the index to a JSON file (atomic).
func (s *MemoryStore) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(s.entries, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Load reads a JSON index file.
func (s *MemoryStore) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var entries []memEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries = entries
	return nil
}
