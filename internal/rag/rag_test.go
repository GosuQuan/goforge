package rag

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTokenizeSplitsIdentifiers(t *testing.T) {
	toks := Tokenize("parseHTTPHeader foo_bar BazQux42")
	set := map[string]bool{}
	for _, tok := range toks {
		set[tok] = true
	}
	for _, want := range []string{"parse", "http", "header", "foo", "bar", "baz", "qux42"} {
		if !set[want] {
			t.Errorf("missing token %q in %v", want, toks)
		}
	}
}

func TestSplitRespectsMaxChars(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 200; i++ {
		b.WriteString("line ")
		b.WriteString(strings.Repeat("x", 60))
		b.WriteString("\n")
	}
	chunks := Split("f.go", []byte(b.String()), ChunkOptions{MaxLines: 60, Overlap: 10, MaxChars: 500})
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for _, c := range chunks {
		if len(c.Content) > 500+8 { // allow a small truncation marker margin
			t.Fatalf("chunk too large: %d chars", len(c.Content))
		}
		if c.ID == "" || c.Path != "f.go" {
			t.Fatalf("bad chunk: %+v", c)
		}
	}
}

func TestIndexAndRetrieve(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("pkg/billing/total.go", `package billing

// ComputeTotal sums the line items and applies tax.
func ComputeTotal(items []int, taxRate float64) float64 {
	var sum int
	for _, it := range items {
		sum += it
	}
	return float64(sum) * (1 + taxRate)
}
`)
	write("README.md", "# Billing\n\nThis service computes invoices.\n")

	store := NewMemoryStore()
	emb := NewHashingEmbedder(512)
	stats, err := Index(context.Background(), IndexOptions{Root: dir, Embedder: emb, Store: store})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Files != 2 || stats.Chunks == 0 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
	r := NewRetriever(emb, store)
	res, err := r.Retrieve(context.Background(), "where is the ComputeTotal function defined", 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 {
		t.Fatal("no results")
	}
	if !strings.Contains(res[0].Chunk.Path, "total.go") {
		t.Fatalf("expected total.go first, got %q (score %.3f)", res[0].Chunk.Path, res[0].Score)
	}
}

func TestCosine(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	if got := Cosine(a, b); math.Abs(float64(got)-1) > 1e-6 {
		t.Fatalf("cosine = %v", got)
	}
	c := []float32{0, 1, 0}
	if got := Cosine(a, c); math.Abs(float64(got)) > 1e-6 {
		t.Fatalf("expected 0, got %v", got)
	}
}

func TestChromaStoreQueryClampsTopK(t *testing.T) {
	dir := t.TempDir()
	store, err := NewChromaStore(dir, "code")
	if err != nil {
		t.Fatal(err)
	}
	chunks := []Chunk{
		{ID: "a.go:1-1", Path: "a.go", Language: "go", StartLine: 1, EndLine: 1, Content: "func Alpha()"},
		{ID: "b.go:1-1", Path: "b.go", Language: "go", StartLine: 1, EndLine: 1, Content: "func Beta()"},
	}
	vecs := [][]float32{{1, 0, 0}, {0, 1, 0}}
	if err := store.Add(context.Background(), chunks, vecs); err != nil {
		t.Fatal(err)
	}
	if store.Count() != 2 {
		t.Fatalf("count = %d", store.Count())
	}
	res, err := store.Query(context.Background(), []float32{1, 0, 0}, 20) // > count
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 results, got %d", len(res))
	}
	if res[0].Chunk.ID != "a.go:1-1" {
		t.Fatalf("expected a.go first, got %s", res[0].Chunk.ID)
	}
}

func TestMemoryStoreSaveLoad(t *testing.T) {
	store := NewMemoryStore()
	chunks := []Chunk{{ID: "x:1-2", Path: "x.go", Language: "go", StartLine: 1, EndLine: 2, Content: "func X()"}}
	vecs := [][]float32{{0.1, 0.2, 0.3}}
	if err := store.Add(context.Background(), chunks, vecs); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "index.json")
	if err := store.Save(path); err != nil {
		t.Fatal(err)
	}
	loaded := NewMemoryStore()
	if err := loaded.Load(path); err != nil {
		t.Fatal(err)
	}
	if loaded.Count() != 1 {
		t.Fatalf("count = %d", loaded.Count())
	}
	res, _ := loaded.Query(context.Background(), []float32{0.1, 0.2, 0.3}, 1)
	if len(res) != 1 || res[0].Chunk.ID != "x:1-2" {
		t.Fatalf("bad result: %+v", res)
	}
}
