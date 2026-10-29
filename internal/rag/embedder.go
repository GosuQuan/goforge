package rag

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// Embedder turns text into fixed-length vectors. Implementations must return
// one vector per input, in order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Name() string
}

// HashingEmbedder is a dependency-free, deterministic embedder using the
// "feature hashing" trick: tokens are hashed into a fixed number of buckets
// with a signed weight. It captures lexical similarity (great for code
// identifiers) with zero network calls, so indexing and search work offline.
//
// It is not a semantic model, but it is a legitimate, well-known technique and
// makes the whole RAG loop runnable in CI. Swap in OpenAIEmbedder for semantics.
type HashingEmbedder struct {
	dim int
}

// NewHashingEmbedder returns an embedder with the given dimension (default 512).
func NewHashingEmbedder(dim int) *HashingEmbedder {
	if dim <= 0 {
		dim = 512
	}
	return &HashingEmbedder{dim: dim}
}

// Name implements Embedder.
func (h *HashingEmbedder) Name() string { return "hashing" }

// Embed implements Embedder.
func (h *HashingEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.embedOne(t)
	}
	return out, nil
}

func (h *HashingEmbedder) embedOne(text string) []float32 {
	vec := make([]float32, h.dim)
	tokens := Tokenize(text)
	for _, tok := range tokens {
		hi := hash64(tok, 0)
		idx := int(hi % uint64(h.dim))
		sign := float32(1)
		if (hi>>63)&1 == 1 {
			sign = -1
		}
		// Sub-linear term frequency damping.
		vec[idx] += sign
		// A second bucket per token reduces collisions.
		idx2 := int(hash64(tok, 0x9e3779b97f4a7c15) % uint64(h.dim))
		vec[idx2] += sign * 0.5
	}
	// L2 normalise so cosine == dot product.
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if norm > 0 {
		inv := float32(1 / math.Sqrt(norm))
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec
}

// Tokenize splits code/text into lowercase tokens, splitting camelCase and
// snake_case so that `parseHTTPHeader` and `parse_http_header` overlap.
func Tokenize(s string) []string {
	var tokens []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			tokens = append(tokens, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
	}
	runes := []rune(s)
	for i, r := range runes {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			if unicode.IsUpper(r) && len(cur) > 0 {
				prev := cur[len(cur)-1]
				nextLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
				// Split before an Upper that follows a lower (camelCase) or that
				// ends an acronym run followed by a lowercase (HTTPHeader -> HTTP, Header).
				if unicode.IsLower(prev) || (unicode.IsUpper(prev) && nextLower) {
					flush()
				}
			}
			cur = append(cur, r)
		case r == '_' || r == '-' || r == '.':
			flush()
		default:
			flush()
		}
	}
	flush()
	// drop very short noise tokens except common code meaning
	filtered := tokens[:0]
	for _, t := range tokens {
		if len([]rune(t)) >= 2 {
			filtered = append(filtered, t)
		}
	}
	return filtered
}

func hash64(s string, seed uint64) uint64 {
	h := fnv.New64a()
	var b [8]byte
	for i := 0; i < 8; i++ {
		b[i] = byte(seed >> (8 * i))
	}
	_, _ = h.Write(b[:])
	_, _ = h.Write([]byte(s))
	return h.Sum64()
}

// Cosine returns the cosine similarity of two equal-length vectors.
func Cosine(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
