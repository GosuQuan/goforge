package app

import (
	"context"
	"errors"
	"fmt"
	"os"

	"github.com/mima/goforge/internal/config"
	"github.com/mima/goforge/internal/graph"
	"github.com/mima/goforge/internal/llm"
	"github.com/mima/goforge/internal/rag"
	"github.com/mima/goforge/internal/tools"
)

// Harness bundles all runtime pieces so the CLI (or tests) can index, run, and
// resume with one object.
type Harness struct {
	Cfg       config.Config
	Client    llm.Client
	Embedder  rag.Embedder
	Store     rag.VectorStore
	Retriever *rag.Retriever
	Registry  *tools.Registry
	CP        graph.Checkpointer
	Graph     *graph.CompiledGraph
}

// New wires the harness from config + a chat client.
func New(cfg config.Config, client llm.Client) (*Harness, error) {
	if client == nil {
		return nil, fmt.Errorf("app: nil llm client")
	}
	embedder := BuildEmbedder(cfg)
	store, err := BuildStore(cfg)
	if err != nil {
		return nil, err
	}
	retriever := rag.NewRetriever(embedder, store)
	reg := tools.DefaultRegistry(tools.Options{
		Root:         cfg.WorkDir,
		ShellTimeout: cfg.ShellTimeout(),
		Retriever:    retriever,
	})
	cp, err := newStateCheckpointer(cfg.CheckpointDir)
	if err != nil {
		return nil, err
	}
	g, err := buildCodingGraph(client, reg, retriever, cp, cfg)
	if err != nil {
		return nil, err
	}
	return &Harness{
		Cfg:       cfg,
		Client:    client,
		Embedder:  embedder,
		Store:     store,
		Retriever: retriever,
		Registry:  reg,
		CP:        cp,
		Graph:     g,
	}, nil
}

// BuildEmbedder picks the embedding backend.
func BuildEmbedder(cfg config.Config) rag.Embedder {
	if cfg.Embedder == "openai" {
		return rag.NewOpenAIEmbedder(cfg.BaseURL, cfg.APIKey, cfg.EmbedModel)
	}
	return rag.NewHashingEmbedder(512)
}

// BuildStore picks the vector store backend.
func BuildStore(cfg config.Config) (rag.VectorStore, error) {
	switch cfg.VectorStore {
	case "chroma":
		return rag.NewChromaStore(cfg.ChromaDir, "code")
	default:
		s := rag.NewMemoryStore()
		if cfg.IndexFile != "" {
			if _, err := os.Stat(cfg.IndexFile); err == nil {
				if err := s.Load(cfg.IndexFile); err != nil {
					return nil, fmt.Errorf("app: load index %s: %w", cfg.IndexFile, err)
				}
			}
		}
		return s, nil
	}
}

// BuildLLM constructs a real OpenAI-compatible client.
func BuildLLM(cfg config.Config) (llm.Client, error) {
	if !cfg.HasAPIKey() {
		return nil, fmt.Errorf("app: no API key set (set GOFORGE_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY, or use --mock)")
	}
	return llm.NewOpenAIClient(llm.OpenAIOptions{
		BaseURL:     cfg.BaseURL,
		APIKey:      cfg.APIKey,
		Model:       cfg.Model,
		Temperature: cfg.Temperature,
		MaxTokens:   cfg.MaxTokens,
	})
}

// Index builds the retrieval index over root and, for the in-memory store,
// persists it to the configured index file.
func (h *Harness) Index(ctx context.Context, root string, exts []string) (rag.IndexStats, error) {
	stats, err := rag.Index(ctx, rag.IndexOptions{
		Root:       root,
		Extensions: exts,
		Embedder:   h.Embedder,
		Store:      h.Store,
		Chunk:      rag.DefaultChunkOptions,
	})
	if err != nil {
		return stats, err
	}
	if ps, ok := h.Store.(rag.PersistentStore); ok {
		if err := ps.Save(h.Cfg.IndexFile); err != nil {
			return stats, fmt.Errorf("app: save index: %w", err)
		}
	}
	return stats, nil
}

// RunOptions controls a run/resume.
type RunOptions struct {
	Task    string
	RunID   string
	Resume  bool
	Approve bool
	OnEvent func(graph.Event)
}

// RunResult is the outcome of a run.
type RunResult struct {
	RunID       string
	Final       string
	Messages    []llm.Message
	Steps       int
	Interrupted bool
	Interrupt   *graph.Interrupt
	State       graph.State
}

// Run executes the coding graph, returning an interrupt when human approval is
// required (call again with Resume=true, Approve=true to continue).
func (h *Harness) Run(ctx context.Context, opts RunOptions) (RunResult, error) {
	runID := opts.RunID
	if runID == "" {
		runID = graph.NewRunID()
	}
	initial := graph.State{keyTask: opts.Task, keyApproved: opts.Approve}
	state, err := h.Graph.InvokeWith(ctx, graph.RunConfig{
		RunID:   runID,
		Initial: initial,
		Resume:  opts.Resume,
		OnEvent: opts.OnEvent,
	})
	res := RunResult{RunID: runID, State: state}
	if state != nil {
		res.Final = state.GetString(keyFinal)
		res.Messages = messagesOf(state)
		res.Steps = len(res.Messages)
	}
	if err != nil {
		var intr *graph.Interrupt
		if errors.As(err, &intr) {
			res.Interrupted = true
			res.Interrupt = intr
			return res, nil
		}
		return res, err
	}
	return res, nil
}

// SaveIndex persists the vector index if the store supports it.
func (h *Harness) SaveIndex() error {
	if ps, ok := h.Store.(rag.PersistentStore); ok {
		return ps.Save(h.Cfg.IndexFile)
	}
	return nil
}
