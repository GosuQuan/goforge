// Command goforge is a small, practical coding-agent harness written in Go.
//
// It combines four ideas the boss asked for, each with real code behind it:
//   - Agent:      a ReAct loop over a typed tool registry
//   - LangGraph:  a Go state-graph engine (nodes, conditional edges, cycles,
//     checkpoints, human-in-the-loop interrupts) in internal/graph
//   - RAG:        code indexing + embeddings + a vector store (built-in cosine
//     or chromem-go) in internal/rag
//   - Golang:     context cancellation, goroutine-friendly tools, a single
//     static binary, zero server dependencies
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mima/goforge/internal/app"
	"github.com/mima/goforge/internal/config"
	"github.com/mima/goforge/internal/graph"
	"github.com/mima/goforge/internal/llm"
	"github.com/mima/goforge/internal/rag"
)

const version = "0.1.0"

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `goforge %s — a Go coding-agent harness (agent + graph + RAG)

Usage:
  goforge index   [flags]              build the code index for RAG
  goforge run     [flags] "<task>"     run the coding agent on a task
  goforge ask     [flags] "<question>" answer a question about the codebase
  goforge resume  --run-id <id>        resume an interrupted run
  goforge graph   [flags]              print the agent graph as Mermaid
  goforge tools   [flags]              list available tools
  goforge version

Common flags (run/ask/resume/graph/tools):
  --config <file>        JSON config file
  --mock                 offline deterministic demo client (no API key)
  --provider <name>      openai | deepseek | ...
  --base-url <url>       OpenAI-compatible base URL
  --model <name>         model name
  --api-key <key>        API key (prefer GOFORGE_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY)
  --dir <path>           workspace directory the agent operates in (default .)
  --vector-store <s>     memory | chroma
  --embedder <e>         hashing | openai
  --max-steps <n>        max agent rounds
  --require-approval     pause before mutating tools (human-in-the-loop)
  --verbose              log graph events

Examples:
  goforge index --dir .
  goforge run --mock "Where is the Add function defined and what does it do?"
  GOFORGE_API_KEY=... goforge run --provider deepseek --model deepseek-chat "Add a Multiply function to pkg/calc and a test"
  goforge graph
`, version)
}

func run(args []string) error {
	if len(args) == 0 {
		usage()
		return nil
	}
	cmd, rest := args[0], args[1:]
	switch cmd {
	case "index":
		return cmdIndex(rest)
	case "run", "ask":
		return cmdRun(cmd, rest)
	case "resume":
		return cmdResume(rest)
	case "graph":
		return cmdGraph(rest)
	case "tools":
		return cmdTools(rest)
	case "version", "--version", "-v":
		fmt.Println("goforge", version)
		return nil
	case "help", "-h", "--help":
		usage()
		return nil
	default:
		return fmt.Errorf("unknown command %q (try `goforge help`)", cmd)
	}
}

// ---------------- common flags ----------------

type commonFlags struct {
	configPath      string
	mock            bool
	provider        string
	baseURL         string
	model           string
	apiKey          string
	dir             string
	vectorStore     string
	embedder        string
	maxSteps        int
	requireApproval bool
	verbose         bool
}

func (c *commonFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&c.configPath, "config", "", "JSON config file")
	fs.BoolVar(&c.mock, "mock", false, "offline deterministic demo client")
	fs.StringVar(&c.provider, "provider", "", "model provider")
	fs.StringVar(&c.baseURL, "base-url", "", "OpenAI-compatible base URL")
	fs.StringVar(&c.model, "model", "", "model name")
	fs.StringVar(&c.apiKey, "api-key", "", "API key")
	fs.StringVar(&c.dir, "dir", "", "workspace directory")
	fs.StringVar(&c.vectorStore, "vector-store", "", "memory|chroma")
	fs.StringVar(&c.embedder, "embedder", "", "hashing|openai")
	fs.IntVar(&c.maxSteps, "max-steps", 0, "max agent rounds")
	fs.BoolVar(&c.requireApproval, "require-approval", false, "require approval before mutating tools")
	fs.BoolVar(&c.verbose, "verbose", false, "log graph events")
}

func (c *commonFlags) load() (config.Config, error) {
	cfg, err := config.Load(c.configPath)
	if err != nil {
		return cfg, err
	}
	if c.provider != "" {
		cfg.Provider = c.provider
	}
	if c.baseURL != "" {
		cfg.BaseURL = c.baseURL
	}
	if c.model != "" {
		cfg.Model = c.model
	}
	if c.apiKey != "" {
		cfg.APIKey = c.apiKey
	}
	if c.dir != "" {
		cfg.WorkDir = c.dir
	}
	if c.vectorStore != "" {
		cfg.VectorStore = c.vectorStore
	}
	if c.embedder != "" {
		cfg.Embedder = c.embedder
	}
	if c.maxSteps > 0 {
		cfg.MaxSteps = c.maxSteps
	}
	if c.requireApproval {
		cfg.RequireApproval = true
	}
	if cfg.WorkDir == "" {
		cfg.WorkDir = "."
	}
	abs, err := filepath.Abs(cfg.WorkDir)
	if err == nil {
		cfg.WorkDir = abs
	}
	// Keep state files inside the workspace by default.
	cfg.IndexFile = under(cfg.WorkDir, cfg.IndexFile)
	cfg.ChromaDir = under(cfg.WorkDir, cfg.ChromaDir)
	cfg.CheckpointDir = under(cfg.WorkDir, cfg.CheckpointDir)
	return cfg, nil
}

func under(root, p string) string {
	if p == "" || filepath.IsAbs(p) {
		return p
	}
	return filepath.Join(root, p)
}

func (c *commonFlags) client(cfg config.Config) (llm.Client, error) {
	if c.mock {
		return llm.NewDemoCodingClient(), nil
	}
	return app.BuildLLM(cfg)
}

// ---------------- index ----------------

func cmdIndex(args []string) error {
	fs := flag.NewFlagSet("index", flag.ContinueOnError)
	var c commonFlags
	c.register(fs)
	dir := fs.String("index-dir", "", "directory to index (default workspace dir)")
	ext := fs.String("ext", "", "comma-separated extensions, e.g. .go,.py")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := c.load()
	if err != nil {
		return err
	}
	root := cfg.WorkDir
	if *dir != "" {
		root, _ = filepath.Abs(*dir)
	}
	var exts []string
	if *ext != "" {
		for _, e := range strings.Split(*ext, ",") {
			e = strings.TrimSpace(e)
			if e == "" {
				continue
			}
			if !strings.HasPrefix(e, ".") {
				e = "." + e
			}
			exts = append(exts, e)
		}
	}

	embedder := app.BuildEmbedder(cfg)
	store, err := freshStore(cfg)
	if err != nil {
		return err
	}
	start := time.Now()
	stats, err := rag.Index(context.Background(), rag.IndexOptions{
		Root:       root,
		Extensions: exts,
		Embedder:   embedder,
		Store:      store,
		Chunk:      rag.DefaultChunkOptions,
	})
	if err != nil {
		return err
	}
	if ps, ok := store.(rag.PersistentStore); ok {
		if err := ps.Save(cfg.IndexFile); err != nil {
			return fmt.Errorf("save index: %w", err)
		}
	}
	fmt.Printf("indexed %d files, %d chunks (%.1f KB) with %s embedder in %s\n",
		stats.Files, stats.Chunks, float64(stats.Bytes)/1024, stats.Embedder, time.Since(start).Round(time.Millisecond))
	if stats.SkippedF > 0 {
		fmt.Printf("skipped %d oversized/unreadable files\n", stats.SkippedF)
	}
	if _, ok := store.(rag.PersistentStore); ok {
		fmt.Printf("index saved to %s\n", cfg.IndexFile)
	} else {
		fmt.Printf("index persisted to %s\n", cfg.ChromaDir)
	}
	return nil
}

func freshStore(cfg config.Config) (rag.VectorStore, error) {
	if cfg.VectorStore == "chroma" {
		return rag.NewChromaStore(cfg.ChromaDir, "code")
	}
	return rag.NewMemoryStore(), nil
}

// ---------------- run / ask ----------------

func cmdRun(cmd string, args []string) error {
	fs := flag.NewFlagSet(cmd, flag.ContinueOnError)
	var c commonFlags
	c.register(fs)
	taskFlag := fs.String("task", "", "task text (alternative to positional args)")
	runID := fs.String("run-id", "", "explicit run id")
	autoApprove := fs.Bool("auto-approve", false, "automatically approve interrupts")
	doIndex := fs.Bool("index", false, "build the index before running")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := c.load()
	if err != nil {
		return err
	}
	task := strings.TrimSpace(*taskFlag)
	if task == "" {
		task = strings.TrimSpace(strings.Join(fs.Args(), " "))
	}
	if task == "" {
		return fmt.Errorf("%s: provide a task, e.g. `goforge %s \"...\"`", cmd, cmd)
	}

	client, err := c.client(cfg)
	if err != nil {
		return err
	}
	h, err := app.New(cfg, client)
	if err != nil {
		return err
	}
	ctx := context.Background()
	if *doIndex {
		stats, ierr := h.Index(ctx, cfg.WorkDir, nil)
		if ierr != nil {
			return fmt.Errorf("index: %w", ierr)
		}
		fmt.Printf("indexed %d files / %d chunks\n", stats.Files, stats.Chunks)
	}
	if h.Store.Count() == 0 {
		fmt.Fprintln(os.Stderr, "note: index is empty; retrieval will be weak. Run `goforge index --dir .` or pass --index.")
	}

	ev := eventLogger(c.verbose)
	res, err := h.Run(ctx, app.RunOptions{Task: task, RunID: *runID, OnEvent: ev})
	if err != nil {
		return err
	}
	for res.Interrupted {
		fmt.Printf("\n⏸  paused for approval at run %s\n", res.RunID)
		if res.Interrupt != nil {
			fmt.Printf("   reason: %s\n", res.Interrupt.Prompt)
			if tools, ok := res.Interrupt.Data["tools"].([]string); ok {
				for _, t := range tools {
					fmt.Printf("   - %s\n", t)
				}
			}
		}
		if !*autoApprove {
			fmt.Printf("\n   resume with: goforge resume --run-id %s\n", res.RunID)
			return nil
		}
		fmt.Println("   auto-approving and resuming...")
		res, err = h.Run(ctx, app.RunOptions{Task: task, RunID: res.RunID, Resume: true, Approve: true, OnEvent: ev})
		if err != nil {
			return err
		}
	}
	fmt.Println("\n=== final answer ===")
	fmt.Println(res.Final)
	fmt.Printf("\n(run %s, %d messages)\n", res.RunID, res.Steps)
	return nil
}

func cmdResume(args []string) error {
	fs := flag.NewFlagSet("resume", flag.ContinueOnError)
	var c commonFlags
	c.register(fs)
	runID := fs.String("run-id", "", "run id to resume (required)")
	task := fs.String("task", "", "task text (optional; restored from checkpoint)")
	autoApprove := fs.Bool("auto-approve", true, "approve and continue")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *runID == "" {
		return fmt.Errorf("resume: --run-id is required")
	}
	cfg, err := c.load()
	if err != nil {
		return err
	}
	client, err := c.client(cfg)
	if err != nil {
		return err
	}
	h, err := app.New(cfg, client)
	if err != nil {
		return err
	}
	res, err := h.Run(context.Background(), app.RunOptions{
		Task: *task, RunID: *runID, Resume: true, Approve: *autoApprove, OnEvent: eventLogger(c.verbose),
	})
	if err != nil {
		return err
	}
	if res.Interrupted {
		fmt.Printf("still paused: %s\n", res.Interrupt.Prompt)
		return nil
	}
	fmt.Println("=== final answer ===")
	fmt.Println(res.Final)
	return nil
}

// ---------------- graph / tools ----------------

func cmdGraph(args []string) error {
	fs := flag.NewFlagSet("graph", flag.ContinueOnError)
	var c commonFlags
	c.register(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := c.load()
	if err != nil {
		return err
	}
	h, err := app.New(cfg, llm.NewMockClient())
	if err != nil {
		return err
	}
	fmt.Println(h.Graph.Mermaid())
	return nil
}

func cmdTools(args []string) error {
	fs := flag.NewFlagSet("tools", flag.ContinueOnError)
	var c commonFlags
	c.register(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := c.load()
	if err != nil {
		return err
	}
	h, err := app.New(cfg, llm.NewMockClient())
	if err != nil {
		return err
	}
	fmt.Printf("tools (%d):\n", len(h.Registry.Names()))
	for _, n := range h.Registry.Names() {
		t, _ := h.Registry.Get(n)
		fmt.Printf("  %-12s %s\n", n, t.Description())
	}
	return nil
}

// ---------------- helpers ----------------

func eventLogger(verbose bool) func(graph.Event) {
	if !verbose {
		return nil
	}
	return func(e graph.Event) {
		switch e.Type {
		case graph.EventStart:
			fmt.Printf("▶ start at %s\n", e.Node)
		case graph.EventNode:
			fmt.Printf("✓ %-9s step %-2d → %s\n", e.Node, e.Step, e.Next)
		case graph.EventInterrupt:
			fmt.Printf("⏸ interrupt at %s\n", e.Node)
		case graph.EventEnd:
			fmt.Println("🏁 done")
		case graph.EventError:
			fmt.Printf("✗ error: %v\n", e.Err)
		}
	}
}
