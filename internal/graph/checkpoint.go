package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Checkpoint is a durable snapshot of a run. Node is the node that will execute
// next when the run is resumed.
type Checkpoint struct {
	RunID     string    `json:"run_id"`
	Step      int       `json:"step"`
	Node      string    `json:"node"`
	State     State     `json:"state"`
	Status    string    `json:"status"` // running | interrupted | done | failed
	UpdatedAt time.Time `json:"updated_at"`
}

// Checkpointer persists checkpoints. Implementations must be safe for use by a
// single run; concurrent runs use distinct RunIDs.
type Checkpointer interface {
	Save(ctx context.Context, cp Checkpoint) error
	Load(ctx context.Context, runID string) (Checkpoint, bool, error)
	Delete(ctx context.Context, runID string) error
}

// MemoryCheckpointer keeps checkpoints in memory. Handy for tests and for the
// "resume within the same process" demo.
type MemoryCheckpointer struct {
	mu sync.Mutex
	m  map[string]Checkpoint
}

// NewMemoryCheckpointer returns an empty in-memory checkpointer.
func NewMemoryCheckpointer() *MemoryCheckpointer {
	return &MemoryCheckpointer{m: map[string]Checkpoint{}}
}

// Save implements Checkpointer.
func (c *MemoryCheckpointer) Save(_ context.Context, cp Checkpoint) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	cp.UpdatedAt = time.Now().UTC()
	c.m[cp.RunID] = cp
	return nil
}

// Load implements Checkpointer.
func (c *MemoryCheckpointer) Load(_ context.Context, runID string) (Checkpoint, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	cp, ok := c.m[runID]
	return cp, ok, nil
}

// Delete implements Checkpointer.
func (c *MemoryCheckpointer) Delete(_ context.Context, runID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.m, runID)
	return nil
}

// FileCheckpointer stores one JSON file per run under Dir. This is the durable
// option used by the CLI so a crashed or interrupted run can be resumed later.
type FileCheckpointer struct {
	Dir string
}

// NewFileCheckpointer creates Dir if needed.
func NewFileCheckpointer(dir string) (*FileCheckpointer, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("checkpoint: mkdir %s: %w", dir, err)
	}
	return &FileCheckpointer{Dir: dir}, nil
}

func (c *FileCheckpointer) path(runID string) string {
	return filepath.Join(c.Dir, runID+".json")
}

// Save implements Checkpointer (atomic write via temp + rename).
func (c *FileCheckpointer) Save(_ context.Context, cp Checkpoint) error {
	cp.UpdatedAt = time.Now().UTC()
	data, err := json.MarshalIndent(cp, "", "  ")
	if err != nil {
		return fmt.Errorf("checkpoint: marshal: %w", err)
	}
	tmp, err := os.CreateTemp(c.Dir, ".cp-*.tmp")
	if err != nil {
		return fmt.Errorf("checkpoint: temp: %w", err)
	}
	tmpName := tmp.Name()
	if _, werr := tmp.Write(data); werr != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("checkpoint: write: %w", werr)
	}
	if cerr := tmp.Close(); cerr != nil {
		os.Remove(tmpName)
		return cerr
	}
	if rerr := os.Rename(tmpName, c.path(cp.RunID)); rerr != nil {
		os.Remove(tmpName)
		return fmt.Errorf("checkpoint: rename: %w", rerr)
	}
	return nil
}

// Load implements Checkpointer.
func (c *FileCheckpointer) Load(_ context.Context, runID string) (Checkpoint, bool, error) {
	data, err := os.ReadFile(c.path(runID))
	if err != nil {
		if os.IsNotExist(err) {
			return Checkpoint{}, false, nil
		}
		return Checkpoint{}, false, fmt.Errorf("checkpoint: read: %w", err)
	}
	var cp Checkpoint
	if err := json.Unmarshal(data, &cp); err != nil {
		return Checkpoint{}, false, fmt.Errorf("checkpoint: unmarshal: %w", err)
	}
	return cp, true, nil
}

// Delete implements Checkpointer.
func (c *FileCheckpointer) Delete(_ context.Context, runID string) error {
	err := os.Remove(c.path(runID))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Interrupt signals that a node wants to pause the graph and hand control back
// to the caller (human-in-the-loop). The executor checkpoints the run at the
// *current* node so that a later resume re-enters it.
type Interrupt struct {
	Prompt string         `json:"prompt"`
	Data   map[string]any `json:"data,omitempty"`
}

// Error implements error.
func (i *Interrupt) Error() string { return "graph: interrupted: " + i.Prompt }
