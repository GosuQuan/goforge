package app

import (
	"context"
	"encoding/json"

	"github.com/mima/goforge/internal/graph"
	"github.com/mima/goforge/internal/llm"
)

// stateCheckpointer wraps a graph.Checkpointer and rehydrates the typed
// conversation history on load. A graph.State is a map[string]any, so JSON
// round-tripping turns []llm.Message into []any; without this step a resumed run
// would lose its tool calls. This is the price of a generic state map, and the
// fix is contained to one place.
type stateCheckpointer struct {
	inner graph.Checkpointer
}

func newStateCheckpointer(dir string) (*stateCheckpointer, error) {
	inner, err := graph.NewFileCheckpointer(dir)
	if err != nil {
		return nil, err
	}
	return &stateCheckpointer{inner: inner}, nil
}

func (c *stateCheckpointer) Save(ctx context.Context, cp graph.Checkpoint) error {
	return c.inner.Save(ctx, cp)
}

func (c *stateCheckpointer) Load(ctx context.Context, runID string) (graph.Checkpoint, bool, error) {
	cp, ok, err := c.inner.Load(ctx, runID)
	if err != nil || !ok {
		return cp, ok, err
	}
	rehydrateState(cp.State)
	return cp, true, nil
}

func (c *stateCheckpointer) Delete(ctx context.Context, runID string) error {
	return c.inner.Delete(ctx, runID)
}

func rehydrateState(st graph.State) {
	if st == nil {
		return
	}
	raw, ok := st[keyMessages]
	if !ok {
		return
	}
	if _, already := raw.([]llm.Message); already {
		return
	}
	b, err := json.Marshal(raw)
	if err != nil {
		return
	}
	var msgs []llm.Message
	if err := json.Unmarshal(b, &msgs); err == nil {
		st[keyMessages] = msgs
	}
}
