package graph

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"
)

// EventType labels a streamed execution event.
type EventType string

const (
	EventStart     EventType = "start"
	EventNode      EventType = "node"      // a node finished and state was merged
	EventInterrupt EventType = "interrupt" // run paused for human input
	EventEnd       EventType = "end"
	EventError     EventType = "error"
)

// Event is emitted to RunConfig.OnEvent as the graph executes.
type Event struct {
	Type   EventType
	Step   int
	Node   string // the node that just ran (or the target of a transition)
	Next   string // where control flows next
	State  State
	Update State
	Err    error
}

// RunConfig controls a single invocation.
type RunConfig struct {
	RunID   string
	Initial State
	Resume  bool
	OnEvent func(Event)
}

// MaxStepsError is returned when a graph exceeds its step budget (a runaway
// cycle). It carries the state so the caller can inspect what happened.
type MaxStepsError struct {
	RunID string
	Max   int
}

func (e *MaxStepsError) Error() string {
	return fmt.Sprintf("graph: run %s exceeded max steps (%d); possible infinite cycle", e.RunID, e.Max)
}

// CompiledGraph is a validated, executable graph.
type CompiledGraph struct {
	g *Graph
}

var runSeq uint64

// NewRunID returns a sortable, unique run id.
func NewRunID() string {
	n := atomic.AddUint64(&runSeq, 1)
	return fmt.Sprintf("run-%s-%d", time.Now().UTC().Format("20060102T150405"), n)
}

// Mermaid returns the graph diagram.
func (c *CompiledGraph) Mermaid() string { return c.g.Mermaid() }

// Name returns the graph display name.
func (c *CompiledGraph) Name() string { return c.g.name }

// Invoke runs the graph from the entry node with the given initial state.
func (c *CompiledGraph) Invoke(ctx context.Context, initial State) (State, error) {
	return c.InvokeWith(ctx, RunConfig{Initial: initial})
}

// InvokeWith runs (or resumes) the graph with full control.
func (c *CompiledGraph) InvokeWith(ctx context.Context, cfg RunConfig) (State, error) {
	runID := cfg.RunID
	if runID == "" {
		runID = NewRunID()
	}
	emit := func(e Event) {
		if cfg.OnEvent != nil {
			cfg.OnEvent(e)
		}
	}

	var state State
	current := c.g.entry
	step := 0

	if cfg.Resume && c.g.checkpoint != nil {
		cp, ok, err := c.g.checkpoint.Load(ctx, runID)
		if err != nil {
			return nil, err
		}
		if ok {
			state = cp.State.Clone()
			current = cp.Node
			step = cp.Step
			// Merge caller-provided control flags (e.g. approval) over the
			// restored state so an interrupt can be resolved on resume.
			for k, v := range cfg.Initial {
				state[k] = v
			}
		}
	}
	if state == nil {
		state = cfg.Initial.Clone()
	}

	emit(Event{Type: EventStart, Step: step, Node: current, State: state.Clone()})

	save := func(node, status string) {
		if c.g.checkpoint == nil {
			return
		}
		_ = c.g.checkpoint.Save(ctx, Checkpoint{
			RunID: runID, Step: step, Node: node, State: state, Status: status,
		})
	}

	for {
		if err := ctx.Err(); err != nil {
			save(current, "failed")
			emit(Event{Type: EventError, Step: step, Node: current, Err: err})
			return state, err
		}
		if current == END {
			save(END, "done")
			emit(Event{Type: EventEnd, Step: step, Node: END, State: state.Clone()})
			return state, nil
		}
		if step >= c.g.maxSteps {
			err := &MaxStepsError{RunID: runID, Max: c.g.maxSteps}
			save(current, "failed")
			emit(Event{Type: EventError, Step: step, Node: current, Err: err})
			return state, err
		}
		node, ok := c.g.nodes[current]
		if !ok {
			err := fmt.Errorf("graph: no such node %q", current)
			save(current, "failed")
			emit(Event{Type: EventError, Step: step, Node: current, Err: err})
			return state, err
		}

		update, err := node.fn(ctx, state)
		if err != nil {
			var intr *Interrupt
			if errors.As(err, &intr) {
				// Pause at this node: checkpoint pre-update state at `current`.
				save(current, "interrupted")
				emit(Event{Type: EventInterrupt, Step: step, Node: current, State: state.Clone(), Err: err})
				return state, err
			}
			save(current, "failed")
			emit(Event{Type: EventError, Step: step, Node: current, Err: err})
			return state, err
		}

		state = c.mergeState(state, update)
		step++

		next, err := c.nextNode(ctx, current, state)
		if err != nil {
			save(current, "failed")
			emit(Event{Type: EventError, Step: step, Node: current, Err: err})
			return state, err
		}
		save(next, "running")
		emit(Event{Type: EventNode, Step: step, Node: current, Next: next, State: state.Clone(), Update: update})
		current = next
	}
}

func (c *CompiledGraph) nextNode(ctx context.Context, current string, state State) (string, error) {
	if router, ok := c.g.cond[current]; ok {
		n, err := router(ctx, state)
		if err != nil {
			return "", fmt.Errorf("graph: router at %q: %w", current, err)
		}
		if n != END {
			if _, ok := c.g.nodes[n]; !ok {
				return "", fmt.Errorf("graph: router at %q returned unknown node %q", current, n)
			}
		}
		return n, nil
	}
	if to, ok := c.g.static[current]; ok {
		return to, nil
	}
	return END, nil
}

func (c *CompiledGraph) mergeState(state, update State) State {
	if state == nil {
		state = State{}
	}
	for k, v := range update {
		if r, ok := c.g.reducers[k]; ok {
			state[k] = r(state[k], v)
		} else {
			state[k] = v
		}
	}
	return state
}
