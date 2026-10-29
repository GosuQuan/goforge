// Package graph is a small, dependency-free implementation of the LangGraph
// programming model in Go: a typed-by-convention State that flows through named
// nodes, connected by static and conditional edges, with checkpointing and
// human-in-the-loop interrupts.
//
// It is deliberately close to LangGraph's mental model so that a team familiar
// with the Python/JS library can read a graph definition at a glance:
//
//	g := graph.New()
//	g.AddNode("retrieve", retrieveNode)
//	g.AddNode("agent", agentNode)
//	g.AddConditionalEdge("agent", route)   // -> "tools" or END
//	g.AddNode("tools", toolsNode)
//	g.AddEdge("tools", "agent")            // the ReAct cycle
//	g.SetEntry("retrieve")
//	runnable, _ := g.Compile()
//	runnable.Invoke(ctx, graph.State{"task": "..."})
//
// What LangGraph gives you across many runtimes, this gives you in ~400 lines of
// Go: nodes, edges, conditional routing, cycles with a step budget, reducers for
// merge semantics, JSON checkpoints, resume, and interrupts. See docs/ for the
// mapping table and the comparison with eino / langchaingo / langgraphgo.
package graph

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// END is the terminal node name.
const END = "__end__"

// NodeFunc is the unit of work. It reads the current State and returns a partial
// update that the engine merges back (respecting reducers). Returning an
// *Interrupt pauses the run; returning an error aborts it.
type NodeFunc func(ctx context.Context, state State) (State, error)

// Router chooses the next node from the current state.
type Router func(ctx context.Context, state State) (string, error)

type nodeEntry struct {
	name string
	fn   NodeFunc
}

// Graph is the mutable builder.
type Graph struct {
	name       string
	nodes      map[string]nodeEntry
	order      []string
	entry      string
	static     map[string]string
	cond       map[string]Router
	reducers   map[string]Reducer
	maxSteps   int
	checkpoint Checkpointer
}

// New creates an empty graph.
func New() *Graph {
	return &Graph{
		nodes:    map[string]nodeEntry{},
		static:   map[string]string{},
		cond:     map[string]Router{},
		reducers: map[string]Reducer{},
		maxSteps: 64,
	}
}

// Named sets a display name (used in mermaid and logs).
func (g *Graph) Named(name string) *Graph { g.name = name; return g }

// AddNode registers a node. Panics on duplicate or empty name.
func (g *Graph) AddNode(name string, fn NodeFunc) *Graph {
	if name == "" || name == END {
		panic("graph: invalid node name " + name)
	}
	if _, dup := g.nodes[name]; dup {
		panic("graph: duplicate node " + name)
	}
	if fn == nil {
		panic("graph: nil node func " + name)
	}
	g.nodes[name] = nodeEntry{name: name, fn: fn}
	g.order = append(g.order, name)
	return g
}

// AddEdge connects from -> to. "to" may be END.
func (g *Graph) AddEdge(from, to string) *Graph {
	if from == END {
		panic("graph: END cannot have outgoing edges")
	}
	g.static[from] = to
	return g
}

// AddConditionalEdge routes from a node to one of several nodes depending on
// the router's return value.
func (g *Graph) AddConditionalEdge(from string, router Router) *Graph {
	if router == nil {
		panic("graph: nil router for " + from)
	}
	g.cond[from] = router
	return g
}

// SetEntry sets the entry node.
func (g *Graph) SetEntry(name string) *Graph { g.entry = name; return g }

// SetReducer installs a merge reducer for a state key (e.g. AppendReducer for
// the message history).
func (g *Graph) SetReducer(key string, r Reducer) *Graph { g.reducers[key] = r; return g }

// SetMaxSteps bounds the number of node executions per run, guarding cycles.
func (g *Graph) SetMaxSteps(n int) *Graph {
	if n > 0 {
		g.maxSteps = n
	}
	return g
}

// WithCheckpointer attaches a checkpointer for persistence / resume.
func (g *Graph) WithCheckpointer(c Checkpointer) *Graph { g.checkpoint = c; return g }

// Compile validates the graph and returns a runnable.
func (g *Graph) Compile() (*CompiledGraph, error) {
	if g.entry == "" {
		return nil, fmt.Errorf("graph: entry node not set")
	}
	if _, ok := g.nodes[g.entry]; !ok {
		return nil, fmt.Errorf("graph: entry node %q not registered", g.entry)
	}
	for from := range g.static {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("graph: edge from unknown node %q", from)
		}
	}
	for from, to := range g.static {
		if to != END {
			if _, ok := g.nodes[to]; !ok {
				return nil, fmt.Errorf("graph: edge %s -> unknown node %q", from, to)
			}
		}
	}
	for from := range g.cond {
		if _, ok := g.nodes[from]; !ok {
			return nil, fmt.Errorf("graph: conditional edge from unknown node %q", from)
		}
	}
	// Every node must have exactly one outgoing route (static or conditional).
	for name := range g.nodes {
		_, hasStatic := g.static[name]
		_, hasCond := g.cond[name]
		if hasStatic && hasCond {
			return nil, fmt.Errorf("graph: node %q has both static and conditional edges", name)
		}
		if !hasStatic && !hasCond {
			// terminal node: implicitly goes to END
			g.static[name] = END
		}
	}
	return &CompiledGraph{g: g}, nil
}

// Mermaid renders the graph as a Mermaid flowchart for docs/PRs.
func (g *Graph) Mermaid() string {
	var b strings.Builder
	b.WriteString("flowchart TD\n")
	names := append([]string{}, g.order...)
	sort.Strings(names)
	// entry marker
	if g.entry != "" {
		b.WriteString(fmt.Sprintf("  __start__([start]) --> %s\n", g.entry))
	}
	for _, n := range g.order {
		label := n
		b.WriteString(fmt.Sprintf("  %s[%q]\n", n, label))
	}
	for _, n := range g.order {
		if to, ok := g.static[n]; ok {
			if to == END {
				b.WriteString(fmt.Sprintf("  %s --> __end__([end])\n", n))
			} else {
				b.WriteString(fmt.Sprintf("  %s --> %s\n", n, to))
			}
		}
		if _, ok := g.cond[n]; ok {
			b.WriteString(fmt.Sprintf("  %s -.->|route| __router__{{?}}\n", n))
		}
	}
	return b.String()
}
