package graph

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestLinearGraph(t *testing.T) {
	g := New()
	var order []string
	mk := func(name string) NodeFunc {
		return func(_ context.Context, s State) (State, error) {
			order = append(order, name)
			return State{"last": name}, nil
		}
	}
	g.AddNode("a", mk("a"))
	g.AddNode("b", mk("b"))
	g.AddNode("c", mk("c"))
	g.SetEntry("a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "c")
	g.AddEdge("c", END)
	r, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	st, err := r.Invoke(context.Background(), State{})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(order, ",") != "a,b,c" {
		t.Fatalf("order = %v", order)
	}
	if st.GetString("last") != "c" {
		t.Fatalf("last = %q", st.GetString("last"))
	}
}

func TestConditionalEdge(t *testing.T) {
	g := New()
	g.AddNode("start", func(_ context.Context, s State) (State, error) {
		return State{"seen": "start"}, nil
	})
	g.AddNode("left", func(_ context.Context, s State) (State, error) { return State{"result": "left"}, nil })
	g.AddNode("right", func(_ context.Context, s State) (State, error) { return State{"result": "right"}, nil })
	g.SetEntry("start")
	g.AddConditionalEdge("start", func(_ context.Context, s State) (string, error) {
		if s.GetBool("goRight") {
			return "right", nil
		}
		return "left", nil
	})
	g.AddEdge("left", END)
	g.AddEdge("right", END)
	r, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	st, _ := r.Invoke(context.Background(), State{"goRight": true})
	if st.GetString("result") != "right" {
		t.Fatalf("expected right, got %q", st.GetString("result"))
	}
}

func TestCycleRespectsRouter(t *testing.T) {
	g := New()
	g.AddNode("loop", func(_ context.Context, s State) (State, error) {
		return State{"n": s.GetInt("n") + 1}, nil
	})
	g.SetEntry("loop")
	g.AddConditionalEdge("loop", func(_ context.Context, s State) (string, error) {
		if s.GetInt("n") >= 5 {
			return END, nil
		}
		return "loop", nil
	})
	r, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	st, err := r.Invoke(context.Background(), State{"n": 0})
	if err != nil {
		t.Fatal(err)
	}
	if st.GetInt("n") != 5 {
		t.Fatalf("n = %d, want 5", st.GetInt("n"))
	}
}

func TestMaxStepsGuard(t *testing.T) {
	g := New().SetMaxSteps(7)
	g.AddNode("spin", func(_ context.Context, s State) (State, error) {
		return State{"n": s.GetInt("n") + 1}, nil
	})
	g.SetEntry("spin")
	g.AddEdge("spin", "spin")
	r, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	_, err = r.Invoke(context.Background(), State{"n": 0})
	var mse *MaxStepsError
	if !errors.As(err, &mse) {
		t.Fatalf("expected MaxStepsError, got %v", err)
	}
}

func TestAppendReducer(t *testing.T) {
	g := New()
	g.SetReducer("items", AppendReducer)
	g.AddNode("a", func(_ context.Context, s State) (State, error) {
		return State{"items": []string{"a"}}, nil
	})
	g.AddNode("b", func(_ context.Context, s State) (State, error) {
		return State{"items": []string{"b"}}, nil
	})
	g.SetEntry("a")
	g.AddEdge("a", "b")
	g.AddEdge("b", END)
	r, _ := g.Compile()
	st, err := r.Invoke(context.Background(), State{})
	if err != nil {
		t.Fatal(err)
	}
	items, _ := st["items"].([]string)
	if len(items) != 2 || items[0] != "a" || items[1] != "b" {
		t.Fatalf("items = %v", items)
	}
}

func TestInterruptAndResume(t *testing.T) {
	dir := t.TempDir()
	cp, err := NewFileCheckpointer(dir)
	if err != nil {
		t.Fatal(err)
	}
	g := New().WithCheckpointer(cp)
	g.AddNode("gate", func(_ context.Context, s State) (State, error) {
		if !s.GetBool("approved") {
			return nil, &Interrupt{Prompt: "need approval", Data: map[string]any{"op": "write"}}
		}
		return State{"done": true}, nil
	})
	g.SetEntry("gate")
	g.AddEdge("gate", END)
	r, err := g.Compile()
	if err != nil {
		t.Fatal(err)
	}
	runID := "test-run"
	_, err = r.InvokeWith(context.Background(), RunConfig{RunID: runID, Initial: State{"approved": false}})
	var intr *Interrupt
	if !errors.As(err, &intr) {
		t.Fatalf("expected interrupt, got %v", err)
	}
	// Resume with approval.
	st, err := r.InvokeWith(context.Background(), RunConfig{
		RunID: runID, Resume: true, Initial: State{"approved": true},
	})
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if !st.GetBool("done") {
		t.Fatalf("expected done after resume, state=%v", st)
	}
}

func TestMermaid(t *testing.T) {
	g := New().Named("demo")
	g.AddNode("a", func(context.Context, State) (State, error) { return nil, nil })
	g.SetEntry("a")
	g.AddEdge("a", END)
	if _, err := g.Compile(); err != nil {
		t.Fatal(err)
	}
	m := g.Mermaid()
	if !strings.Contains(m, "flowchart TD") || !strings.Contains(m, "a -->") {
		t.Fatalf("unexpected mermaid:\n%s", m)
	}
}

func TestCompileRejectsMissingEntry(t *testing.T) {
	g := New()
	g.AddNode("a", func(context.Context, State) (State, error) { return nil, nil })
	if _, err := g.Compile(); err == nil {
		t.Fatal("expected error for missing entry")
	}
}
