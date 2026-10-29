package graph

import (
	"reflect"
)

// State is the shared data structure that flows through the graph. Nodes receive
// a snapshot and return a partial update; the engine merges the update back into
// the state using per-key reducers (see Graph.SetReducer).
//
// The type intentionally mirrors LangGraph's "State" concept: a plain map with
// merge semantics, so it stays serialisable (JSON) for checkpoints.
type State map[string]any

// Clone returns a shallow copy of the state. Values themselves are shared; the
// engine never mutates nested values, it only replaces top-level keys.
func (s State) Clone() State {
	if s == nil {
		return State{}
	}
	out := make(State, len(s))
	for k, v := range s {
		out[k] = v
	}
	return out
}

// GetString returns a string value or "".
func (s State) GetString(key string) string {
	if v, ok := s[key]; ok {
		if str, ok := v.(string); ok {
			return str
		}
	}
	return ""
}

// GetInt returns an int value or a zero value when missing / wrong type.
func (s State) GetInt(key string) int {
	if v, ok := s[key]; ok {
		switch n := v.(type) {
		case int:
			return n
		case int64:
			return int(n)
		case float64:
			return int(n)
		}
	}
	return 0
}

// GetBool returns a bool value or false.
func (s State) GetBool(key string) bool {
	v, _ := s[key].(bool)
	return v
}

// Get returns a raw value and whether the key exists.
func (s State) Get(key string) (any, bool) {
	v, ok := s[key]
	return v, ok
}

// Reducer merges an update for a single key into the current value.
type Reducer func(current, update any) any

// AppendReducer concatenates slices. It is used for the conversation history so
// that every node can append messages without clobbering earlier ones.
func AppendReducer(current, update any) any {
	if current == nil {
		return update
	}
	if update == nil {
		return current
	}
	cv := reflect.ValueOf(current)
	uv := reflect.ValueOf(update)
	if cv.Kind() != reflect.Slice || uv.Kind() != reflect.Slice {
		return update
	}
	if cv.Type() != uv.Type() {
		return update
	}
	out := reflect.MakeSlice(cv.Type(), 0, cv.Len()+uv.Len())
	out = reflect.AppendSlice(out, cv)
	out = reflect.AppendSlice(out, uv)
	return out.Interface()
}
