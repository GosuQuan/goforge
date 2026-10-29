// Package config loads harness configuration from JSON plus environment
// overrides. Keeping config dependency-free (encoding/json) means the binary
// builds without a YAML library and behaves identically in CI.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config is the full harness configuration.
type Config struct {
	// Model provider (OpenAI-compatible: openai, deepseek, moonshot, vllm, ollama)
	Provider    string  `json:"provider"`
	BaseURL     string  `json:"base_url"`
	APIKey      string  `json:"api_key"`
	Model       string  `json:"model"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`

	// Retrieval
	Embedder    string `json:"embedder"`     // hashing | openai
	EmbedModel  string `json:"embed_model"`  // for openai embedder
	VectorStore string `json:"vector_store"` // memory | chroma
	IndexFile   string `json:"index_file"`   // JSON index for memory store
	ChromaDir   string `json:"chroma_dir"`   // persistent dir for chroma

	// Workspace / behaviour
	WorkDir             string `json:"work_dir"`
	MaxSteps            int    `json:"max_steps"`
	ShellTimeoutSeconds int    `json:"shell_timeout_seconds"`
	RequireApproval     bool   `json:"require_approval"`
	CheckpointDir       string `json:"checkpoint_dir"`
}

// Default returns a runnable default config.
func Default() Config {
	return Config{
		Provider:            "openai",
		BaseURL:             "https://api.openai.com/v1",
		Model:               "gpt-4o-mini",
		Temperature:         0.2,
		MaxTokens:           0,
		Embedder:            "hashing",
		EmbedModel:          "text-embedding-3-small",
		VectorStore:         "memory",
		IndexFile:           ".goforge/index.json",
		ChromaDir:           ".goforge/chroma",
		WorkDir:             ".",
		MaxSteps:            24,
		ShellTimeoutSeconds: 60,
		RequireApproval:     false,
		CheckpointDir:       ".goforge/checkpoints",
	}
}

// Load reads JSON from path (if it exists) over the defaults. A missing file is
// not an error: defaults + env still work.
func Load(path string) (Config, error) {
	c := Default()
	if path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			if !os.IsNotExist(err) {
				return c, fmt.Errorf("config: read %s: %w", path, err)
			}
		} else if err := json.Unmarshal(data, &c); err != nil {
			return c, fmt.Errorf("config: parse %s: %w", path, err)
		}
	}
	c.ApplyEnv()
	c.ApplyProviderPreset()
	return c, nil
}

// ApplyProviderPreset fills in sensible base_url/model for well-known providers
// when the user only changed --provider. It never overrides explicit values: it
// only replaces the built-in OpenAI defaults.
func (c *Config) ApplyProviderPreset() {
	baseIsDefault := c.BaseURL == "" || c.BaseURL == "https://api.openai.com/v1"
	modelIsDefault := c.Model == "" || c.Model == "gpt-4o-mini"
	switch strings.ToLower(c.Provider) {
	case "deepseek":
		if baseIsDefault {
			c.BaseURL = "https://api.deepseek.com/v1"
		}
		if modelIsDefault {
			c.Model = "deepseek-chat"
		}
	case "moonshot":
		if baseIsDefault {
			c.BaseURL = "https://api.moonshot.cn/v1"
		}
		if modelIsDefault {
			c.Model = "moonshot-v1-8k"
		}
	case "ollama":
		if baseIsDefault {
			c.BaseURL = "http://localhost:11434/v1"
		}
		if modelIsDefault {
			c.Model = "qwen2.5-coder"
		}
	}
}

// ApplyEnv overrides config from environment variables. Precedence:
// GOFORGE_* > provider-specific keys > file.
func (c *Config) ApplyEnv() {
	if v := os.Getenv("GOFORGE_PROVIDER"); v != "" {
		c.Provider = v
	}
	if v := os.Getenv("GOFORGE_BASE_URL"); v != "" {
		c.BaseURL = v
	}
	if v := os.Getenv("GOFORGE_MODEL"); v != "" {
		c.Model = v
	}
	if v := os.Getenv("GOFORGE_API_KEY"); v != "" {
		c.APIKey = v
	}
	if v := os.Getenv("OPENAI_API_KEY"); v != "" && c.APIKey == "" {
		c.APIKey = v
	}
	if v := os.Getenv("DEEPSEEK_API_KEY"); v != "" && c.APIKey == "" {
		c.APIKey = v
		if os.Getenv("GOFORGE_BASE_URL") == "" {
			c.BaseURL = "https://api.deepseek.com/v1"
		}
		if os.Getenv("GOFORGE_MODEL") == "" {
			c.Model = "deepseek-chat"
		}
	}
	if v := os.Getenv("GOFORGE_EMBEDDER"); v != "" {
		c.Embedder = v
	}
	if v := os.Getenv("GOFORGE_VECTOR_STORE"); v != "" {
		c.VectorStore = v
	}
	if v := os.Getenv("GOFORGE_WORKDIR"); v != "" {
		c.WorkDir = v
	}
	if v := os.Getenv("GOFORGE_MAX_STEPS"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			c.MaxSteps = n
		}
	}
	if v := os.Getenv("GOFORGE_CHECKPOINT_DIR"); v != "" {
		c.CheckpointDir = v
	}
	if v := os.Getenv("GOFORGE_REQUIRE_APPROVAL"); v != "" {
		c.RequireApproval = strings.EqualFold(v, "1") || strings.EqualFold(v, "true")
	}
}

// ShellTimeout returns the configured timeout as a duration.
func (c Config) ShellTimeout() time.Duration {
	if c.ShellTimeoutSeconds <= 0 {
		return 60 * time.Second
	}
	return time.Duration(c.ShellTimeoutSeconds) * time.Second
}

// HasAPIKey reports whether a real provider can be used.
func (c Config) HasAPIKey() bool { return strings.TrimSpace(c.APIKey) != "" }
