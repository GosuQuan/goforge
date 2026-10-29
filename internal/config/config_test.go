package config

import "testing"

func TestProviderPreset(t *testing.T) {
	c := Default()
	c.Provider = "deepseek"
	c.ApplyProviderPreset()
	if c.BaseURL != "https://api.deepseek.com/v1" {
		t.Fatalf("base = %q", c.BaseURL)
	}
	if c.Model != "deepseek-chat" {
		t.Fatalf("model = %q", c.Model)
	}
}

func TestExplicitValuesNotOverridden(t *testing.T) {
	c := Default()
	c.Provider = "deepseek"
	c.BaseURL = "https://my-proxy.example/v1"
	c.Model = "my-model"
	c.ApplyProviderPreset()
	if c.BaseURL != "https://my-proxy.example/v1" || c.Model != "my-model" {
		t.Fatalf("preset clobbered explicit values: %+v", c)
	}
}

func TestLoadMissingFileAndEnv(t *testing.T) {
	t.Setenv("GOFORGE_MODEL", "env-model")
	c, err := Load("/definitely/not/here.json")
	if err != nil {
		t.Fatalf("missing file should not error: %v", err)
	}
	if c.Model != "env-model" {
		t.Fatalf("env not applied: %q", c.Model)
	}
	if c.MaxSteps != 24 {
		t.Fatalf("default max steps = %d", c.MaxSteps)
	}
}

func TestShellTimeout(t *testing.T) {
	c := Default()
	c.ShellTimeoutSeconds = 5
	if c.ShellTimeout().Seconds() != 5 {
		t.Fatalf("timeout = %v", c.ShellTimeout())
	}
	c.ShellTimeoutSeconds = 0
	if c.ShellTimeout().Seconds() != 60 {
		t.Fatalf("default timeout = %v", c.ShellTimeout())
	}
}
