GO   ?= go
BIN  := bin/goforge
PKG  := ./...

.PHONY: all build test race cover vet fmt tidy index demo graph clean

all: build

build:
	@mkdir -p bin
	$(GO) build -trimpath -o $(BIN) ./cmd/goforge

test:
	$(GO) test $(PKG)

race:
	$(GO) test -race $(PKG)

cover:
	$(GO) test -coverprofile=coverage.out $(PKG)
	@$(GO) tool cover -func=coverage.out | tail -1

vet:
	$(GO) vet $(PKG)

fmt:
	gofmt -w .

tidy:
	$(GO) mod tidy

index: build
	./$(BIN) index --dir testdata/sample-repo

demo: build
	./$(BIN) index --dir testdata/sample-repo
	./$(BIN) run --mock --verbose --dir testdata/sample-repo "Where is the Add function defined and what does it do?"

graph: build
	./$(BIN) graph --dir testdata/sample-repo

clean:
	rm -rf bin coverage.out testdata/sample-repo/.goforge
