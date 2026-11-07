# ZoneWatch edge gateway (Go). Run from gateway/.
GO          ?= go
VERSION     ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
BIN         := bin/zonewatch-gateway
PKG         := ./cmd/zonewatch-gateway
LDFLAGS     := -s -w -X main.version=$(VERSION)
# Never auto-download a newer toolchain: the build must match the Dockerfile's Go 1.24.
export GOTOOLCHAIN := local
export CGO_ENABLED := 0

.PHONY: all build test vet fmt fmt-check tidy docker

all: fmt-check vet test build

build:
	$(GO) build -trimpath -ldflags '$(LDFLAGS)' -o $(BIN) $(PKG)

test:
	$(GO) test -race -count=1 ./...

vet:
	$(GO) vet ./...

fmt:
	gofmt -l -w ./cmd ./internal

fmt-check:
	@out="$$(gofmt -l ./cmd ./internal)"; if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi

tidy:
	$(GO) mod tidy

docker:
	docker build --build-arg VERSION=$(VERSION) -t zonewatch-gateway:$(VERSION) .
