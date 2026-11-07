# syntax=docker/dockerfile:1
# Build context: the gateway/ directory (docker build -f gateway/Dockerfile gateway/).

FROM golang:1.24-alpine AS build
WORKDIR /src
ENV GOTOOLCHAIN=local CGO_ENABLED=0 GOFLAGS=-mod=readonly
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
RUN go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/sceneproof-gateway ./cmd/sceneproof-gateway

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/sceneproof-gateway /sceneproof-gateway
# Queue/ledger and the evidence directory shared with the analyzer live under /data.
ENV SP_GW_QUEUE_DIR=/data/gateway \
    SP_GW_EVIDENCE_ROOT=/data/evidence \
    SP_GW_LISTEN=:50051 \
    SP_GW_HTTP=:9100
VOLUME ["/data"]
EXPOSE 50051 9100
USER nonroot:nonroot
ENTRYPOINT ["/sceneproof-gateway"]
