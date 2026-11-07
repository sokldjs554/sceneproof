# syntax=docker/dockerfile:1
# Build context: the gateway/ directory (docker build -f gateway/Dockerfile gateway/).

FROM golang:1.24-alpine AS build
WORKDIR /src
ENV GOTOOLCHAIN=local CGO_ENABLED=0 GOFLAGS=-mod=readonly
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
RUN go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/zonewatch-gateway ./cmd/zonewatch-gateway

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/zonewatch-gateway /zonewatch-gateway
# Queue/ledger and the evidence directory shared with the analyzer live under /data.
ENV ZW_GW_QUEUE_DIR=/data/gateway \
    ZW_GW_EVIDENCE_ROOT=/data/evidence \
    ZW_GW_LISTEN=:50051 \
    ZW_GW_HTTP=:9100
VOLUME ["/data"]
EXPOSE 50051 9100
USER nonroot:nonroot
ENTRYPOINT ["/zonewatch-gateway"]
