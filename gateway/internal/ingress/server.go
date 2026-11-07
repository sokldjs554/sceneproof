// Package ingress implements EdgeIngress, the gRPC service the analyzer on
// the same device talks to. It is the gateway's only entry point for events:
// an event is answered accepted=true only after its WAL record is on disk.
package ingress

import (
	"context"
	"errors"
	"log/slog"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	grpcstatus "google.golang.org/grpc/status"

	sceneproofv1 "github.com/sokldjs554/sceneproof/gateway/gen/sceneproofv1"
	"github.com/sokldjs554/sceneproof/gateway/internal/queue"
	"github.com/sokldjs554/sceneproof/gateway/internal/status"
)

// ResultSink receives CommandResults for forwarding to the server.
type ResultSink interface {
	Push(res *sceneproofv1.CommandResult)
}

// Options configures the EdgeIngress server.
type Options struct {
	Queue    *queue.Queue
	Status   *status.Registry
	Bus      *Bus
	Results  ResultSink
	DeviceID string
	Logger   *slog.Logger
}

// Server implements sceneproofv1.EdgeIngressServer.
type Server struct {
	sceneproofv1.UnimplementedEdgeIngressServer
	o   Options
	log *slog.Logger
}

// New creates the server.
func New(o Options) *Server {
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	return &Server{o: o, log: o.Logger.With("component", "ingress")}
}

// Register attaches the service to a gRPC server.
func (s *Server) Register(gs *grpc.Server) { sceneproofv1.RegisterEdgeIngressServer(gs, s) }

// SubmitEvent durably enqueues one event snapshot.
//
// Ack semantics mirror the server's: accepted=true means "this revision is
// now the pending one and is on disk"; duplicate=true means "an equal or
// newer revision is already pending" (nothing written, nothing lost). Both
// tell the analyzer it may forget the snapshot. Errors (disk full, oversize
// event) are returned as gRPC errors so the analyzer knows the event is NOT
// safe to forget.
func (s *Server) SubmitEvent(ctx context.Context, ev *sceneproofv1.Event) (*sceneproofv1.Ack, error) {
	if ev.GetEventId() == "" {
		return nil, grpcstatus.Error(codes.InvalidArgument, "event_id is required")
	}
	if ev.DeviceId == "" {
		ev.DeviceId = s.o.DeviceID
	}
	res, err := s.o.Queue.Put(ev)
	if err != nil {
		switch {
		case errors.Is(err, queue.ErrTooLarge):
			return nil, grpcstatus.Error(codes.ResourceExhausted, err.Error())
		case errors.Is(err, queue.ErrClosed):
			return nil, grpcstatus.Error(codes.Unavailable, "gateway is shutting down")
		default:
			s.log.Error("queue put failed", "event_id", ev.EventId, "error", err.Error())
			return nil, grpcstatus.Error(codes.Internal, "queue write failed: "+err.Error())
		}
	}
	if res.Evicted > 0 {
		s.log.Warn("queue evicted events to admit new one", "evicted", res.Evicted, "event_id", ev.EventId)
	}
	s.log.Debug("event accepted", "event_id", ev.EventId, "revision", ev.Revision,
		"status", ev.GetStatus().String(), "duplicate", res.Duplicate)
	return &sceneproofv1.Ack{
		EventId:   ev.EventId,
		Revision:  ev.Revision,
		Accepted:  res.Stored,
		Duplicate: res.Duplicate,
	}, nil
}

// ReportStatus stores the analyzer's latest status (not queued: only the
// newest matters for the next heartbeat).
func (s *Server) ReportStatus(ctx context.Context, ds *sceneproofv1.DeviceStatus) (*sceneproofv1.Empty, error) {
	if ds.DeviceId == "" {
		ds.DeviceId = s.o.DeviceID
	}
	s.o.Status.SetAnalyzerStatus(ds)
	return &sceneproofv1.Empty{}, nil
}

// SubscribeCommands streams commands forwarded from the server until the
// analyzer hangs up.
func (s *Server) SubscribeCommands(req *sceneproofv1.WatchRequest, stream sceneproofv1.EdgeIngress_SubscribeCommandsServer) error {
	sub := s.o.Bus.Subscribe()
	defer s.o.Bus.Unsubscribe(sub)
	s.log.Info("analyzer subscribed to commands", "device_id", req.GetDeviceId(), "subscribers", s.o.Bus.Subscribers())
	ctx := stream.Context()
	for {
		select {
		case <-ctx.Done():
			s.log.Info("analyzer command subscription closed", "reason", ctx.Err().Error())
			return nil
		case cmd := <-sub.C:
			if err := stream.Send(cmd); err != nil {
				s.log.Warn("send command to analyzer failed", "command_id", cmd.GetCommandId(), "error", err.Error())
				return err
			}
		}
	}
}

// ReportCommandResult forwards the analyzer's outcome to the server.
func (s *Server) ReportCommandResult(ctx context.Context, res *sceneproofv1.CommandResult) (*sceneproofv1.Empty, error) {
	if res.GetCommandId() == "" {
		return nil, grpcstatus.Error(codes.InvalidArgument, "command_id is required")
	}
	if res.DeviceId == "" {
		res.DeviceId = s.o.DeviceID
	}
	s.o.Results.Push(res)
	return &sceneproofv1.Empty{}, nil
}
