# ZoneWatch Edge Gateway (Go)

산업현장 제한구역 영상 사건 검토 서비스의 **디바이스 측 게이트웨이**입니다. 같은 호스트(또는 같은 Pod)에서 돌아가는 Python 분석기(analyzer)가 만든 사건 스냅샷·상태를 받아 **디스크에 안전하게 보관한 뒤** 네트워크 너머의 검토 서버(review server)로 전달하고, 반대로 서버의 명령(Command)을 분석기에 중계합니다.

```
analyzer (Python, 디바이스) --gRPC EdgeIngress (localhost:50051)--> gateway (Go, 디바이스) --gRPC Ingest (network, server:50052)--> review server (Python)
```

와이어 계약은 `proto/zonewatch/v1/zonewatch.proto` 하나이며, 생성 코드는 `gen/zonewatchv1/`에 있습니다(수정·재생성하지 않습니다).

---

## 1. 책임과 경계

| 구분 | 분석기 (Python) | **게이트웨이 (Go, 이 디렉터리)** | 검토 서버 (Python) |
|---|---|---|---|
| 사건 생성 | 탐지·추적·구역 판정, `Event` 스냅샷 생성 (`revision` 증가) | – | – |
| 사건 보관 | 게이트웨이가 `accepted=true`를 돌려줄 때까지만 | **내구성 있는 유계(bounded) 큐 (WAL)** | 영구 저장, `(event_id, revision)` 중복 제거의 최종 권위 |
| 증거 파일 | 키프레임 jpg·클립 mp4를 공유 디렉터리에 기록 | 파일을 찾아 **청크 업로드**, 업로드 완료 원장 유지 | 저장, `already_present` 응답 |
| 디바이스 상태 | `ReportStatus`로 카메라 상태 보고 | 최신 것만 메모리에 보관, `UplinkStatus`·버전을 합쳐 **하트비트** 전송 | 디바이스 온라인 여부 표시 |
| 명령 | `SubscribeCommands`로 수신, 실행 후 `ReportCommandResult` | 서버 `WatchCommands` 스트림 유지, **팬아웃**, 결과 중계 | 콘솔에서 명령 발행 |
| 연결 | localhost 만 | **재연결·백오프, 상태 머신** | – |

게이트웨이가 **하지 않는 것**: 사건 내용 해석, 중복 제거의 최종 판단(서버가 함), 증거 파일 생성·삭제, 인증.

분석기와 게이트웨이는 증거 디렉터리(`--evidence-root`)를 바인드 마운트로 공유합니다. `Event.evidence.key_frames[].path`·`clip_path`는 이 루트 기준 상대 경로입니다.

---

## 2. 코드 구성

```
cmd/zonewatch-gateway/   main.go — 플래그/환경변수, 와이어링, 정상 종료
internal/queue/          WAL 기반 내구성 큐 (코얼레싱, 퇴출, 컴팩션, 복구)
internal/uplink/         서버 방향: Conn(상태 머신·백오프), Sender, Heartbeat,
                         CommandWatcher, Results(명령 결과), Uploader+Ledger(증거)
internal/ingress/        분석기 방향: EdgeIngress 서버, 명령 Bus(팬아웃·지연 구독 버퍼)
internal/status/         단일 상태 레지스트리 + /healthz /readyz /status /metrics
internal/version/        빌드 버전
```

각 구성요소는 상태 레지스트리(`status.Registry`)에 사실을 기록하고, 하트비트·`/status`·`/metrics`는 모두 같은 레지스트리를 읽습니다. 그래서 세 곳의 숫자가 서로 어긋나지 않습니다.

---

## 3. 자료구조

### 3.1 WAL 레코드 포맷 (`<queue-dir>/events.wal`)

append-only 파일, 모든 정수는 little-endian:

| 오프셋 | 크기 | 내용 |
|---|---|---|
| 0 | uint32 | `length` = 뒤따르는 바이트 수 (type 1바이트 + payload) |
| 4 | uint32 | `crc32` (IEEE) — `type ‖ payload` 에 대한 체크섬 |
| 8 | byte | `type`: `1=PUT`, `2=ACK`, `3=DROP` |
| 9 | length-1 | `payload` |

- `PUT`: payload = `proto.Marshal(Event)`
- `ACK`, `DROP`: payload = `uint32 revision ‖ event_id`

길이가 앞에 있어 순차 읽기가 가능하고, CRC가 있어 **찢어진 꼬리(torn write)** 와 비트 손상을 구분 없이 "신뢰할 수 없는 레코드"로 처리할 수 있습니다. 하나의 레코드는 단일 `Write` 호출로 기록되므로 크래시는 파일 **끝**에만 손상을 남깁니다.

### 3.2 인메모리 인덱스와 FIFO

```
index : map[event_id] -> *entry{ev *Event, size int64, elem *list.Element}
order : container/list  — 최초 enqueue 순서 (FIFO)
```

- `Put`: 같은 `event_id`가 대기 중이고 새 `revision`이 더 크면 **그 자리에서 교체**(FIFO 위치 유지). 같거나 작으면 `Duplicate`로 응답하고 아무것도 쓰지 않습니다.
- `Pending()`: FIFO 순서의 스냅샷(각 event_id의 최신 revision).
- `Ack(event_id, revision)`: `revision ≥ 대기 중 revision` 일 때만 제거.
- `liveBytes` = 대기 중 payload 바이트 합. 이것이 `--queue-max-bytes` 와 비교되는 값입니다.

복구는 파일을 처음부터 끝까지 한 번 읽으면서 위와 동일한 `Put/Ack` 로직을 재적용하는 것이라 메모리 상태와 디스크 상태가 정의상 일치합니다.

### 3.3 업로드 원장 (`<queue-dir>/uploads.ledger`)

탭 구분 텍스트, append-only:

```
W<TAB>event_id<TAB>path                 wanted: ack된 사건이 이 파일을 참조함
D<TAB>event_id<TAB>path<TAB>sha256      done: 서버가 보유 (업로드 또는 already_present)
```

재시작 시 `W − D` 를 다시 스케줄합니다. 이것이 없으면 "CLOSED revision은 ack됐는데 클립 업로드 전에 프로세스가 죽은" 경우 클립이 조용히 사라집니다. 포기한 파일은 빈 sha256으로 `D`를 기록해 되살아나지 않게 합니다.

---

## 4. 정책과 이유

| 정책 | 내용 | 이유 |
|---|---|---|
| **코얼레싱** | 같은 event_id의 새 revision이 대기 중인 옛 revision을 대체 | proto 설계상 `Event`는 완전한 스냅샷이므로 옛 것을 보낼 이유가 없음. 오프라인 동안 entry→dwell→exit 3개가 쌓여도 1개만 전송 |
| **FIFO 위치 유지** | 코얼레싱해도 최초 enqueue 위치 유지 | 오래 열려 있던 에피소드가 나중에 들어온 사건들 뒤로 밀려 굶지 않도록 |
| **Ack 의미** | `accepted`·`duplicate` 모두 큐에서 제거 | 둘 다 "서버가 그 revision을 가지고 있다"는 뜻. 전송 오류만 재시도 |
| **Ack vs 새 revision** | 전송 중 새 revision이 도착하면 옛 revision의 ack로 제거하지 않음 | 새 스냅샷은 아직 서버에 없으므로 다음 패스에서 전송돼야 함 |
| **퇴출(eviction)** | `liveBytes > max` 이면 **가장 오래된** 사건부터 DROP, `dropped_events` 집계, WARN 로그 | 장시간 오프라인 후엔 최신 에피소드가 운영자에게 더 가치 있음. 한 사건이 예산보다 크면 `ResourceExhausted`로 거부(거짓 accepted를 주지 않음) |
| **fsync** | PUT 뒤 fsync(기본), ACK/DROP은 fsync 안 함 | PUT은 분석기에 준 `accepted=true` 약속의 근거. ACK 유실은 재전송→서버 dedup으로 흡수되므로 비용을 치르지 않음 |
| **컴팩션** | 파일 > 2×live 이고 ≥ 1 MiB(기본) 일 때, 그리고 시작 시 죽은 레코드가 있으면 live PUT만 FIFO 순으로 새 파일에 쓰고 rename | 쓰기 비용을 사건당 O(1)로 상각. 임시 파일→fsync→rename→dir fsync 순서라 언제든 완전한 파일 하나가 존재 |
| **손상된 꼬리** | 첫 불량 레코드에서 멈추고 그 오프셋으로 truncate, WARN | 길이 접두 스트림 안에서 안전하게 재동기화할 방법이 없음. CRC가 맞는데 proto 파싱이 실패하는 레코드는 건너뜀(찢어진 것이 아니므로 뒤의 정상 레코드를 버리지 않음) |
| **연결 상태 머신** | `DISCONNECTED → CONNECTING → CONNECTED`, 실패 시 ClientConn을 버리고 새로 dial | grpc-go의 내장 재연결은 상태·횟수를 숨김. 시도마다 새 연결을 만들면 `reconnect_count`, `last_error`가 정확 |
| **세대(generation)** | 연결마다 번호를 매기고 실패 보고에 세대를 첨부 | 느린 RPC가 옛 연결의 오류를 보고해 새 건강한 연결을 끊는 일을 방지 |
| **백오프** | 500 ms × 2ⁿ, 최대 30 s, equal-jitter(`[d/2, d]`), 성공 시 리셋 | 서버 복구 시 여러 디바이스가 동시에 재접속하는 thundering herd 방지. 하한 `d/2`로 0에 수렴하지 않음 |
| **전송 오류 분류** | `Unavailable`·`DeadlineExceeded` → 재연결. 그 외 코드 → 연결 유지, 해당 사건만 30 s 쿨다운 후 재시도 | 서버가 거부한 사건 하나가 큐 전체를 막지 않도록(head-of-line blocking 방지). 거부가 일시적일 수 있으므로(롤링 업그레이드) 게이트웨이는 용량 외의 이유로 사건을 버리지 않음 |
| **순차 전송** | 사건을 한 번에 하나씩 전송 | 디바이스별 순서 보존, "전송 중"이 최대 1건이라 재연결당 중복이 최대 1건 |
| **하트비트** | 2 s(기본)마다 분석기 최신 상태 복사본 + `UplinkStatus` + `gateway_version`, `ts`=전송 시각. 분석기 보고가 없어도 빈 cameras로 전송 | 서버가 "게이트웨이는 살아 있고 큐 깊이는 이렇다"를 알 수 있어야 함. 연결 끊김 중의 tick은 건너뜀(하트비트는 지금만 의미 있음). RPC 데드라인은 10 s(간격보다 길게) — 느린 링크를 죽은 링크로 오인하지 않기 위해; TCP 단절은 채널 상태 감시로 즉시 감지 |
| **명령 팬아웃** | 구독자마다 64개 버퍼 채널, 가득 차면 가장 오래된 것 폐기. 구독자가 없으면 최근 32개를 60 s 보관 후 첫 구독자에게 전달하고 비움 | 분석기가 재시작 중일 때가 바로 콘솔이 START_CAMERA를 보내는 순간. 버퍼를 작고 짧게 두는 이유: 1분 지난 명령은 원하는 것보다 stale일 가능성이 높고 서버가 재발행함. command_id 중복 제거는 하지 않음(분석기 측 멱등성이 계약이고, 여기서 억제하면 분석기가 못 본 명령을 숨길 수 있음) |
| **명령 결과** | 메모리 큐 256개, 가득 차면 가장 오래된 것 폐기, 반복 거부 시 5회 후 포기 | 결과는 보조 정보(카메라 상태가 DeviceStatus로도 전달됨). 사건과 달리 내구성이 필요하지 않음 |
| **증거 업로드** | ack 직후 참조 파일 스케줄, 256 KiB 청크, `offset`, 마지막 청크에 `sha256`·`total_size`. 파일이 없으면 백오프 재시도, `--upload-retry-window`(10 m) 지나면 포기·로그. `ok` 또는 `already_present` 모두 성공 | 클립은 close 수 초 뒤에 쓰임. 해시는 전송한 바이트 그대로 계산하고 크기는 시작 시 고정해 `(sha256, total_size)`가 항상 자기 일관적 |
| **settle 대기** | mtime이 1 s 이내인 파일은 아직 쓰는 중으로 보고 잠시 뒤 재시도 | 쓰는 중인 mp4를 올리면 서버에 잘린 클립이 남음 |
| **경로 안전** | `filepath.IsLocal`이 아닌 경로(`..`, 절대경로)는 읽지 않고 ERROR | 공유 디렉터리가 파일 유출 통로가 되지 않도록 |
| **원장 키** | `(event_id, path)` 기준으로 "이미 올림" 판단 | 분석기는 증거 파일을 한 번 쓰고 덮어쓰지 않음. 매 revision마다 큰 클립을 재해시하는 CPU 비용 회피 |
| **종료 순서** | gRPC 수신 중단(GracefulStop, 5 s) → 루프 취소 → HTTP 종료 → 원장·WAL fsync 후 close | 종료 중 분석기는 ack 대신 오류를 받아 스냅샷을 보존하고, 이미 accepted 된 것은 반드시 디스크에 남음 |

---

## 5. 장애 처리 시나리오

| 상황 | 동작 |
|---|---|
| **서버 다운** | 채널 상태 감시 또는 RPC `Unavailable`로 즉시 `DISCONNECTED`. 백오프로 재접속 시도(`reconnect_count` 증가, `last_error` 기록). 그동안 분석기의 `SubmitEvent`는 정상적으로 WAL에 쌓이고 `accepted=true`. 복구되면 대기 사건을 FIFO로 각 1회 전송, 명령 스트림 재수립, 하트비트 재개 |
| **게이트웨이 재시작/크래시** | WAL 재생 → 동일한 대기 집합·순서 복원 → 죽은 레코드가 있으면 컴팩션. 원장의 `W−D` 업로드 재스케줄. 재시작 직전 "전송했지만 ack를 못 받은" 사건 1건은 재전송되며 서버가 `duplicate=true`로 흡수 |
| **중복** | 분석기→게이트웨이: 대기 중 revision 이하이면 `duplicate=true`(쓰기 없음). 게이트웨이→서버: 서버 dedup이 최종 권위, `duplicate` ack도 큐에서 제거하고 `duplicate_acks` 집계. 업로드: 원장 + 서버 `already_present` |
| **큐 가득 참** | 가장 오래된 사건부터 DROP(WAL에 기록되어 재시작 후에도 돌아오지 않음), `dropped_events`/`zonewatch_gateway_events_dropped_total` 증가, WARN 로그. 예산보다 큰 단일 사건은 `ResourceExhausted` 거부 |
| **WAL 꼬리 손상** | 복구 시 첫 불량 레코드에서 truncate, `queue.truncated=true`가 `/status`에 표시 |
| **디스크 쓰기 실패** | `SubmitEvent`가 `Internal` 오류 → 분석기는 스냅샷을 버리면 안 됨 |
| **분석기 미접속** | 하트비트는 빈 `cameras`로 계속 전송. 명령은 최근 32개/60 s 버퍼 |
| **서버가 사건 거부**(`InvalidArgument` 등) | 연결 유지, 해당 사건만 30 s 쿨다운, 다른 사건은 계속 전송. 사건은 버리지 않음 |
| **증거 파일이 끝내 안 생김** | 재시도 창 뒤 포기, ERROR 로그, `uploads_failed` 증가, 원장에 기록(재시작 후 되살아나지 않음) |

---

## 6. 플래그 / 환경변수

모든 플래그는 `ZW_GW_<대문자, -→_>` 환경변수로 대체 가능하며 플래그가 우선합니다.

| 플래그 | 환경변수 | 기본값 | 설명 |
|---|---|---|---|
| `--listen` | `ZW_GW_LISTEN` | `:50051` | EdgeIngress gRPC 수신 주소 (분석기 측) |
| `--server` | `ZW_GW_SERVER` | `127.0.0.1:50052` | 검토 서버 Ingest 주소 |
| `--device-id` | `ZW_GW_DEVICE_ID` | `edge-01` | 디바이스 ID |
| `--queue-dir` | `ZW_GW_QUEUE_DIR` | `./data/gateway` | WAL·원장 디렉터리 |
| `--queue-max-bytes` | `ZW_GW_QUEUE_MAX_BYTES` | `67108864` (64 MiB) | 큐 예산(live payload 바이트) |
| `--queue-fsync` | `ZW_GW_QUEUE_FSYNC` | `true` | PUT 뒤 fsync |
| `--evidence-root` | `ZW_GW_EVIDENCE_ROOT` | `./data/evidence` | 분석기와 공유하는 증거 디렉터리 |
| `--http` | `ZW_GW_HTTP` | `:9100` | 로컬 HTTP 주소(빈 문자열이면 비활성) |
| `--heartbeat-interval` | `ZW_GW_HEARTBEAT_INTERVAL` | `2s` | 하트비트 주기 |
| `--upload-retry-window` | `ZW_GW_UPLOAD_RETRY_WINDOW` | `10m` | 없는 증거 파일을 기다리는 시간 |
| `--insecure` | `ZW_GW_INSECURE` | `false` | 서버와 평문 gRPC. 미지정 시 시스템 루트 CA 기반 TLS(클라이언트 인증 없음) |
| `--log-level` | `ZW_GW_LOG_LEVEL` | `info` | `debug|info|warn|error` (JSON, stderr) |
| `--version` | – | – | 버전 출력 후 종료 |

### 로컬 HTTP

| 경로 | 내용 |
|---|---|
| `/healthz` | 프로세스가 살아 있으면 200 |
| `/readyz` | 큐가 열려 있고 gRPC 리스너가 떠 있으면 200, 아니면 503 |
| `/status` | JSON: 업링크 상태·카운터, 큐 통계, 업로드 통계, 분석기 최신 `DeviceStatus`(protojson), 버전 |
| `/metrics` | Prometheus 텍스트: `zonewatch_gateway_queue_pending`, `_queue_bytes`, `_events_sent_total`, `_events_acked_total`, `_events_dropped_total`, `_uplink_connected`, `_uplink_reconnects_total`, `_uploads_pending`, `_uploads_done_total` 외 |

---

## 7. 실행과 테스트

```bash
cd gateway
make build            # bin/zonewatch-gateway (ldflags -X main.version=<git describe>)
make test             # go test -race ./...
make vet              # go vet ./...
make fmt              # gofmt -w

./bin/zonewatch-gateway --insecure --server 10.0.0.5:50052 --device-id edge-01 \
  --queue-dir /var/lib/zonewatch/gateway --evidence-root /var/lib/zonewatch/evidence

curl -s localhost:9100/status | jq .uplink
curl -s localhost:9100/metrics | grep zonewatch_gateway_queue
```

Docker (정적 바이너리, distroless, 비루트, `EXPOSE 50051 9100`):

```bash
docker build -f gateway/Dockerfile -t zonewatch-gateway gateway/
docker run --rm -p 50051:50051 -p 9100:9100 -v zw-data:/data \
  -e ZW_GW_SERVER=server:50052 -e ZW_GW_INSECURE=true zonewatch-gateway
```

컨테이너 안에서는 `/data/gateway`(큐)와 `/data/evidence`(증거)가 기본값이며, 증거 볼륨은 분석기 컨테이너와 같은 것을 마운트해야 합니다.

테스트는 모두 인프로세스입니다. `internal/uplink`의 테스트는 실제 gRPC 서버(`127.0.0.1:0`)에 `(event_id → max revision)` dedup을 가진 가짜 Ingest를 띄워 전송·ack, 서버 중단→백오프, **같은 포트로 재시작→대기 사건 정확히 1회씩 재전송**, 코얼레싱된 revision은 최신만 전송, 증거 청크·sha256·`already_present`·지연 생성 파일·재시작 후 미재업로드, 하트비트 주기·병합을 검증합니다. 툴체인은 Go 1.24(`GOTOOLCHAIN=local`), 의존성은 `google.golang.org/grpc v1.80.0`, `google.golang.org/protobuf v1.36.12` 뿐입니다.

---

## 8. 다루지 않는 것 (현재 범위 밖)

- **TLS 클라이언트 인증 / 토큰 인증**: `--insecure`가 아니면 시스템 루트 CA로 서버 인증서만 검증합니다. mTLS, API 토큰, 디바이스 등록은 없습니다.
- **다중 디바이스 fan-in**: 게이트웨이 하나 = 디바이스 하나. 여러 분석기를 한 게이트웨이에 묶는 구성은 지원하지 않습니다(`device_id`가 비어 있으면 게이트웨이 것을 채워 넣기만 합니다).
- **증거 압축·트랜스코딩**: 파일을 있는 그대로 올립니다.
- **업로드 이어받기(resume)**: 전송 중 끊기면 파일 전체를 다시 보냅니다(서버 `already_present`로 중복은 흡수).
- **원장 로테이션**: `uploads.ledger`는 파일당 두 줄씩 계속 자랍니다(수십만 파일에 수십 MB 수준).
- **명령 command_id 중복 제거**: 분석기 측 멱등성에 맡깁니다.
- **서버의 `Ack` 외 백프레셔 신호**: 서버가 과부하를 알릴 수단이 계약에 없으므로 게이트웨이는 순차 전송과 데드라인으로만 속도를 제한합니다.
- **멀티 프로세스 큐 공유**: 같은 `--queue-dir`을 두 프로세스가 동시에 열면 안 됩니다(잠금 파일 없음).
