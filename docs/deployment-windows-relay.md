# Windows TCP relay deployment

The supported layout keeps SQLite, source media, generated artifacts, and all
workers on one Linux/Ascend server. A Windows management machine can expose the
same HTTP service to the LAN using a full TCP relay; it does not run VFIEval jobs
or copy the database.

## 1. Start VFIEval on the compute server

Bind VFIEval to an address reachable from the Windows relay and run the deep
diagnostic first:

```text
python -m vfieval.cli --workspace .vfieval doctor
python -m vfieval.cli --workspace .vfieval serve --host 0.0.0.0 --port 8765
```

Restrict the compute-server firewall so port 8765 accepts the relay machine,
not the whole untrusted network.

## 2. Install the Windows relay

Open PowerShell as Administrator in the repository directory. Replace the
target address with the compute server's stable IPv4 address or DNS name:

```powershell
.\scripts\windows_tcp_relay.ps1 install -TargetAddress 10.0.0.20 -TargetPort 8765 -ListenPort 8765
```

The script creates one exact `netsh interface portproxy` rule and one inbound
firewall rule for Domain/Private networks. It resolves DNS to IPv4 during
installation so the forwarding target is explicit.

Inspect or remove that exact listener with:

```powershell
.\scripts\windows_tcp_relay.ps1 status -ListenPort 8765
.\scripts\windows_tcp_relay.ps1 remove -ListenPort 8765
```

## 3. Verify the real application path

Run the self-test on the relay machine:

```powershell
.\scripts\windows_tcp_relay.ps1 self-test -BaseUrl http://127.0.0.1:8765
```

It checks the homepage, `/api/health`, the isolated blind-evaluation page, and
an HTTP byte-range response used by video playback. A successful TCP connection
alone is not considered sufficient.

LAN reviewers then open `http://<relay-machine-ip>:8765/evaluate/<opaque-token>`.
The opaque Campaign link is created only after package publication succeeds.

## Troubleshooting

- Run `vfieval doctor --json` on the compute server for devices, FFmpeg,
  metrics, SQLite, permissions, disk, and port checks.
- Internal server errors include `support_id` and `request_id`. Search those in
  `.vfieval/logs/server.jsonl`.
- Create a sanitized bundle with
  `vfieval --workspace .vfieval diagnostics --campaign-id <id>` or
  `--run-id <id>`.
- A logged `http.client_disconnected`/Broken pipe event describes the response
  connection. Campaign publication remains a durable queued operation and must
  be checked through the Campaign status rather than inferred from that socket.
