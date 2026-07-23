import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readFile } from "node:fs/promises";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(currentDir, "..", "..");
const webRoot = path.join(projectRoot, "src", "vfieval", "web");
const port = Number(process.env.VFIEVAL_BROWSER_TEST_PORT || 4173);

const files = new Map([
  ["/", ["index.html", "text/html; charset=utf-8"]],
  ["/index.html", ["index.html", "text/html; charset=utf-8"]],
  ["/app.js", ["app.js", "text/javascript; charset=utf-8"]],
  ["/compare.js", ["compare.js", "text/javascript; charset=utf-8"]],
  ["/run-detail.js", ["run-detail.js", "text/javascript; charset=utf-8"]],
  ["/media.js", ["media.js", "text/javascript; charset=utf-8"]],
  ["/shared.js", ["shared.js", "text/javascript; charset=utf-8"]],
  ["/studio.js", ["studio.js", "text/javascript; charset=utf-8"]],
  ["/styles.css", ["styles.css", "text/css; charset=utf-8"]],
  ["/studio.css", ["studio.css", "text/css; charset=utf-8"]],
  ["/blind.html", ["blind.html", "text/html; charset=utf-8"]],
  ["/blind.js", ["blind.js", "text/javascript; charset=utf-8"]],
  ["/blind.css", ["blind.css", "text/css; charset=utf-8"]],
]);

const pixelPng = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);

function send(response, status, contentType, body) {
  response.writeHead(status, {
    "Content-Type": contentType,
    "Cache-Control": "no-store",
  });
  response.end(body);
}

const server = http.createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host || "127.0.0.1"}`);
  if (url.pathname === "/__fixture_health") {
    send(response, 200, "text/plain; charset=utf-8", "ok");
    return;
  }
  if (url.pathname === "/fixtures/pixel.png") {
    send(response, 200, "image/png", pixelPng);
    return;
  }
  if (url.pathname.startsWith("/fixtures/pending")) {
    request.on("close", () => {
      if (!response.writableEnded) response.destroy();
    });
    return;
  }
  if (url.pathname.startsWith("/evaluate/")) {
    const [relativePath, contentType] = files.get("/blind.html");
    send(response, 200, contentType, await readFile(path.join(webRoot, relativePath)));
    return;
  }
  const staticFile = files.get(url.pathname);
  if (staticFile) {
    const [relativePath, contentType] = staticFile;
    send(response, 200, contentType, await readFile(path.join(webRoot, relativePath)));
    return;
  }
  send(response, 404, "text/plain; charset=utf-8", "fixture not found");
});

server.listen(port, "127.0.0.1", () => {
  process.stdout.write(`VFIEval browser fixture listening on http://127.0.0.1:${port}\n`);
});

function stop() {
  server.close(() => process.exit(0));
}

process.on("SIGINT", stop);
process.on("SIGTERM", stop);
