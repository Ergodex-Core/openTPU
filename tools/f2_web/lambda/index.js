// openTPU on F2: the demo page, as a Lambda web function (Node.js, no dependencies). It serves
// public/index.html and forwards /api/* to the F2 host's gateway (tools/f2_web/gateway.py) over
// the VPC (the function's network connector; GATEWAY_URL, the instance's private address). The
// endpoint is ApplicationManaged: every /api/* call needs the passcode (PASSCODE, header
// x-passcode), and the endpoint's rate limit and one environment bound what reaches the card.
'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const GATEWAY = new URL(process.env.GATEWAY_URL || 'http://127.0.0.1:8080');
const PASSCODE = process.env.PASSCODE || '';
const PAGE = fs.readFileSync(path.join(__dirname, 'public', 'index.html'));
const MAX_BODY = 64 * 1024;

function passOk(req) {
  const got = Buffer.from(String(req.headers['x-passcode'] || ''));
  const want = Buffer.from(PASSCODE);
  return PASSCODE !== '' && got.length === want.length && crypto.timingSafeEqual(got, want);
}

function send(res, code, obj) {
  const b = Buffer.from(JSON.stringify(obj));
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': b.length });
  res.end(b);
}

function forward(req, res, body) {
  const up = http.request({
    host: GATEWAY.hostname, port: GATEWAY.port || 80, path: req.url, method: req.method,
    headers: { 'content-type': 'application/json', 'content-length': body.length },
    timeout: 600000,
  }, (r) => {
    const h = { 'content-type': r.headers['content-type'] || 'application/json',
                'cache-control': 'no-cache' };
    res.writeHead(r.statusCode || 502, h);
    r.pipe(res);                          // SSE chunks stream straight through
  });
  up.on('timeout', () => up.destroy(new Error('gateway timeout')));
  up.on('error', (e) => { if (!res.headersSent) send(res, 502, { error: String(e.message) }); else res.end(); });
  // the client went away before the reply ended (a request's 'close' fires once its body is
  // read, so the response's is the one that means a disconnect)
  res.on('close', () => { if (!res.writableEnded) up.destroy(); });
  up.end(body);
}

const server = http.createServer((req, res) => {
  if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
    return res.end(PAGE);
  }
  if (req.url === '/healthz') return send(res, 200, { ok: true });
  const api = req.url === '/api/models' || req.url === '/api/select' || req.url === '/api/chat';
  if (!api) return send(res, 404, { error: 'not found' });
  if (!passOk(req)) return send(res, 401, { error: 'passcode required' });
  const chunks = [];
  let n = 0;
  req.on('data', (c) => { n += c.length; if (n > MAX_BODY) req.destroy(); else chunks.push(c); });
  req.on('end', () => forward(req, res, Buffer.concat(chunks)));
});

const [host, port] = (process.env.AWS_LAMBDA_HTTP_ENDPOINT || '0.0.0.0:3000').split(':');
server.listen(Number(port), host, () => console.log(`opentpu-f2 web on ${host}:${port} -> ${GATEWAY.href}`));
