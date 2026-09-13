#!/usr/bin/env node
/* End-to-end test of the 3D page in headless Chrome, with no dependency (Node >= 22 for
   its built-in WebSocket, and Chrome). It starts a test server on a spare port (a running
   ./run.sh is not disturbed), drives every mode, follows a run streamed by flyenv, replays
   it and films a video. Screenshots go to tests/out/. About two minutes.

       node tests/e2e.mjs               # CHROME=/path/to/chrome to choose the browser

   Each check prints PASS or FAIL; the exit code is the number of failures. */
import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const OUT = path.join(ROOT, "tests", "out"), PY = path.join(ROOT, ".venv", "bin", "python");
const REC = path.join(ROOT, "runs", "recordings");
const sleep = ms => new Promise(r => setTimeout(r, ms));
const results = [];
const check = (name, ok, detail = "") => { results.push(!!ok); console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? " · " + detail : ""}`); };

function findChrome() {
  for (const p of [process.env.CHROME, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                   "/Applications/Chromium.app/Contents/MacOS/Chromium", "google-chrome", "chromium", "chromium-browser"]) {
    if (p && (p.includes("/") ? existsSync(p) : spawnSync("which", [p]).status === 0)) return p;
  }
  return null;
}
const freePort = () => new Promise(res => { const s = createServer(); s.listen(0, "127.0.0.1", () => { const p = s.address().port; s.close(() => res(p)); }); });
async function waitFor(url) {
  for (let i = 0; i < 120; i++) { try { if ((await fetch(url)).ok) return; } catch {} await sleep(500); }
  throw new Error(`timeout waiting for ${url}`);
}

const chrome = findChrome();
if (typeof WebSocket === "undefined") { console.log("Node >= 22 is needed (built-in WebSocket)"); process.exit(2); }
if (!chrome) { console.log("Chrome not found: set CHROME=/path/to/chrome"); process.exit(2); }
mkdirSync(OUT, { recursive: true });
const profile = path.join(OUT, "chrome-profile");
rmSync(profile, { recursive: true, force: true });
const recBefore = new Set(existsSync(REC) ? readdirSync(REC) : []);
const port = await freePort(), cdpPort = await freePort(), U = `http://127.0.0.1:${port}`;
const srv = spawn(PY, ["-u", path.join(ROOT, "dashboard.py"), "--port", String(port)], { stdio: "ignore" });
const browser = spawn(chrome, ["--headless=new", `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${profile}`,
  "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--window-size=1200,800", "--no-first-run", "about:blank"], { stdio: "ignore" });
const cleanup = () => {
  browser.kill(); srv.kill();
  for (const f of existsSync(REC) ? readdirSync(REC) : []) if (!recBefore.has(f)) { try { unlinkSync(path.join(REC, f)); } catch {} }
};
const watchdog = setTimeout(() => {                         // never hang: fail after 5 minutes
  check("the test finished within 5 minutes", false); cleanup(); process.exit(99);
}, 300000);

try {
  await waitFor(`${U}/api/live/session`);
  await waitFor(`http://127.0.0.1:${cdpPort}/json/version`);
  const tgt = await (await fetch(`http://127.0.0.1:${cdpPort}/json/new?${U}/fly`, { method: "PUT" })).json();
  const ws = new WebSocket(tgt.webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener("open", r, { once: true }));
  let id = 0;
  const pend = new Map(), errors = [];
  ws.addEventListener("message", e => {
    const m = JSON.parse(e.data);
    if (m.id && pend.has(m.id)) { pend.get(m.id)(m); pend.delete(m.id); }
    else if (m.method === "Runtime.exceptionThrown") errors.push((m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 200));
    else if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") errors.push(m.params.args.map(a => a.value ?? a.description).join(" ").slice(0, 200));
  });
  const cdp = (method, params = {}) => new Promise(r => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async x => { const r = await cdp("Runtime.evaluate", { expression: x, returnByValue: true, awaitPromise: true }); return r.result?.result?.value; };
  const until = async (expr, ms = 15000) => { const end = Date.now() + ms; while (Date.now() < end) { if (await ev(expr)) return true; await sleep(300); } return false; };
  const key = async (code, k, vk, holdMs = 0) => {
    await cdp("Input.dispatchKeyEvent", { type: "keyDown", code, key: k, windowsVirtualKeyCode: vk, ...(k.length === 1 ? { text: k } : {}) });
    if (holdMs) await sleep(holdMs);
    await cdp("Input.dispatchKeyEvent", { type: "keyUp", code, key: k, windowsVirtualKeyCode: vk });
  };
  const shot = async name => writeFileSync(path.join(OUT, `${name}.png`), Buffer.from((await cdp("Page.captureScreenshot", { format: "png" })).result.data, "base64"));
  const mode = m => ev(`document.querySelector('[data-mode="${m}"]').click()`);
  await cdp("Runtime.enable"); await cdp("Page.enable");
  await cdp("Page.setDownloadBehavior", { behavior: "allow", downloadPath: OUT });
  await sleep(2500);

  check("the page loads with its four modes",
        (await ev(`[...document.querySelectorAll('#mode-seg button')].map(b => b.dataset.mode).join()`)) === "brain,keys,board,car");

  await mode("keys");
  await until(`document.querySelector('#live-start').textContent === 'Stop'`);
  await sleep(1000);
  const z0 = await ev(`window.fly.model.root.position.z`);
  await key("ArrowUp", "ArrowUp", 38, 2500);
  const z1 = await ev(`window.fly.model.root.position.z`);
  check("Keyboard: ↑ walks forward (ICL012m → DNp09)", z1 - z0 > 2, `moved ${(z1 - z0).toFixed(1)}`);
  await shot("keyboard");

  await mode("board"); await sleep(2000);
  await key("Digit1", "1", 49);
  check("Switches: key 1 flips switch 1 (DNg12_e → front leg)", await until(`window.fly.board.isOn(0)`, 6000));
  await shot("switches");

  await mode("car"); await sleep(1500);
  await key("ArrowUp", "ArrowUp", 38, 2500);
  const speed = await ev(`(window.fly.brain.kart || [0, 0, 0, 0])[3]`);
  check("Drive: ↑ accelerates the kart (DNg16 → right hind leg)", speed > 2, `speed ${(+speed).toFixed(1)}`);
  await shot("drive");

  await ev(`document.querySelector('#vid-rec').click()`); await sleep(2500);
  await ev(`document.querySelector('#vid-rec').click()`);
  const filmed = await until(`window.fly.lastVideo && window.fly.lastVideo.size > 10000`, 6000);
  check("Record video: the 3D view is filmed to WebM", filmed, filmed ? `${Math.round((await ev(`window.fly.lastVideo.size`)) / 1e3)} kB` : "no video");

  const trainer = spawn(PY, ["-c", `
import sys; sys.path.insert(0, ${JSON.stringify(ROOT)})
import flyenv
env = flyenv.DriveEnv(max_steps=60, viewer=${JSON.stringify(U)})
env.reset(seed=0)
for _ in range(60): env.step(flyenv.paired_action([1, 0]))
env.brain.viewer.flush()`], { stdio: "ignore" });
  const trainerDone = new Promise(r => trainer.on("exit", r));   // listen now: it may end before we wait
  const followed = await until(`window.fly.brain.remote === true && document.querySelector('#mode-seg .on').dataset.mode === 'car'`, 30000);
  check("the page follows a run streamed by flyenv", followed, await ev(`document.querySelector('#state-text').textContent`));
  await sleep(1500); await shot("watch");
  await trainerDone;

  await ev(`document.querySelector('#rec-refresh').click()`); await sleep(800);
  const rec = await ev(`document.querySelector('#rec-list').value`);
  await ev(`document.querySelector('#rec-replay').click()`);
  check("a recorded run replays on the page", await until(`(window.fly.brain.info?.remote || '').startsWith('Replay')`, 10000), rec);

  check("no error in the page console", errors.length === 0, errors.slice(0, 3).join(" | "));
  ws.close();
} catch (e) {
  check("the test ran to the end", false, e.message);
} finally {
  clearTimeout(watchdog);
  cleanup();
  const fails = results.filter(x => !x).length;
  console.log(`\n${results.length - fails}/${results.length} passed · screenshots in tests/out/`);
  process.exit(fails);
}
