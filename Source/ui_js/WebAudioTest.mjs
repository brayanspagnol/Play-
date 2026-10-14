// Offline check of the audio worklet in WebAudio.js (node Source/ui_js/WebAudioTest.mjs).
// Simulates the VM thread writing into the ring at a given speed and checks the worklet output.
import { readFileSync } from "node:fs";

const src = readFileSync(new URL("./WebAudio.js", import.meta.url), "utf8");
let Processor = null;
globalThis.AudioWorkletProcessor = class { constructor() { this.port = {}; } };
globalThis.registerProcessor = (name, cls) => { Processor = cls; };
new Function("Module", src + "\nplayAudioWorkletMain();")({});

const RING_FRAMES = 32768, RATE = 44100;
function run({ speed, seconds, chunk = 450, signal }) {
	const speedAt = typeof speed === "function" ? speed : () => speed;
	const memory = new SharedArrayBuffer(16 + RING_FRAMES * 4);
	const hdr = new Uint32Array(memory, 0, 4); hdr[3] = RING_FRAMES;
	const ring = new Int16Array(memory, 16);
	const stats = new SharedArrayBuffer(8 * 80);
	const p = new Processor({ processorOptions: { memory, ring: 0, stats } });
	const s = new Float64Array(stats);
	let produced = 0, debt = 0, t = 0;
	const out = new Float32Array(Math.round(seconds * RATE) * 2);
	const L = new Float32Array(128), R = new Float32Array(128);
	let wall = 1000;
	const realNow = Date.now; Date.now = () => wall;
	let cpuNs = 0n;
	for(let q = 0; q * 128 * 2 < out.length; q++) {
		debt += 128 * speedAt(q * 128 / RATE); // producer: `speed` input frames per output frame, written in chunks
		while(debt >= chunk) {
			for(let i = 0; i < chunk; i++, produced++) {
				const k = (produced % RING_FRAMES) * 2, v = Math.round(signal(produced) * 16000);
				ring[k] = v; ring[k + 1] = v;
			}
			Atomics.store(hdr, 0, produced); debt -= chunk;
		}
		wall += 128 / RATE * 1000;
		const t1 = process.hrtime.bigint();
		p.process([], [[L, R]]);
		cpuNs += process.hrtime.bigint() - t1;
		for(let i = 0; i < 128; i++) { out[(q * 128 + i) * 2] = L[i]; out[(q * 128 + i) * 2 + 1] = R[i]; }
	}
	Date.now = realNow;
	const cpuMs = Number(cpuNs) / 1e6;
	return { out, underruns: s[1], silent: s[2], tempo: s[5], rate: s[6], fillMs: s[7] / 44.1, cpuMsPerSec: cpuMs / seconds };
}

const tone = (f) => (i) => Math.sin(2 * Math.PI * f * i / RATE);
let failed = false;
const check = (name, ok, info) => { console.log((ok ? "ok   " : "FAIL ") + name, JSON.stringify(info)); if(!ok) failed = true; };

// Full speed: after startup the output must be the input, delayed, bit for bit (tempo exactly 1).
{
	const r = run({ speed: 1, seconds: 10, signal: tone(440) });
	const o = r.out;
	let maxJump = 0; // sine at 440 Hz, amplitude 16000/32768: max sample step 0.0306
	for(let i = 2 * RATE * 2; i < o.length - 2; i += 2) maxJump = Math.max(maxJump, Math.abs(o[i + 2] - o[i]));
	check("full speed, no gaps, passthrough", r.underruns === 0 && Math.abs(r.tempo - 1) < 0.02 && maxJump < 0.032, { ...r, out: undefined, maxJump });
}
// 45% speed: no gaps after the first start, tempo near 0.45, pitch kept (zero-crossing rate of the tone).
for(const speed of [0.45, 0.25, 0.8]) {
	const r = run({ speed, seconds: 20, signal: tone(220) });
	const o = r.out;
	let silentRun = 0, maxSilent = 0, crossings = 0, maxJump = 0;
	for(let i = 5 * RATE * 2; i < o.length; i += 2) {
		if(Math.abs(o[i]) < 1e-4) { silentRun++; maxSilent = Math.max(maxSilent, silentRun); } else silentRun = 0;
		if(o[i - 2] < 0 && o[i] >= 0) crossings++;
		maxJump = Math.max(maxJump, Math.abs(o[i] - o[i - 2]));
	}
	const freq = crossings / ((o.length / 2 - 5 * RATE) / RATE);
	check(`speed ${speed}: continuous, pitch kept`, r.underruns <= 1 && maxSilent < 10 && Math.abs(freq - 220) < 3 && Math.abs(r.tempo - speed) < 0.1 && maxJump < 0.017,
		{ ...r, out: undefined, maxSilent, maxJump, freq: +freq.toFixed(1) });
}
// Full speed with the VM's per-frame bursts (one PAL frame of audio every 20 ms): stays exactly at tempo 1.
{
	const r = run({ speed: 1, seconds: 10, chunk: 882, signal: tone(440) });
	check("full speed, bursty producer", r.underruns === 0 && r.tempo === 1 || Math.abs(r.tempo - 1) < 0.05, { ...r, out: undefined });
}
// Speed changes: 1.0 -> 0.5 -> 1.0, no gap after the first start except possibly one on the abrupt drop.
{
	const r = run({ speed: (t) => (t > 4 && t < 12) ? 0.5 : 1, seconds: 16, signal: tone(330) });
	check("speed steps 1 -> 0.5 -> 1", r.underruns <= 1, { ...r, out: undefined });
}
process.exit(failed ? 1 : 0);
