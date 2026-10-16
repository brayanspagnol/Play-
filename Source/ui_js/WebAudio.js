// Audio output of the web build (linked with --pre-js).
// The VM thread writes SPU samples (44.1 kHz stereo int16) into a lock-free ring in wasm memory
// (SH_WebAudio.cpp). An AudioWorklet reads that ring directly from the shared wasm memory, so the
// main thread is not involved. When the emulator runs slower than real time the worklet time-stretches
// the audio (WSOLA: tempo change without pitch change) so it slows down with the game instead of
// cutting; at full speed the tempo settles at exactly 1.0 and samples pass through unchanged.

// Runs inside the AudioWorkletGlobalScope (its source is loaded through a Blob URL).
function playAudioWorkletMain() {
	const RATE = 44100;
	const HS = 512;          // synthesis hop (output frames per WSOLA step), 11.6 ms
	const N = 2 * HS;        // frame length, Hann window with 50% overlap
	const SEEK = 512;        // search range for the best splice point, +/- 11.6 ms (one period down to 86 Hz)
	const COARSE = 4;        // coarse search decimation
	const TARGET = 4410;     // input frames kept queued (ring + stretcher), 100 ms
	const MAX_FILL = 6 * TARGET;
	const MIN_TEMPO = 0.1, MAX_TEMPO = 1.5; // below 10% speed the producer is effectively stalled: let it gap
	const RATE_TAU = 0.5 * RATE;    // producer rate estimate time constant (frames)
	const CORRECT_LOW = 0.5 * RATE; // a fill below target is corrected over ~0.5 s (draining leads to a gap)
	const CORRECT_HIGH = 2 * RATE;  // a fill above target over ~2 s
	const SLEW = 2 / RATE;          // max tempo change: 2.0 per second
	const DEADBAND = 0.05;          // tempo within 5% of 1.0 -> exactly 1.0 (pure passthrough)
	const INCAP = 32768;
	const OUTCAP = 4 * HS;
	const FILL_BINS = 41, TEMPO_BINS = 16;
	// Stats layout (Float64Array shared with the main thread, see Module.audioStats)
	const S_OUT = 0, S_UNDERRUNS = 1, S_SILENT = 2, S_DROPS = 3, S_DROPPED = 4, S_TEMPO = 5, S_RATE = 6, S_FILL = 7,
		S_STRETCHED = 8, S_IN = 9, S_FLUSHES = 10, S_CAPTURE = 11, S_FILL_HIST = 12, S_TEMPO_HIST = S_FILL_HIST + FILL_BINS;

	const winUp = new Float32Array(HS), winDown = new Float32Array(HS);
	const ZERO = new Float32Array(HS);
	for(let i = 0; i < HS; i++) {
		const w = 0.5 - 0.5 * Math.cos(Math.PI * i / HS); // periodic Hann of length N: w[i] + w[i + HS] = 1
		winUp[i] = w;
		winDown[i] = 1 - w;
	}

	class PlayAudioProcessor extends AudioWorkletProcessor {
		constructor(options) {
			super();
			const o = options.processorOptions;
			this.hdr = new Uint32Array(o.memory, o.ring, 4);
			this.ringFrames = this.hdr[3];
			this.ring = new Int16Array(o.memory, o.ring + 16, this.ringFrames * 2);
			this.stats = new Float64Array(o.stats);
			this.inL = new Float32Array(INCAP); this.inR = new Float32Array(INCAP); this.inM = new Float32Array(INCAP);
			this.inBase = 0; this.inLen = 0;          // absolute input frame of inL[0], frames held
			this.outL = new Float32Array(OUTCAP); this.outR = new Float32Array(OUTCAP);
			this.outPos = 0; this.outLen = 0;
			this.olaL = new Float32Array(HS); this.olaR = new Float32Array(HS); // second half of the last frame
			this.running = false;   // false: prefilling (silence) until TARGET frames are queued
			this.started = false;   // produced sound at least once (silence before that is not a gap)
			this.pos = 0; this.nominal = 0;
			this.tempo = 1; this.rate = 1;
			this.lastWrite = Atomics.load(this.hdr, 0);
			this.lastWall = 0;
			this.capture = null; this.capturePos = 0;
			this.port.onmessage = (e) => {
				if(e.data.capture) { this.capture = new Int16Array(e.data.capture); this.capturePos = 0; this.stats[S_CAPTURE] = 0; }
			};
		}

		flush() { // discard everything queued (context was suspended or the thread stalled: that audio is stale)
			Atomics.store(this.hdr, 1, Atomics.load(this.hdr, 0));
			this.inBase += this.inLen; this.inLen = 0;
			this.outLen = 0;
			this.running = false;
			this.stats[S_FLUSHES]++;
		}

		pull() {
			const write = Atomics.load(this.hdr, 0);
			let read = Atomics.load(this.hdr, 1);
			if(this.inBase + this.inLen > this.keepFrom() + INCAP / 2 || this.inLen > INCAP / 2) this.compact();
			const n = Math.min((write - read) >>> 0, INCAP - this.inLen);
			const mask = this.ringFrames - 1, ring = this.ring;
			let j = this.inLen;
			for(let i = 0; i < n; i++, j++, read++) {
				const k = (read & mask) * 2;
				const l = ring[k] / 32768, r = ring[k + 1] / 32768;
				this.inL[j] = l; this.inR[j] = r; this.inM[j] = l + r;
			}
			this.inLen = j;
			Atomics.store(this.hdr, 1, read);
			this.stats[S_IN] += n;
			return write;
		}

		keepFrom() { return this.running ? Math.floor(Math.min(this.pos, this.nominal - SEEK)) : this.inBase; }

		compact() {
			const drop = Math.max(0, Math.min(this.keepFrom() - this.inBase, this.inLen));
			if(drop === 0) return;
			this.inL.copyWithin(0, drop, this.inLen); this.inR.copyWithin(0, drop, this.inLen); this.inM.copyWithin(0, drop, this.inLen);
			this.inBase += drop; this.inLen -= drop;
		}

		queued() { return this.inBase + this.inLen - (this.running ? this.nominal : this.inBase); }

		pushOut(l, r, i, gainArray) { // append HS frames: pending overlap + in[i..] * gain
			if(this.outPos + this.outLen + HS > OUTCAP) {
				this.outL.copyWithin(0, this.outPos, this.outPos + this.outLen); this.outR.copyWithin(0, this.outPos, this.outPos + this.outLen);
				this.outPos = 0;
			}
			const o = this.outPos + this.outLen;
			for(let k = 0; k < HS; k++) {
				const g = gainArray[k];
				this.outL[o + k] = this.olaL[k] + l[i + k] * g;
				this.outR[o + k] = this.olaR[k] + r[i + k] * g;
			}
			this.outLen += HS;
		}

		setOla(i) {
			const l = this.inL, r = this.inR;
			for(let k = 0; k < HS; k++) { this.olaL[k] = l[i + HS + k] * winDown[k]; this.olaR[k] = r[i + HS + k] * winDown[k]; }
		}

		start(at) { // first frame after a (re)start: fades in
			this.olaL.fill(0); this.olaR.fill(0);
			const i = at - this.inBase;
			this.pushOut(this.inL, this.inR, i, winUp);
			this.setOla(i);
			this.pos = at; this.nominal = at;
			this.tempo = Math.min(1, this.rate);
			this.running = true; this.started = true;
		}

		fadeOut() { // input ran out: play the tail of the last frame (it fades out), then silence until refilled
			this.pushOut(this.inL, this.inR, 0, ZERO);
			this.olaL.fill(0); this.olaR.fill(0);
			const used = Math.min(this.inLen, Math.max(0, Math.floor(this.nominal) - this.inBase)); // restart after what was played
			this.inL.copyWithin(0, used, this.inLen); this.inR.copyWithin(0, used, this.inLen); this.inM.copyWithin(0, used, this.inLen);
			this.inBase += used; this.inLen -= used;
			this.running = false;
		}

		step() { // produce HS output frames; false if there is not enough input
			if(!this.running) {
				if(this.inLen < TARGET || this.inLen < N + SEEK) return false;
				this.start(this.inBase + this.inLen - TARGET);
				return true;
			}
			const tempo = Math.abs(this.tempo - 1) < DEADBAND ? 1 : this.tempo;
			const next = this.nominal + HS * tempo;
			const center = Math.round(next);
			const end = this.inBase + this.inLen;
			if(end < center + SEEK + N) return false;
			const natural = this.pos + HS; // exact continuation of the last frame
			let best;
			if(tempo === 1 && Math.abs(natural - center) <= SEEK) {
				best = natural;
			} else {
				best = this.search(Math.max(center - SEEK, this.inBase), center + SEEK, natural);
				this.stats[S_STRETCHED] += HS;
			}
			const i = best - this.inBase;
			this.pushOut(this.inL, this.inR, i, winUp);
			this.setOla(i);
			this.pos = best;
			this.nominal = next;
			return true;
		}

		search(lo, hi, natural) { // splice point in [lo, hi] whose start best matches the continuation of the last frame
			const m = this.inM, t = natural - this.inBase, base = this.inBase;
			let best = lo, bestC = -Infinity;
			for(let p = lo; p <= hi; p += COARSE) {
				const o = p - base;
				let num = 0, den = 1e-9;
				for(let k = 0; k < HS; k += COARSE) { const b = m[o + k]; num += m[t + k] * b; den += b * b; }
				const c = num / Math.sqrt(den);
				if(c > bestC) { bestC = c; best = p; }
			}
			const flo = Math.max(lo, best - COARSE + 1), fhi = Math.min(hi, best + COARSE - 1);
			bestC = -Infinity;
			for(let p = flo; p <= fhi; p++) {
				const o = p - base;
				let num = 0, den = 1e-9;
				for(let k = 0; k < HS; k++) { const b = m[o + k]; num += m[t + k] * b; den += b * b; }
				const c = num / Math.sqrt(den);
				if(c > bestC) { bestC = c; best = p; }
			}
			return best;
		}

		process(inputs, outputs) {
			const outL = outputs[0][0], outR = outputs[0][1] || outputs[0][0];
			const frames = outL.length;
			const s = this.stats;
			const now = Date.now();
			if(this.lastWall && now - this.lastWall > 250) this.flush();
			this.lastWall = now;

			const write = this.pull();
			const produced = (write - this.lastWrite) >>> 0;
			this.lastWrite = write;
			this.rate += (produced / frames - this.rate) * (frames / RATE_TAU);

			let queued = this.queued();
			if(this.running && queued > MAX_FILL) { // far behind (should not happen with the tempo control): restart at the newest audio
				s[S_DROPS]++; s[S_DROPPED] += queued - TARGET;
				this.fadeOut();
				const drop = this.inLen - TARGET;
				this.inL.copyWithin(0, drop, this.inLen); this.inR.copyWithin(0, drop, this.inLen); this.inM.copyWithin(0, drop, this.inLen);
				this.inBase += drop; this.inLen = TARGET;
				queued = TARGET;
			}
			if(this.running) {
				const t = Math.min(MAX_TEMPO, Math.max(MIN_TEMPO, this.rate + (queued - TARGET) / (queued < TARGET ? CORRECT_LOW : CORRECT_HIGH)));
				this.tempo += Math.max(-SLEW * frames, Math.min(SLEW * frames, t - this.tempo));
			}

			while(this.outLen < frames && this.step()) {}
			let n = Math.min(frames, this.outLen);
			if(n < frames && this.running) { // underrun: fade out what we have
				s[S_UNDERRUNS]++;
				this.fadeOut();
				n = Math.min(frames, this.outLen);
			}
			outL.set(this.outL.subarray(this.outPos, this.outPos + n));
			outR.set(this.outR.subarray(this.outPos, this.outPos + n));
			this.outPos += n; this.outLen -= n;
			if(n < frames) {
				outL.fill(0, n); outR.fill(0, n);
				if(this.started) s[S_SILENT] += frames - n;
			}

			s[S_OUT] += frames;
			s[S_TEMPO] = this.running ? this.tempo : 0;
			s[S_RATE] = this.rate;
			s[S_FILL] = queued;
			s[S_FILL_HIST + Math.min(FILL_BINS - 1, Math.floor(queued / 441))]++;
			if(this.running) s[S_TEMPO_HIST + Math.min(TEMPO_BINS - 1, Math.floor((Math.abs(this.tempo - 1) < DEADBAND ? 1 : this.tempo) * 10))]++;

			const cap = this.capture;
			if(cap) {
				let c = this.capturePos;
				for(let i = 0; i < frames && c < cap.length; i++, c += 2) {
					cap[c] = Math.max(-32768, Math.min(32767, Math.round(outL[i] * 32768)));
					cap[c + 1] = Math.max(-32768, Math.min(32767, Math.round(outR[i] * 32768)));
				}
				this.capturePos = c;
				s[S_CAPTURE] = c / 2;
				if(c >= cap.length) this.capture = null;
			}
			return true;
		}
	}
	registerProcessor("play-audio", PlayAudioProcessor);
}

// Main thread side, called from SH_WebAudio.cpp (CSH_WebAudio::Start) with the ring address.
Module["webAudioStart"] = function(ring) {
	const stats = new SharedArrayBuffer(8 * 80);
	const statsView = new Float64Array(stats);
	const ctx = new AudioContext({ sampleRate: 44100, latencyHint: "interactive" });
	const gain = ctx.createGain(); // page volume (Module.setAudioVolume)
	gain.connect(ctx.destination);
	let node = null;
	const url = URL.createObjectURL(new Blob(["(" + playAudioWorkletMain.toString() + ")();"], { type: "text/javascript" }));
	ctx.audioWorklet.addModule(url).then(() => {
		node = new AudioWorkletNode(ctx, "play-audio", {
			numberOfInputs: 0, outputChannelCount: [2],
			processorOptions: { memory: HEAPU8.buffer, ring, stats },
		});
		node.connect(gain);
	}, (e) => console.error("Audio worklet failed to load:", e));

	// Autoplay policy: the context may only start after a user gesture; hidden tab: suspend.
	const resume = () => { if(ctx.state === "suspended" && !document.hidden) ctx.resume(); };
	for(const type of ["pointerdown", "keydown", "touchstart"]) window.addEventListener(type, resume, true);
	document.addEventListener("visibilitychange", () => { if(document.hidden) ctx.suspend(); else resume(); });
	resume();

	const hdr = () => new Uint32Array(HEAPU8.buffer, ring, 4);
	Module["audioContext"] = ctx;
	// Output volume, 0 (mute) to 1 (unchanged); a 15 ms ramp avoids clicks.
	Module["setAudioVolume"] = (volume) => gain.gain.setTargetAtTime(Math.max(0, Math.min(1, +volume || 0)), ctx.currentTime, 0.015);
	Module["getAudioVolume"] = () => gain.gain.value;
	// Output statistics: counters are cumulative since start, histograms count 128-frame render quanta.
	Module["audioStats"] = () => {
		const s = statsView;
		return {
			state: ctx.state, outFrames: s[0], underruns: s[1], silentFrames: s[2], drops: s[3], droppedFrames: s[4],
			tempo: s[5], rate: s[6], fillMs: s[7] / 44.1, stretchedFrames: s[8], inFrames: s[9], flushes: s[10],
			overflowFrames: hdr()[2], baseLatencyMs: ctx.baseLatency * 1000, outputLatencyMs: (ctx.outputLatency || 0) * 1000,
			fillHist10ms: Array.from(s.subarray(12, 12 + 41)), tempoHist: Array.from(s.subarray(53, 53 + 16)),
		};
	};
	// Records the next `seconds` of output (what the worklet sends to the speakers) for offline checks.
	let capture = null;
	Module["audioCapture"] = (seconds) => {
		capture = new SharedArrayBuffer(Math.round(seconds * 44100) * 4);
		node.port.postMessage({ capture });
	};
	Module["audioCaptureWav"] = () => { // WAV bytes once the capture is complete, else null
		if(!capture || statsView[11] * 4 < capture.byteLength) return null;
		const data = new Uint8Array(capture), wav = new Uint8Array(44 + data.length), v = new DataView(wav.buffer);
		const str = (o, t) => { for(let i = 0; i < t.length; i++) wav[o + i] = t.charCodeAt(i); };
		str(0, "RIFF"); v.setUint32(4, 36 + data.length, true); str(8, "WAVEfmt ");
		v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 2, true); v.setUint32(24, 44100, true);
		v.setUint32(28, 44100 * 4, true); v.setUint16(32, 4, true); v.setUint16(34, 16, true);
		str(36, "data"); v.setUint32(40, data.length, true);
		wav.set(data, 44);
		return wav;
	};
};
