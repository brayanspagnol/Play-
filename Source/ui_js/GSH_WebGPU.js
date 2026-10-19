// GPU side of CGSH_WebGPU (GSH_WebGPU.cpp): WebGPU objects and WGSL shaders, all on the GS thread (a
// pthread worker; WebGPU objects can't move between workers). Linked with --pre-js, so it exists in
// every worker; only the GS thread calls it. The C++ side records a command list (OP_* below, see
// CGSH_WebGPU::OPCODE) and an upload arena in wasm memory and calls GsWgpu.execute() once per flush.
// GS local memory is a 4 MB storage buffer; every shader works on it directly, like the Vulkan backend
// (Source/gs/GSH_Vulkan), so frame buffers, textures and CLUTs of any format alias exactly.

var GsWgpu = {
	OP_UPLOAD_TO_MEMORY: 1,
	OP_UPLOAD_TO_SWIZZLE: 2,
	OP_MEMORY_TO_COPY: 3,
	OP_DISPATCH: 4,
	OP_PRESENT: 5,
	OP_READBACK: 6,
	OP_UPLOAD_TO_CLUT: 7,

	RAMSIZE: 0x400000,
	CLUT_BUFFER_SIZE: 1024 * 512 * 4, // CGSH_WebGPU::CLUT_SLOTS slots
	SWIZZLE_TABLES_SIZE: 45056 * 4,
	PARAMS_SIZE: 256,

	device: null,
	errorCount: 0,
	debug: 0,
	// Per-interval counters, logged every 2 s when debug is set (CGSH_WebGPU::SetDebug)
	stats: { frames: 0, executes: 0, batches: 0, prims: 0, uploadBytes: 0, gpuNs: 0, gpuPasses: 0, gpuExecutes: 0, jsMs: 0, since: 0 },
	MAX_TIMED_PASSES: 512,
	MAX_FRAMES_IN_FLIGHT: 2,
	// Draw shader variant: tile = workgroup side in pixels.
	// Bit i of a draw's feature mask (CGSH_WebGPU::DRAW_FEATURE_*) = override constant drawFeatures[i]
	drawFeatures: ["F_TEXTURE", "F_LINEAR", "F_MIPLOD", "F_CLUT", "F_MEMCOPY", "F_BLEND", "F_FOG", "F_ALPHATEST",
		"F_DATE", "F_DEPTHTEST", "F_DEPTHWRITE", "F_TRIANGLE", "F_SPRITE", "F_LINE"],
	drawConfig: { tile: 8 }, // tile must match CGSH_WebGPU::DRAW_TILE_SIZE
	framesInFlight: 0,
	waitingForGpu: false,

	heap() {
		return new Uint8Array(wasmMemory.buffer);
	},

	callWasm(fn, ...args) {
		callUserCallback(() => getWasmTableEntry(fn)(...args));
	},

	init(gs, resumeFn, readyFn, width, height) {
		this.gs = gs;
		this.resumeFn = resumeFn;
		this.canvas = new OffscreenCanvas(width, height);
		(async () => {
			let ok = false;
			try {
				const adapter = await navigator.gpu.requestAdapter();
				this.timestamps = adapter.features.has("timestamp-query");
				this.device = await adapter.requestDevice({ requiredFeatures: this.timestamps ? ["timestamp-query"] : [] });
				this.device.lost.then((info) => console.error("GSH_WebGPU: device lost:", info.message));
				this.device.onuncapturederror = (e) => {
					if (this.errorCount++ < 20) console.error("GSH_WebGPU:", e.error.message);
				};
				this.setup();
				ok = true;
			} catch (e) {
				console.error("GSH_WebGPU: initialization failed:", e);
			}
			this.callWasm(readyFn, gs, ok ? 1 : 0);
			this.callWasm(resumeFn, gs);
		})();
	},

	resize(width, height) {
		this.canvas.width = width;
		this.canvas.height = height;
	},

	setup() {
		const d = this.device;
		const B = GPUBufferUsage;
		this.memory = d.createBuffer({ label: "GS memory", size: this.RAMSIZE, usage: B.STORAGE | B.COPY_SRC | B.COPY_DST });
		this.memoryCopy = d.createBuffer({ label: "GS memory copy", size: this.RAMSIZE, usage: B.STORAGE | B.COPY_SRC | B.COPY_DST });
		this.clut = d.createBuffer({ label: "CLUT", size: this.CLUT_BUFFER_SIZE, usage: B.STORAGE | B.COPY_SRC | B.COPY_DST });
		this.swizzle = d.createBuffer({ label: "Swizzle tables", size: this.SWIZZLE_TABLES_SIZE, usage: B.STORAGE | B.COPY_SRC | B.COPY_DST });
		this.readback = d.createBuffer({ label: "Readback", size: this.RAMSIZE, usage: B.MAP_READ | B.COPY_DST });

		const S = GPUShaderStage;
		this.computeLayout = d.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: S.COMPUTE, buffer: { type: "storage" } },
				{ binding: 1, visibility: S.COMPUTE, buffer: { type: "read-only-storage" } },
				{ binding: 2, visibility: S.COMPUTE, buffer: { type: "storage" } },
				{ binding: 3, visibility: S.COMPUTE, buffer: { type: "read-only-storage" } },
				{ binding: 4, visibility: S.COMPUTE, buffer: { type: "read-only-storage" } },
				{ binding: 5, visibility: S.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
				{ binding: 6, visibility: S.COMPUTE, buffer: { type: "read-only-storage" } },
				{ binding: 7, visibility: S.COMPUTE, buffer: { type: "read-only-storage" } },
			],
		});
		this.presentLayout = d.createBindGroupLayout({
			entries: [
				{ binding: 0, visibility: S.FRAGMENT, buffer: { type: "read-only-storage" } },
				{ binding: 1, visibility: S.FRAGMENT, buffer: { type: "read-only-storage" } },
				{ binding: 2, visibility: S.VERTEX | S.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
			],
		});

		const computeModule = this.computeModule = d.createShaderModule({ label: "GS compute", code: this.wgslCompute() });
		const computePipelineLayout = this.computePipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
		this.drawPipelines = new Map();
		// Index = CGSH_WebGPU::PIPELINE
		this.computePipelines = ["xferHost", "xferLocal", "clutLoad", "draw"].map((entryPoint) =>
			d.createComputePipeline({ label: entryPoint, layout: computePipelineLayout, compute: { module: computeModule, entryPoint } }));

		this.format = navigator.gpu.getPreferredCanvasFormat();
		this.context = this.canvas.getContext("webgpu");
		this.context.configure({ device: d, format: this.format, alphaMode: "opaque" });
		const presentModule = d.createShaderModule({ label: "GS present", code: this.wgslPresent() });
		const presentPipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [this.presentLayout] });
		const blends = [
			undefined,
			{ color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" }, alpha: { srcFactor: "zero", dstFactor: "one" } },
			{ color: { srcFactor: "constant", dstFactor: "one-minus-constant" }, alpha: { srcFactor: "zero", dstFactor: "one" } },
		];
		this.presentPipelines = blends.map((blend) => d.createRenderPipeline({
			label: "present",
			layout: presentPipelineLayout,
			vertex: { module: presentModule, entryPoint: "vs" },
			fragment: { module: presentModule, entryPoint: "fs", targets: [{ format: this.format, blend }] },
			primitive: { topology: "triangle-strip" },
		}));

		if (this.timestamps) {
			this.querySet = d.createQuerySet({ type: "timestamp", count: this.MAX_TIMED_PASSES * 2 });
			this.queryResolve = d.createBuffer({ size: this.MAX_TIMED_PASSES * 16, usage: B.QUERY_RESOLVE | B.COPY_SRC });
			this.queryRead = d.createBuffer({ size: this.MAX_TIMED_PASSES * 16, usage: B.MAP_READ | B.COPY_DST });
			this.queryReadBusy = false;
		}
		this.upload = this.prims = this.states = null;
		this.ensureBuffers(16 << 20, 4 << 20, 1 << 20);
	},

	// Upload arena, primitives and states of a command list; grown (and the bind groups remade) on demand.
	// Draw pipeline specialized for a feature mask, compiled in the background; the general one meanwhile.
	drawPipeline(features) {
		const pipeline = this.drawPipelines.get(features);
		if (pipeline) return pipeline;
		if (pipeline === undefined && this.drawConfig.specialize !== false) {
			this.drawPipelines.set(features, null);
			const constants = Object.fromEntries(this.drawFeatures.map((f, i) => [f, (features >> i) & 1]));
			this.device.createComputePipelineAsync({ label: `draw ${features.toString(16)}`, layout: this.computePipelineLayout, compute: { module: this.computeModule, entryPoint: "draw", constants } })
				.then((p) => this.drawPipelines.set(features, p), (e) => console.error("GSH_WebGPU: draw pipeline:", e));
		}
		return this.computePipelines[3];
	},

	ensureBuffers(uploadSize, primsSize, statesSize) {
		const B = GPUBufferUsage;
		const grow = (name, size, usage) => {
			const buffer = this[name];
			if (buffer && buffer.size >= size) return false;
			let newSize = buffer ? buffer.size : (1 << 20);
			while (newSize < size) newSize *= 2;
			buffer?.destroy();
			this[name] = this.device.createBuffer({ label: name, size: newSize, usage });
			return true;
		};
		let changed = grow("upload", uploadSize, B.STORAGE | B.UNIFORM | B.COPY_SRC | B.COPY_DST);
		changed = grow("prims", primsSize, B.STORAGE | B.COPY_DST) || changed;
		changed = grow("states", statesSize, B.STORAGE | B.COPY_DST) || changed;
		if (!changed) return;
		this.computeBindGroup = this.device.createBindGroup({
			layout: this.computeLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.memory } },
				{ binding: 1, resource: { buffer: this.memoryCopy } },
				{ binding: 2, resource: { buffer: this.clut } },
				{ binding: 3, resource: { buffer: this.swizzle } },
				{ binding: 4, resource: { buffer: this.upload } },
				{ binding: 5, resource: { buffer: this.upload, size: this.PARAMS_SIZE } },
				{ binding: 6, resource: { buffer: this.prims } },
				{ binding: 7, resource: { buffer: this.states } },
			],
		});
		this.presentBindGroup = this.device.createBindGroup({
			layout: this.presentLayout,
			entries: [
				{ binding: 0, resource: { buffer: this.memory } },
				{ binding: 1, resource: { buffer: this.swizzle } },
				{ binding: 2, resource: { buffer: this.upload, size: this.PARAMS_SIZE } },
			],
		});
	},

	execute(commandsPtr, commandCount, uploadPtr, uploadSize, primsPtr, primsSize, statesPtr, statesSize) {
		const d = this.device;
		const heap = this.heap();
		const cmds = new Uint32Array(heap.buffer, commandsPtr, commandCount);
		this.captureExecute(heap, commandsPtr, commandCount, uploadPtr, uploadSize, primsPtr, primsSize, statesPtr, statesSize);
		this.ensureBuffers(uploadSize, primsSize, statesSize);
		if (uploadSize > 0) d.queue.writeBuffer(this.upload, 0, heap.buffer, uploadPtr, uploadSize);
		if (primsSize > 0) d.queue.writeBuffer(this.prims, 0, heap.buffer, primsPtr, primsSize);
		if (statesSize > 0) d.queue.writeBuffer(this.states, 0, heap.buffer, statesPtr, statesSize);
		const enc = d.createCommandEncoder();
		let pass = null;
		const endPass = () => {
			if (pass) pass.end();
			pass = null;
		};
		let presented = false;
		let readback = null;
		// GPU time of the compute passes (debug only; one readback in flight at a time)
		const timing = (this.debug || this.drawConfig.profile) && this.timestamps && !this.queryReadBusy;
		// drawConfig.profile (replay tool): one pass per dispatch, times in this.profile = [[pipeline, params offset, ns], ...]
		const passPerDispatch = timing && this.drawConfig.profile;
		const passInfo = [];
		let timedPasses = 0;
		const st = this.stats;
		st.executes++;
		const executeStart = performance.now();
		st.prims += primsSize / 112;
		st.uploadBytes += uploadSize + primsSize + statesSize;
		for (let i = 0; i < commandCount;) {
			switch (cmds[i]) {
			case this.OP_UPLOAD_TO_MEMORY:
				endPass();
				enc.copyBufferToBuffer(this.upload, cmds[i + 1], this.memory, cmds[i + 2], cmds[i + 3]);
				i += 4;
				break;
			case this.OP_UPLOAD_TO_SWIZZLE:
				endPass();
				enc.copyBufferToBuffer(this.upload, cmds[i + 1], this.swizzle, 0, cmds[i + 2]);
				i += 3;
				break;
			case this.OP_UPLOAD_TO_CLUT:
				endPass();
				enc.copyBufferToBuffer(this.upload, cmds[i + 1], this.clut, cmds[i + 2], cmds[i + 3]);
				i += 4;
				break;
			case this.OP_MEMORY_TO_COPY:
				endPass();
				enc.copyBufferToBuffer(this.memory, cmds[i + 1], this.memoryCopy, cmds[i + 1], cmds[i + 2]);
				i += 3;
				break;
			case this.OP_DISPATCH:
				if (passPerDispatch) endPass();
				if (!pass) {
					passInfo.push([cmds[i + 1], cmds[i + 2]]);
					if (timing && timedPasses < this.MAX_TIMED_PASSES) {
						pass = enc.beginComputePass({ timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: timedPasses * 2, endOfPassWriteIndex: timedPasses * 2 + 1 } });
						timedPasses++;
					} else {
						pass = enc.beginComputePass();
					}
				}
				if ((cmds[i + 1] & 0xFF) === 3) {
					st.batches++;
					if (this.debug & 4) { i += 6; break; } // debug: skip draws (CPU-bound speed)
				}
				pass.setPipeline((cmds[i + 1] & 0xFF) === 3 ? this.drawPipeline(cmds[i + 1] >>> 8) : this.computePipelines[cmds[i + 1]]);
				pass.setBindGroup(0, this.computeBindGroup, [cmds[i + 2]]);
				pass.dispatchWorkgroups(cmds[i + 3], cmds[i + 4], cmds[i + 5]);
				i += 6;
				break;
			case this.OP_PRESENT: {
				endPass();
				const paramsOffset = cmds[i + 1], layerCount = cmds[i + 2];
				const rp = enc.beginRenderPass({
					colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: "store" }],
				});
				const params = new Uint32Array(heap.buffer, uploadPtr, (uploadSize >> 2));
				const paramsF = new Float32Array(heap.buffer, uploadPtr, (uploadSize >> 2));
				for (let l = 0; l < layerCount; l++) {
					const p = (paramsOffset + l * this.PARAMS_SIZE) >> 2;
					const blendMode = params[p + 9];
					const [vx, vy, vw, vh] = [params[p + 11] | 0, params[p + 12] | 0, params[p + 13] | 0, params[p + 14] | 0];
					if (vw <= 0 || vh <= 0) continue;
					rp.setViewport(vx, vy, vw, vh, 0, 1);
					rp.setPipeline(this.presentPipelines[blendMode]);
					if (blendMode === 2) {
						const a = paramsF[p + 10];
						rp.setBlendConstant({ r: a, g: a, b: a, a });
					}
					rp.setBindGroup(0, this.presentBindGroup, [paramsOffset + l * this.PARAMS_SIZE]);
					rp.draw(4);
				}
				rp.end();
				presented = true;
				i += 3;
				break;
			}
			case this.OP_READBACK:
				endPass();
				readback = { offset: cmds[i + 1], size: cmds[i + 2], ptr: cmds[i + 3] };
				enc.copyBufferToBuffer(this.memory, readback.offset, this.readback, 0, readback.size);
				i += 4;
				break;
			default:
				throw new Error(`GSH_WebGPU: bad command ${cmds[i]} at ${i}`);
			}
		}
		endPass();
		if (timedPasses) {
			enc.resolveQuerySet(this.querySet, 0, timedPasses * 2, this.queryResolve, 0);
			enc.copyBufferToBuffer(this.queryResolve, 0, this.queryRead, 0, timedPasses * 16);
		}
		d.queue.submit([enc.finish()]);
		if (timedPasses) {
			this.queryReadBusy = true;
			this.queryRead.mapAsync(GPUMapMode.READ, 0, timedPasses * 16).then(() => {
				const t = new BigInt64Array(this.queryRead.getMappedRange(0, timedPasses * 16));
				for (let p = 0; p < timedPasses; p++) st.gpuNs += Number(t[p * 2 + 1] - t[p * 2]);
				if (passPerDispatch) this.profile = passInfo.slice(0, timedPasses).map((x, p) => [...x, Number(t[p * 2 + 1] - t[p * 2])]);
				st.gpuPasses += timedPasses;
				st.gpuExecutes++;
				this.queryRead.unmap();
				this.queryReadBusy = false;
			}, () => { this.queryReadBusy = false; });
		}
		let gpuBusy = 0;
		if (presented) {
			st.frames++;
			// Frame pacing: the GS thread waits (suspended) while MAX_FRAMES_IN_FLIGHT frames are queued.
			this.framesInFlight++;
			d.queue.onSubmittedWorkDone().then(() => {
				this.framesInFlight--;
				if (this.waitingForGpu && this.framesInFlight < this.MAX_FRAMES_IN_FLIGHT) {
					this.waitingForGpu = false;
					this.callWasm(this.resumeFn, this.gs);
				}
			});
			if (this.framesInFlight >= this.MAX_FRAMES_IN_FLIGHT && !readback) {
				this.waitingForGpu = true;
				gpuBusy = 1;
			}
		}
		const now = performance.now();
		st.jsMs += now - executeStart;
		if (this.debug && now - st.since > 2000) {
			const f = Math.max(st.frames, 1);
			console.log(`GSH_WebGPU: ${(st.frames * 1000 / (now - st.since)).toFixed(1)} frames/s, per frame: ${(st.executes / f).toFixed(1)} submits, ${(st.batches / f).toFixed(0)} batches, ${(st.prims / f).toFixed(0)} prims, ${(st.uploadBytes / f / 1024).toFixed(0)} KB, JS ${(st.jsMs / f).toFixed(1)} ms, compute GPU ${(st.gpuNs / 1e6 / Math.max(st.gpuExecutes, 1)).toFixed(2)} ms per submit (${st.gpuExecutes} sampled)`);
			Object.assign(st, { frames: 0, executes: 0, batches: 0, prims: 0, uploadBytes: 0, gpuNs: 0, gpuPasses: 0, gpuExecutes: 0, jsMs: 0, since: now });
		}
		if (presented && this.capture?.submits && !this.capture.finishing) this.finishCapture();
		this.lastExecutePresented = presented;
		if (presented) {
			// Only complete frames ever reach the page (same path as GSH_OpenGLJs).
			const bitmap = this.canvas.transferToImageBitmap();
			postMessage({ cmd: "callHandler", handler: "onGsFrame", args: [bitmap] }, [bitmap]);
		}
		if (readback) {
			// CGSH_WebGPU::Readback suspended the GS thread: resume it once the data is in wasm memory.
			this.readback.mapAsync(GPUMapMode.READ, 0, readback.size).then(() => {
				this.heap().set(new Uint8Array(this.readback.getMappedRange(0, readback.size)), readback.ptr);
				this.readback.unmap();
				this.callWasm(this.resumeFn, this.gs);
			}, (e) => {
				console.error("GSH_WebGPU: readback failed:", e);
				this.callWasm(this.resumeFn, this.gs);
			});
		}
		return gpuBusy;
	},

	// Frame capture (debug bit 2, see CGSH_WebGPU::SetDebug): GPU memory/CLUT/copy buffers at the start of the
	// next frame plus every command list until its present, sent to the page as Module.onGsCapture(object)
	// for offline replay (tools: notes/webgpu.md).
	captureExecute(heap, commandsPtr, commandCount, uploadPtr, uploadSize, primsPtr, primsSize, statesPtr, statesSize) {
		if (!(this.debug & 2)) return;
		if (!this.capture) {
			if (!this.lastExecutePresented) return;
			const B = GPUBufferUsage;
			const snap = (src, size) => {
				const dst = this.device.createBuffer({ size, usage: B.MAP_READ | B.COPY_DST });
				const enc = this.device.createCommandEncoder();
				enc.copyBufferToBuffer(src, 0, dst, 0, size);
				this.device.queue.submit([enc.finish()]);
				return dst;
			};
			this.capture = { memory: snap(this.memory, this.RAMSIZE), memoryCopy: snap(this.memoryCopy, this.RAMSIZE), clut: snap(this.clut, this.CLUT_BUFFER_SIZE), swizzle: snap(this.swizzle, this.SWIZZLE_TABLES_SIZE), submits: [] };
		}
		const copy = (ptr, size) => heap.slice(ptr, ptr + size);
		this.capture.submits.push({ commands: copy(commandsPtr, commandCount * 4), upload: copy(uploadPtr, uploadSize), prims: copy(primsPtr, primsSize), states: copy(statesPtr, statesSize) });
	},

	finishCapture() {
		const c = this.capture;
		const prims = c.submits.reduce((n, sub) => n + sub.prims.length / 112, 0);
		if (prims < 1000) {
			// Nearly empty frame (e.g. a flip without drawing): try the next one
			for (const b of [c.memory, c.memoryCopy, c.clut, c.swizzle]) b.destroy();
			this.capture = null;
			return;
		}
		this.debug &= ~2;
		c.finishing = true;
		const read = async (b) => { await b.mapAsync(GPUMapMode.READ); const r = b.getMappedRange().slice(0); b.destroy(); return r; };
		Promise.all([read(c.memory), read(c.memoryCopy), read(c.clut), read(c.swizzle)]).then(([memory, memoryCopy, clut, swizzle]) => {
			const out = { memory, memoryCopy, clut, swizzle, submits: c.submits };
			postMessage({ cmd: "callHandler", handler: "onGsCapture", args: [out] });
			this.capture = null;
		});
	},

	// GS memory access helpers, shared by the compute and present modules. load(i) reads word i.
	wgslMemory(load) {
		return `
const PSMCT32 = 0x00u;
const PSMCT24 = 0x01u;
const PSMCT16 = 0x02u;
const PSMCT24_UNK = 0x09u;
const PSMCT16S = 0x0Au;
const PSMT8 = 0x13u;
const PSMT4 = 0x14u;
const PSMT8H = 0x1Bu;
const PSMT4HL = 0x24u;
const PSMT4HH = 0x2Cu;
const PSMZ32 = 0x30u;
const PSMZ24 = 0x31u;
const PSMZ16 = 0x32u;
const PSMZ16S = 0x3Au;

const WORDMASK = 0xFFFFFu; // RAMSIZE / 4 - 1

fn loadWord(i : u32) -> u32 { return ${load}; }
fn read32(a : u32) -> u32 { return loadWord(a >> 2u); }
fn read24(a : u32) -> u32 { return read32(a) & 0xFFFFFFu; }
fn read16(a : u32) -> u32 { return (read32(a) >> ((a & 2u) * 8u)) & 0xFFFFu; }
fn read8(a : u32) -> u32 { return (read32(a) >> ((a & 3u) * 8u)) & 0xFFu; }
fn read4(nibAddress : u32) -> u32 { return (loadWord(nibAddress >> 3u) >> ((nibAddress & 7u) * 4u)) & 0xFu; }

// Page swizzle table of a format: (offset in swz, log2 page width, log2 page height). Same order as
// CGSH_WebGPU::UploadSwizzleTables; formats sharing a layout share a table (GSH_VulkanContext::GetSwizzleTable).
fn swizzleInfo(psm : u32) -> vec3<u32> {
	switch(psm) {
		case 0x30u, 0x31u: { return vec3<u32>(2048u, 6u, 5u); }  // PSMZ32, PSMZ24
		case 0x02u: { return vec3<u32>(4096u, 6u, 6u); }         // PSMCT16
		case 0x0Au: { return vec3<u32>(8192u, 6u, 6u); }         // PSMCT16S
		case 0x32u: { return vec3<u32>(12288u, 6u, 6u); }        // PSMZ16
		case 0x3Au: { return vec3<u32>(16384u, 6u, 6u); }        // PSMZ16S
		case 0x13u: { return vec3<u32>(20480u, 7u, 6u); }        // PSMT8
		case 0x14u: { return vec3<u32>(28672u, 7u, 7u); }        // PSMT4
		default: { return vec3<u32>(0u, 6u, 5u); }               // PSMCT32 and the formats stored like it
	}
}

// Byte address of pixel (x, y) (nibble address for PSMT4), as CMemoryUtils::GetPixelAddress.
fn pixelAddress(psm : u32, bufAddress : u32, bufWidth : u32, x : u32, y : u32) -> u32 {
	let info = swizzleInfo(psm);
	let pageWidth = 1u << info.y;
	let pageHeight = 1u << info.z;
	let pageNum = (x >> info.y) + ((y >> info.z) * bufWidth) / pageWidth;
	let pageOffset = swz[info.x + ((y & (pageHeight - 1u)) << info.y) + (x & (pageWidth - 1u))];
	if(psm == PSMT4) {
		return (bufAddress + pageNum * 8192u) * 2u + pageOffset;
	}
	return bufAddress + pageNum * 8192u + pageOffset;
}

// Raw pixel value of any format (CLUT index for indexed formats).
fn readPixel(psm : u32, bufAddress : u32, bufWidth : u32, x : u32, y : u32) -> u32 {
	let a = pixelAddress(psm, bufAddress, bufWidth, x, y);
	switch(psm) {
		case 0x01u, 0x09u, 0x31u: { return read24(a); }               // PSMCT24, PSMCT24_UNK, PSMZ24
		case 0x02u, 0x0Au, 0x32u, 0x3Au: { return read16(a); }        // 16-bit formats
		case 0x13u: { return read8(a); }                              // PSMT8
		case 0x14u: { return read4(a); }                              // PSMT4
		case 0x1Bu: { return read8(a + 3u); }                         // PSMT8H
		case 0x24u: { return read4((a + 3u) * 2u); }                  // PSMT4HL
		case 0x2Cu: { return read4(((a + 3u) * 2u) | 1u); }           // PSMT4HH
		default: { return read32(a); }
	}
}

fn psm32ToColor(c : u32) -> vec4<f32> {
	return vec4<f32>(f32(c & 0xFFu), f32((c >> 8u) & 0xFFu), f32((c >> 16u) & 0xFFu), f32(c >> 24u)) / 255.0;
}

fn psm16ToColor(c : u32) -> vec4<f32> {
	return vec4<f32>(f32(c & 0x1Fu) / 31.0, f32((c >> 5u) & 0x1Fu) / 31.0, f32((c >> 10u) & 0x1Fu) / 31.0, f32((c >> 15u) & 1u));
}
`;
	},

	wgslCompute() {
		return `
@group(0) @binding(0) var<storage, read_write> mem : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> memCopy : array<u32>;
@group(0) @binding(2) var<storage, read_write> clut : array<u32>;
@group(0) @binding(3) var<storage, read> swz : array<u32>;
@group(0) @binding(4) var<storage, read> upload : array<u32>;
@group(0) @binding(5) var<uniform> P : array<vec4<u32>, 16>;
${this.wgslMemory("atomicLoad(&mem[i & WORDMASK])")}

fn param(i : u32) -> u32 { return P[i >> 2u][i & 3u]; }

fn readCopy32(a : u32) -> u32 { return memCopy[(a >> 2u) & WORDMASK]; }

fn readPixelFrom(fromCopy : bool, psm : u32, bufAddress : u32, bufWidth : u32, x : u32, y : u32) -> u32 {
	if(!fromCopy) { return readPixel(psm, bufAddress, bufWidth, x, y); }
	let a = pixelAddress(psm, bufAddress, bufWidth, x, y);
	switch(psm) {
		case 0x01u, 0x09u, 0x31u: { return readCopy32(a) & 0xFFFFFFu; }
		case 0x02u, 0x0Au, 0x32u, 0x3Au: { return (readCopy32(a) >> ((a & 2u) * 8u)) & 0xFFFFu; }
		case 0x13u: { return (readCopy32(a) >> ((a & 3u) * 8u)) & 0xFFu; }
		case 0x14u: { return (memCopy[(a >> 3u) & WORDMASK] >> ((a & 7u) * 4u)) & 0xFu; }
		case 0x1Bu: { return readCopy32(a) >> 24u; }
		case 0x24u: { return (readCopy32(a) >> 24u) & 0xFu; }
		case 0x2Cu: { return readCopy32(a) >> 28u; }
		default: { return readCopy32(a); }
	}
}

fn writeMasked(word : u32, mask : u32, value : u32) {
	let i = word & WORDMASK;
	atomicAnd(&mem[i], ~mask);
	atomicOr(&mem[i], value & mask);
}

fn write32(a : u32, v : u32) { atomicStore(&mem[(a >> 2u) & WORDMASK], v); }
fn write24(a : u32, v : u32) { writeMasked(a >> 2u, 0xFFFFFFu, v); }
fn write16(a : u32, v : u32) { let s = (a & 2u) * 8u; writeMasked(a >> 2u, 0xFFFFu << s, v << s); }
fn write8(a : u32, v : u32) { let s = (a & 3u) * 8u; writeMasked(a >> 2u, 0xFFu << s, v << s); }
fn write4(n : u32, v : u32) { let s = (n & 7u) * 4u; writeMasked(n >> 3u, 0xFu << s, v << s); }

fn writePixel(psm : u32, bufAddress : u32, bufWidth : u32, x : u32, y : u32, v : u32) {
	let a = pixelAddress(psm, bufAddress, bufWidth, x, y);
	switch(psm) {
		case 0x01u, 0x09u, 0x31u: { write24(a, v); }
		case 0x02u, 0x0Au, 0x32u, 0x3Au: { write16(a, v); }
		case 0x13u: { write8(a, v); }
		case 0x14u: { write4(a, v); }
		case 0x1Bu: { write8(a + 3u, v); }
		case 0x24u: { write4((a + 3u) * 2u, v); }
		case 0x2Cu: { write4(((a + 3u) * 2u) | 1u, v); }
		default: { write32(a, v); }
	}
}

fn uploadByte(byteOffset : u32) -> u32 { return (upload[byteOffset >> 2u] >> ((byteOffset & 3u) * 8u)) & 0xFFu; }

// Host -> local transfers (GSH_VulkanTransferHost), several per dispatch: P = count, table (XFER_WORDS words per
// transfer: psm, bufAddress, bufWidth, rrw, dsax, dsay, data offset (words), pixel count, first workgroup).
// One invocation per pixel of the stream.
@compute @workgroup_size(256)
fn xferHost(@builtin(workgroup_id) wid : vec3<u32>, @builtin(local_invocation_index) li : u32) {
	let count = param(0u);
	let table = param(1u);
	var t = 0u;
	while((t + 1u < count) && (upload[table + (t + 1u) * 9u + 8u] <= wid.x)) { t++; }
	let x = table + t * 9u;
	let psm = upload[x + 0u];
	let bufAddress = upload[x + 1u];
	let bufWidth = upload[x + 2u];
	let rrw = upload[x + 3u];
	let dsax = upload[x + 4u];
	let dsay = upload[x + 5u];
	let dataOffset = upload[x + 6u];
	let pixelCount = upload[x + 7u];
	let i = (wid.x - upload[x + 8u]) * 256u + li;
	if(i >= pixelCount) { return; }
	let px = ((i % rrw) + dsax) % 2048u;
	let py = ((i / rrw) + dsay) % 2048u;
	var value = 0u;
	switch(psm) {
		case 0x01u, 0x09u, 0x31u: {
			let b = dataOffset * 4u + i * 3u;
			value = uploadByte(b) | (uploadByte(b + 1u) << 8u) | (uploadByte(b + 2u) << 16u);
		}
		case 0x02u, 0x0Au, 0x32u, 0x3Au: { value = (upload[dataOffset + (i >> 1u)] >> ((i & 1u) * 16u)) & 0xFFFFu; }
		case 0x13u, 0x1Bu: { value = uploadByte(dataOffset * 4u + i); }
		case 0x14u, 0x24u, 0x2Cu: { value = (upload[dataOffset + (i >> 3u)] >> ((i & 7u) * 4u)) & 0xFu; }
		default: { value = upload[dataOffset + i]; }
	}
	writePixel(psm, bufAddress, bufWidth, px, py, value);
}

// Local -> local transfer (GSH_VulkanTransferLocal).
@compute @workgroup_size(16, 16)
fn xferLocal(@builtin(global_invocation_id) id : vec3<u32>) {
	let rrw = param(10u);
	let rrh = param(11u);
	if((id.x >= rrw) || (id.y >= rrh)) { return; }
	let srcX = (id.x + param(6u)) % 2048u;
	let srcY = (id.y + param(7u)) % 2048u;
	let dstX = (id.x + param(8u)) % 2048u;
	let dstY = (id.y + param(9u)) % 2048u;
	let pixel = readPixelFrom(param(12u) != 0u, param(0u), param(2u), param(3u), srcX, srcY);
	writePixel(param(1u), param(4u), param(5u), dstX, dstY, pixel);
}

// CLUT load into one 512-entry slot of the CLUT buffer (GSH_VulkanClutLoad).
@compute @workgroup_size(256)
fn clutLoad(@builtin(local_invocation_index) li : u32) {
	let idx8 = param(0u) != 0u;
	let csm = param(1u);
	let cpsm = param(2u);
	let clutBufPtr = param(3u);
	let clutBufWidth = param(4u);
	let csa = param(5u);
	let base = param(8u);
	var x = 0u;
	var y = 0u;
	var index = 0u;
	if(csm == 0u) {
		if(idx8) {
			x = li % 16u;
			y = li / 16u;
			index = li;
			index = (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
		} else {
			if(li >= 16u) { return; }
			x = li % 8u;
			y = li / 8u;
			index = li + csa * 16u;
		}
	} else {
		if(!idx8 && (li >= 16u)) { return; }
		x = param(6u) + li;
		y = param(7u);
		index = li;
	}
	if((cpsm == PSMCT16) || (cpsm == PSMCT16S)) {
		clut[base + (index & 511u)] = readPixel(cpsm, clutBufPtr, clutBufWidth, x, y);
	} else {
		let color = read32(pixelAddress(cpsm, clutBufPtr, clutBufWidth, x, y));
		// 32-bit CLUTs are kept as whole colors in the first 256 entries (one read per texel): the GS's two 16-bit
		// halves (index, index + 256) of a load at index >= 256 swap places.
		clut[base + (index & 255u)] = select(color, (color >> 16u) | (color << 16u), (index & 256u) != 0u);
	}
}

// Draws (see notes/webgpu.md): compute rasterizer working on GS memory, the per-pixel part follows
// GSH_VulkanDrawDesktop::CreateFragmentShader / GSH_VulkanDrawUtils.
// Primitives and states as vec4s: one load per 4 words (separately bounds-checked scalar loads aren't merged)
@group(0) @binding(6) var<storage, read> prims : array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> states : array<vec4<u32>>;

const PRIM_WORDS = 28u;
const STATE_WORDS = 40u;

const PRIM_TRIANGLE = 0u;
const PRIM_SPRITE = 1u;
const PRIM_LINE = 2u;
const PRIM_POINT = 3u;

const TILE_SIZE = ${this.drawConfig.tile}u;   // tile = workgroup = TILE_SIZE x TILE_SIZE pixels
const GROUP_SIZE = ${this.drawConfig.tile ** 2}u;

// Features used by a batch (CGSH_WebGPU::DRAW_FEATURE_*): pipelines are specialized per combination so that
// the unused paths are compiled out (the shader is big; registers decide the speed). All on = any batch.
${this.drawFeatures.map((f) => `override ${f} : bool = true;`).join("\n")}

// The tile's primitives of the current chunk and their coverage of the tile (bit y * TILE_SIZE + x).
var<workgroup> chunkPrims : array<u32, ${this.drawConfig.tile ** 2}>;
var<workgroup> chunkMasks : array<array<u32, ${this.drawConfig.tile ** 2 / 32}>, ${this.drawConfig.tile ** 2}>;

// Word k of primitive p
fn pw(p : u32, k : u32) -> u32 {
	return prims[p * (PRIM_WORDS / 4u) + k / 4u][k % 4u];
}

// Frame/depth buffer value of the invocation's pixel, kept for the whole batch: within a batch only this
// invocation touches it (textures overlapping the batch's buffers end the batch, see CGSH_WebGPU).
struct PixelCache {
	fbAddress : u32,
	zbAddress : u32,
	fb : u32,
	z : u32,
	flags : u32,   // 1: fb loaded, 2: z loaded, 4: fb written, 8: z written
};

// Word i of the state at s (= state index * STATE_WORDS / 4)
fn sw(s : u32, i : u32) -> u32 { return states[s + i / 4u][i % 4u]; }
fn bits(v : u32, first : u32, count : u32) -> u32 { return extractBits(v, first, count); }

// Edge function (b - a) x (s - a) in 1/16 pixel units. Exact when it fits in 31 bits; beyond that only
// its sign matters and the float estimate is far from 0.
fn edgeFunction(dx : i32, dy : i32, sx : i32, sy : i32) -> i32 {
	let approx = f32(dx) * f32(sy) - f32(dy) * f32(sx);
	if(abs(approx) > 1073741824.0) { return select(-1, 1, approx > 0.0); }
	return dx * sy - dy * sx;
}

// Top-left fill rule (y down, positive orientation).
fn edgeCovers(e : i32, dx : i32, dy : i32) -> bool {
	return (e > 0) || ((e == 0) && ((dy < 0) || ((dy == 0) && (dx > 0))));
}

fn unpackColor(c : u32) -> vec4<f32> {
	return vec4<f32>(f32(c & 0xFFu), f32((c >> 8u) & 0xFFu), f32((c >> 16u) & 0xFFu), f32(c >> 24u));
}

fn readTexWord(fromCopy : bool, i : u32) -> u32 {
	if(F_MEMCOPY && fromCopy) { return memCopy[i & WORDMASK]; }
	return atomicLoad(&mem[i & WORDMASK]);
}

// Texel (raw value) of a texture format at (x, y), from GS memory or from the copy (texture = frame buffer).
fn readTexel(fromCopy : bool, psm : u32, bufAddress : u32, bufWidth : u32, x : u32, y : u32) -> u32 {
	let a = pixelAddress(psm, bufAddress, bufWidth, x, y);
	switch(psm) {
		case 0x01u, 0x09u, 0x31u: { return readTexWord(fromCopy, a >> 2u) & 0xFFFFFFu; }
		case 0x02u, 0x0Au, 0x32u, 0x3Au: { return (readTexWord(fromCopy, a >> 2u) >> ((a & 2u) * 8u)) & 0xFFFFu; }
		case 0x13u: { return (readTexWord(fromCopy, a >> 2u) >> ((a & 3u) * 8u)) & 0xFFu; }
		case 0x14u: { return (readTexWord(fromCopy, a >> 3u) >> ((a & 7u) * 4u)) & 0xFu; }
		case 0x1Bu: { return readTexWord(fromCopy, a >> 2u) >> 24u; }
		case 0x24u: { return (readTexWord(fromCopy, a >> 2u) >> 24u) & 0xFu; }
		case 0x2Cu: { return readTexWord(fromCopy, a >> 2u) >> 28u; }
		default: { return readTexWord(fromCopy, a >> 2u); }
	}
}

// 16-bit color with alpha expansion (TEXA): the A bit selects TA1/TA0; AEM makes black with A = 0 transparent.
fn expand16(c : u32, tcc : bool, aem : bool, ta0 : u32, ta1 : u32) -> vec4<f32> {
	let rgb = vec3<f32>(f32((c & 0x1Fu) << 3u), f32(((c >> 5u) & 0x1Fu) << 3u), f32(((c >> 10u) & 0x1Fu) << 3u));
	var a = 0u;
	if((c & 0x8000u) != 0u) { a = ta1; } else if(!aem || ((c & 0x7FFFu) != 0u)) { a = ta0; }
	return vec4<f32>(rgb, f32(a));
}

fn texelColor(s : u32, raw : u32) -> vec4<f32> {
	let psm = sw(s, 2u);
	let flags0 = sw(s, 0u);
	let tcc = bits(flags0, 1u, 1u) != 0u;
	let aem = bits(flags0, 2u, 1u) != 0u;
	let ta0 = sw(s, 10u);
	let ta1 = sw(s, 11u);
	switch(psm) {
		case 0x00u, 0x30u: { return unpackColor(raw); }
		case 0x01u, 0x09u, 0x31u: {
			var a = ta0;
			if(aem && ((raw & 0xFFFFFFu) == 0u)) { a = 0u; }
			return vec4<f32>(unpackColor(raw).rgb, f32(a));
		}
		case 0x02u, 0x0Au, 0x32u, 0x3Au: { return expand16(raw, tcc, aem, ta0, ta1); }
		default: {
			if(!F_CLUT) { return vec4<f32>(0.0); }
			// Indexed: CLUT lookup (GSH_VulkanDrawUtils GetClutColor)
			var index = raw;
			if((psm == PSMT4) || (psm == PSMT4HL) || (psm == PSMT4HH)) { index = raw + sw(s, 9u); }
			let clutBase = sw(s, 19u);
			let cpsm = sw(s, 3u);
			if((cpsm == PSMCT16) || (cpsm == PSMCT16S)) {
				return expand16(clut[clutBase + (index & 511u)], tcc, aem, ta0, ta1);
			}
			return unpackColor(clut[clutBase + (index & 0xFFu)]);
		}
	}
}

fn clampTexCoord(mode : u32, c : i32, size : i32, minValue : i32, maxValue : i32) -> i32 {
	switch(mode) {
		case 1u: { return clamp(c, 0, size - 1); }
		case 2u: { return clamp(c, minValue, maxValue); }
		case 3u: { return (c & minValue) | maxValue; }
		default: { return c & (size - 1); }
	}
}

fn fetchTexel(s : u32, iuv : vec2<i32>, level : u32, bufAddress : u32, bufWidth : u32) -> vec4<f32> {
	let flags0 = sw(s, 0u);
	let size = vec2<i32>(i32(sw(s, 6u)), i32(sw(s, 7u)));
	let u = clampTexCoord(bits(flags0, 7u, 2u), iuv.x, size.x, i32(sw(s, 12u)), i32(sw(s, 14u)));
	let v = clampTexCoord(bits(flags0, 9u, 2u), iuv.y, size.y, i32(sw(s, 13u)), i32(sw(s, 15u)));
	let fromCopy = bits(sw(s, 1u), 4u, 1u) != 0u;
	let raw = readTexel(fromCopy, sw(s, 2u), bufAddress, bufWidth, u32(u) >> level, u32(v) >> level);
	return texelColor(s, raw);
}

fn sampleTexture(s : u32, st : vec2<f32>, q : f32) -> vec4<f32> {
	let flags0 = sw(s, 0u);
	let size = vec2<f32>(f32(sw(s, 6u)), f32(sw(s, 7u)));
	var texPos : vec2<f32>;
	if(bits(sw(s, 1u), 5u, 1u) != 0u) {
		texPos = st;                 // UV: already in texels
	} else {
		texPos = (st / q) * size;    // STQ
	}
	var level = sw(s, 8u);
	var bufAddress = sw(s, 4u);
	var bufWidth = sw(s, 5u);
	if(F_MIPLOD && (bits(flags0, 6u, 1u) != 0u)) {
		// Dynamic LOD (GSH_VulkanDrawDesktop): LOD = (log2(1/|Q|) << L) + K
		let mipL = f32(1u << sw(s, 22u));
		let mipK = bitcast<f32>(sw(s, 23u));
		let lod = trunc(log2(abs(1.0 / q)) * mipL + mipK - 0.03125);
		let mipLevel = min(i32(lod), i32(sw(s, 21u)));
		if(mipLevel >= 1) {
			level = u32(mipLevel);
			bufAddress = sw(s, 24u + (level - 1u) * 2u);
			bufWidth = sw(s, 25u + (level - 1u) * 2u);
		}
	}
	if(F_LINEAR && (bits(flags0, 5u, 1u) != 0u)) {
		let pos = texPos - 0.5;
		var ab = fract(pos);
		if(any(abs(texPos) > vec2<f32>(1e30))) { ab = vec2<f32>(0.0); }
		let i0 = vec2<i32>(floor(pos));
		let i1 = i0 + vec2<i32>(1, 1);
		let ca = fetchTexel(s, i0, level, bufAddress, bufWidth);
		let cb = fetchTexel(s, vec2<i32>(i1.x, i0.y), level, bufAddress, bufWidth);
		let cc = fetchTexel(s, vec2<i32>(i0.x, i1.y), level, bufAddress, bufWidth);
		let cd = fetchTexel(s, i1, level, bufAddress, bufWidth);
		return mix(mix(ca, cb, ab.x), mix(cc, cd, ab.x), ab.y);
	}
	return fetchTexel(s, vec2<i32>(floor(texPos)), level, bufAddress, bufWidth);
}

fn readFramePixel(psm : u32, a : u32) -> u32 {
	switch(psm) {
		case 0x01u, 0x09u, 0x31u: { return read24(a); }
		case 0x02u, 0x0Au, 0x3Au: { return read16(a); }
		default: { return read32(a); }
	}
}

// Whether primitive p (index in prims) covers pixel (px, py): same tests as shade(), used to build the
// tile's coverage masks.
fn covers(p : u32, px : u32, py : u32) -> bool {
	let bboxMin = pw(p, 26u);
	let bboxMax = pw(p, 27u);
	if((px < (bboxMin & 0xFFFFu)) || (py < (bboxMin >> 16u)) || (px > (bboxMax & 0xFFFFu)) || (py > (bboxMax >> 16u))) { return false; }
	let sx = i32(px) * 16;
	let sy = i32(py) * 16;
	let x0 = bitcast<i32>(pw(p, 0u));
	let y0 = bitcast<i32>(pw(p, 1u));
	let x1 = bitcast<i32>(pw(p, 8u));
	let y1 = bitcast<i32>(pw(p, 9u));
	switch(pw(p, 24u)) {
		case 0u: {
			let x2 = bitcast<i32>(pw(p, 16u));
			let y2 = bitcast<i32>(pw(p, 17u));
			let area = f32(x1 - x0) * f32(y2 - y0) - f32(y1 - y0) * f32(x2 - x0);
			if(area == 0.0) { return false; }
			let o = select(-1, 1, area > 0.0);
			return edgeCovers(o * edgeFunction(x2 - x1, y2 - y1, sx - x1, sy - y1), o * (x2 - x1), o * (y2 - y1)) &&
			       edgeCovers(o * edgeFunction(x0 - x2, y0 - y2, sx - x2, sy - y2), o * (x0 - x2), o * (y0 - y2)) &&
			       edgeCovers(o * edgeFunction(x1 - x0, y1 - y0, sx - x0, sy - y0), o * (x1 - x0), o * (y1 - y0));
		}
		case 1u: {
			return (sx >= min(x0, x1)) && (sx < max(x0, x1)) && (sy >= min(y0, y1)) && (sy < max(y0, y1));
		}
		case 2u: {
			let dx = x1 - x0;
			let dy = y1 - y0;
			if(abs(dx) >= abs(dy)) {
				if((sx < min(x0, x1)) || (sx >= max(x0, x1))) { return false; }
				let ly = f32(y0) + f32(sx - x0) / f32(dx) * f32(dy);
				return i32(floor((ly + 8.0) / 16.0)) == i32(py);
			}
			if((sy < min(y0, y1)) || (sy >= max(y0, y1))) { return false; }
			let lx = f32(x0) + f32(sy - y0) / f32(dy) * f32(dx);
			return i32(floor((lx + 8.0) / 16.0)) == i32(px);
		}
		default: { return true; }
	}
}

// Edge setup for the tile's coverage mask: value at the tile's first pixel, steps per pixel, threshold
// (top-left rule: e >= 0 on top/left edges, e > 0 otherwise). Far edges (beyond 2^30) don't change sign
// within a tile: their steps are 0.
struct TileEdge {
	e : i32,
	stepX : i32,
	stepY : i32,
	bias : i32,
};

fn tileEdge(o : i32, dx : i32, dy : i32, sx : i32, sy : i32) -> TileEdge {
	var r : TileEdge;
	let ox = o * dx;
	let oy = o * dy;
	r.e = o * edgeFunction(dx, dy, sx, sy);
	r.stepX = -oy * 16;
	r.stepY = ox * 16;
	if(abs(r.e) == 1 && (abs(f32(dx) * f32(sy) - f32(dy) * f32(sx)) > 1073741824.0)) {
		r.e = r.e << 30u;
		r.stepX = 0;
		r.stepY = 0;
	}
	r.bias = select(1, 0, (oy < 0) || ((oy == 0) && (ox > 0)));
	return r;
}

// Coverage of the tile at (tileX, tileY) by primitive p: bit (y * TILE_SIZE + x) of word bit / 32.
fn coverageMask(p : u32, tileX : u32, tileY : u32) -> array<u32, ${Math.max(this.drawConfig.tile ** 2 / 32, 1)}> {
	var mask : array<u32, ${Math.max(this.drawConfig.tile ** 2 / 32, 1)}>;
	let bboxMin = pw(p, 26u);
	let bboxMax = pw(p, 27u);
	// Rows and columns of the tile inside the bounding box
	let c0 = i32(max(bboxMin & 0xFFFFu, tileX)) - i32(tileX);
	let c1 = i32(min(bboxMax & 0xFFFFu, tileX + TILE_SIZE - 1u)) - i32(tileX);
	let r0 = i32(max(bboxMin >> 16u, tileY)) - i32(tileY);
	let r1 = i32(min(bboxMax >> 16u, tileY + TILE_SIZE - 1u)) - i32(tileY);
	if((c0 > c1) || (r0 > r1)) { return mask; }
	let rowBits = ((2u << u32(c1)) - 1u) & ~((1u << u32(c0)) - 1u);
	let primType = pw(p, 24u);
	if(primType == PRIM_TRIANGLE) {
		let x0 = bitcast<i32>(pw(p, 0u));
		let y0 = bitcast<i32>(pw(p, 1u));
		let x1 = bitcast<i32>(pw(p, 8u));
		let y1 = bitcast<i32>(pw(p, 9u));
		let x2 = bitcast<i32>(pw(p, 16u));
		let y2 = bitcast<i32>(pw(p, 17u));
		let area = f32(x1 - x0) * f32(y2 - y0) - f32(y1 - y0) * f32(x2 - x0);
		if(area == 0.0) { return mask; }
		let o = select(-1, 1, area > 0.0);
		let sx = i32(tileX) * 16;
		let sy = i32(tileY) * 16;
		let e0 = tileEdge(o, x2 - x1, y2 - y1, sx - x1, sy - y1);
		let e1 = tileEdge(o, x0 - x2, y0 - y2, sx - x2, sy - y2);
		let e2 = tileEdge(o, x1 - x0, y1 - y0, sx - x0, sy - y0);
		for(var r = r0; r <= r1; r++) {
			var a = e0.e + e0.stepY * r + e0.stepX * c0;
			var bb = e1.e + e1.stepY * r + e1.stepX * c0;
			var c = e2.e + e2.stepY * r + e2.stepX * c0;
			// Edge values are linear along the row: whole row in or out when both ends agree
			let aEnd = a + e0.stepX * (c1 - c0);
			let bEnd = bb + e1.stepX * (c1 - c0);
			let cEnd = c + e2.stepX * (c1 - c0);
			var bits = 0u;
			if((min(a, aEnd) >= e0.bias) && (min(bb, bEnd) >= e1.bias) && (min(c, cEnd) >= e2.bias)) {
				bits = rowBits;
			} else if((max(a, aEnd) >= e0.bias) && (max(bb, bEnd) >= e1.bias) && (max(c, cEnd) >= e2.bias)) {
				for(var col = c0; col <= c1; col++) {
					if((a >= e0.bias) && (bb >= e1.bias) && (c >= e2.bias)) { bits |= 1u << u32(col); }
					a += e0.stepX;
					bb += e1.stepX;
					c += e2.stepX;
				}
			}
			let bit = u32(r) * TILE_SIZE;
			mask[bit >> 5u] |= bits << (bit & 31u);
		}
		return mask;
	}
	if(primType == PRIM_SPRITE) {
		// covers() checks the exact sprite bounds; the bounding box is already exact for sprites
		for(var r = r0; r <= r1; r++) {
			let bit = u32(r) * TILE_SIZE;
			mask[bit >> 5u] |= rowBits << (bit & 31u);
		}
		return mask;
	}
	for(var r = r0; r <= r1; r++) {
		for(var col = c0; col <= c1; col++) {
			if(covers(p, tileX + u32(col), tileY + u32(r))) {
				let bit = u32(r) * TILE_SIZE + u32(col);
				mask[bit >> 5u] |= 1u << (bit & 31u);
			}
		}
	}
	return mask;
}

fn shade(p : u32, px : u32, py : u32, cache : ptr<function, PixelCache>) {
	let primType = pw(p, 24u);
	let sx = i32(px) * 16;
	let sy = i32(py) * 16;
	let x0 = bitcast<i32>(pw(p, 0u));
	let y0 = bitcast<i32>(pw(p, 1u));
	let x1 = bitcast<i32>(pw(p, 8u));
	let y1 = bitcast<i32>(pw(p, 9u));

	// Weights of vertex 1 and 2 (vertex 0 gets the rest). The pixel is covered (coverageMask).
	var w = vec2<f32>(0.0);
	switch(primType) {
		case 0u: {
			if(!F_TRIANGLE) { return; }
			let x2 = bitcast<i32>(pw(p, 16u));
			let y2 = bitcast<i32>(pw(p, 17u));
			let area = f32(x1 - x0) * f32(y2 - y0) - f32(y1 - y0) * f32(x2 - x0);
			let o = select(-1, 1, area > 0.0);
			let e1 = o * edgeFunction(x0 - x2, y0 - y2, sx - x2, sy - y2);
			let e2 = o * edgeFunction(x1 - x0, y1 - y0, sx - x0, sy - y0);
			let absArea = abs(area);
			w = vec2<f32>(f32(e1), f32(e2)) / absArea;
		}
		case 1u: {
			if(!F_SPRITE) { return; }
			w = vec2<f32>(f32(sx - x0) / f32(x1 - x0), f32(sy - y0) / f32(y1 - y0));
		}
		case 2u: {
			if(!F_LINE) { return; }
			let dx = x1 - x0;
			let dy = y1 - y0;
			if(abs(dx) >= abs(dy)) {
				w = vec2<f32>(f32(sx - x0) / f32(dx), 0.0);
			} else {
				w = vec2<f32>(f32(sy - y0) / f32(dy), 0.0);
			}
		}
		default: { }
	}

	let s = pw(p, 25u) * (STATE_WORDS / 4u);
	let flags0 = sw(s, 0u);
	let flags1 = sw(s, 1u);
	let fbPsm = param(0u);
	let zbPsm = param(3u);

	// Tests that don't depend on the source color come first (scan mask, depth test, DATE): a fragment failing
	// them is dropped whatever the alpha test gives, and its texturing is skipped.
	let scanMask = bits(flags0, 13u, 2u);
	if(((scanMask == 2u) && ((py & 1u) == 0u)) || ((scanMask == 3u) && ((py & 1u) != 0u))) { return; }
	let depthTest = bits(flags0, 26u, 2u);
	if(depthTest == 0u) { return; }

	// Depth
	let z0 = pw(p, 2u);
	var z = z0;
	if((primType == PRIM_TRIANGLE) || (primType == PRIM_LINE)) {
		var dz = f32(bitcast<i32>(pw(p, 10u) - z0)) * w.x;
		if(primType == PRIM_TRIANGLE) {
			dz += f32(bitcast<i32>(pw(p, 18u) - z0)) * w.y;
		}
		z = z0 + u32(i32(round(dz)));
	}
	var srcDepth = z;
	switch(zbPsm) {
		case 0x31u: { srcDepth = min(z, 0xFFFFFFu); }
		case 0x32u, 0x3Au: { srcDepth = min(z, 0xFFFFu); }
		default: { }
	}
	if(F_DEPTHTEST && (depthTest >= 2u)) {
		if(((*cache).flags & 2u) == 0u) {
			switch(zbPsm) {
				case 0x31u: { (*cache).z = read24((*cache).zbAddress); }
				case 0x32u, 0x3Au: { (*cache).z = read16((*cache).zbAddress); }
				default: { (*cache).z = read32((*cache).zbAddress); }
			}
			(*cache).flags |= 2u;
		}
		var dstDepth = (*cache).z;
		if((param(1u) == param(4u)) && (((*cache).flags & 4u) != 0u)) {
			// Depth buffer = frame buffer (depth writes are off then): see the colors written in this batch
			dstDepth = (*cache).fb;
			if(zbPsm == PSMZ24) { dstDepth &= 0xFFFFFFu; }
		}
		if((depthTest == 2u) && (srcDepth < dstDepth)) { return; }
		if((depthTest == 3u) && (srcDepth <= dstDepth)) { return; }
	}

	let is16 = (fbPsm == PSMCT16) || (fbPsm == PSMCT16S) || (fbPsm == PSMZ16S);
	let is24 = (fbPsm == PSMCT24) || (fbPsm == PSMZ24) || (fbPsm == PSMCT24_UNK);
	if(((*cache).flags & 1u) == 0u) {
		(*cache).fb = readFramePixel(fbPsm, (*cache).fbAddress);
		(*cache).flags |= 1u;
	}
	let dstPixel = (*cache).fb;

	// Destination alpha test (DATE)
	if(F_DATE && (bits(flags1, 2u, 1u) != 0u)) {
		var alphaBit = (dstPixel & 0x80000000u) != 0u;
		if(is16) { alphaBit = (dstPixel & 0x8000u) != 0u; }
		if(!is24 && (alphaBit != (bits(flags1, 3u, 1u) != 0u))) { return; }
	}

	// Other attributes
	var color = unpackColor(pw(p, 3u));
	var st = vec2<f32>(bitcast<f32>(pw(p, 4u)), bitcast<f32>(pw(p, 5u)));
	var q = bitcast<f32>(pw(p, 6u));
	var fog = bitcast<f32>(pw(p, 7u));
	if(primType == PRIM_SPRITE) {
		let st1 = vec2<f32>(bitcast<f32>(pw(p, 12u)), bitcast<f32>(pw(p, 13u)));
		st = st + (st1 - st) * w;
	} else if(primType != PRIM_POINT) {
		let c1 = unpackColor(pw(p, 11u));
		let st1 = vec2<f32>(bitcast<f32>(pw(p, 12u)), bitcast<f32>(pw(p, 13u)));
		let q1 = bitcast<f32>(pw(p, 14u));
		let f1 = bitcast<f32>(pw(p, 15u));
		var c = color + (c1 - color) * w.x;
		var stv = st + (st1 - st) * w.x;
		var qv = q + (q1 - q) * w.x;
		var fv = fog + (f1 - fog) * w.x;
		if(primType == PRIM_TRIANGLE) {
			let c2 = unpackColor(pw(p, 19u));
			let st2 = vec2<f32>(bitcast<f32>(pw(p, 20u)), bitcast<f32>(pw(p, 21u)));
			let q2 = bitcast<f32>(pw(p, 22u));
			let f2 = bitcast<f32>(pw(p, 23u));
			c += (c2 - color) * w.y;
			stv += (st2 - st) * w.y;
			qv += (q2 - q) * w.y;
			fv += (f2 - fog) * w.y;
		}
		color = c;
		st = stv;
		q = qv;
		fog = fv;
	}

	// Texture function (TFX), in integers like the GS: (Ct * Cf) >> 7
	var src = vec4<i32>(clamp(color, vec4<f32>(0.0), vec4<f32>(255.0)));
	if(F_TEXTURE && (bits(flags0, 0u, 1u) != 0u)) {
		let tex = vec4<i32>(sampleTexture(s, st, q));
		let tcc = bits(flags0, 1u, 1u) != 0u;
		var rgb = tex.rgb;
		var a = src.a;
		switch(bits(flags0, 3u, 2u)) {
			case 0u: {
				rgb = min((tex.rgb * src.rgb) >> vec3<u32>(7u), vec3<i32>(255));
				if(tcc) { a = min((tex.a * src.a) >> 7u, 255); }
			}
			case 1u: {
				if(tcc) { a = tex.a; }
			}
			case 2u: {
				rgb = min(((tex.rgb * src.rgb) >> vec3<u32>(7u)) + vec3<i32>(src.a), vec3<i32>(255));
				if(tcc) { a = min(tex.a + src.a, 255); }
			}
			default: {
				rgb = min(((tex.rgb * src.rgb) >> vec3<u32>(7u)) + vec3<i32>(src.a), vec3<i32>(255));
				if(tcc) { a = tex.a; }
			}
		}
		src = vec4<i32>(rgb, a);
	}
	if(F_FOG && (bits(flags0, 11u, 1u) != 0u)) {
		let f = i32(clamp(fog, 0.0, 255.0));
		let fogColor = vec3<i32>(i32(sw(s, 20u) & 0xFFu), i32((sw(s, 20u) >> 8u) & 0xFFu), i32((sw(s, 20u) >> 16u) & 0xFFu));
		src = vec4<i32>((src.rgb * f + fogColor * (255 - f)) >> vec3<u32>(8u), src.a);
	}

	var writeColor = true;
	var writeDepth = F_DEPTHWRITE && (bits(flags0, 12u, 1u) != 0u);
	var writeAlpha = true;

	// Alpha test
	let alphaTest = select(1u, bits(flags0, 28u, 3u), F_ALPHATEST);
	let alphaRef = i32(sw(s, 18u));
	var alphaPass = true;
	switch(alphaTest) {
		case 0u: { alphaPass = false; }
		case 2u: { alphaPass = src.a < alphaRef; }
		case 3u: { alphaPass = src.a <= alphaRef; }
		case 4u: { alphaPass = src.a == alphaRef; }
		case 5u: { alphaPass = src.a >= alphaRef; }
		case 6u: { alphaPass = src.a > alphaRef; }
		case 7u: { alphaPass = src.a != alphaRef; }
		default: { }
	}
	if(!alphaPass) {
		switch(bits(flags1, 0u, 2u)) {
			case 0u: { return; }
			case 1u: { writeDepth = false; }
			case 2u: { writeColor = false; }
			default: { writeDepth = false; writeAlpha = false; }
		}
	}

	var dst : vec4<i32>;
	if(is16) {
		dst = vec4<i32>(i32((dstPixel & 0x1Fu) << 3u), i32(((dstPixel >> 5u) & 0x1Fu) << 3u), i32(((dstPixel >> 10u) & 0x1Fu) << 3u), select(0, 0x80, (dstPixel & 0x8000u) != 0u));
	} else {
		dst = vec4<i32>(unpackColor(dstPixel));
		if(is24) { dst.a = 0x80; }
	}

	// Alpha blending: ((A - B) * C >> 7) + D
	var outColor = src;
	if(F_BLEND && (bits(flags0, 15u, 1u) != 0u)) {
		let alphaFix = i32(sw(s, 17u));
		var ca = src.rgb;
		switch(bits(flags0, 16u, 2u)) { case 1u: { ca = dst.rgb; } case 2u: { ca = vec3<i32>(0); } default: { } }
		var cb = src.rgb;
		switch(bits(flags0, 18u, 2u)) { case 1u: { cb = dst.rgb; } case 2u: { cb = vec3<i32>(0); } default: { } }
		var cc = src.a;
		switch(bits(flags0, 20u, 2u)) { case 1u: { cc = dst.a; } case 2u: { cc = alphaFix; } default: { } }
		var cd = src.rgb;
		switch(bits(flags0, 22u, 2u)) { case 1u: { cd = dst.rgb; } case 2u: { cd = vec3<i32>(0); } default: { } }
		var rgb = (((ca - cb) * cc) >> vec3<u32>(7u)) + cd;
		if(bits(flags0, 24u, 1u) != 0u) {
			rgb = clamp(rgb, vec3<i32>(0), vec3<i32>(255));
		} else {
			rgb = rgb & vec3<i32>(255);
		}
		outColor = vec4<i32>(rgb, src.a);
	}
	if(bits(flags0, 25u, 1u) != 0u) { outColor.a = outColor.a | 0x80; }
	if(!writeAlpha) { outColor.a = dst.a; }

	if(writeColor) {
		var pixel : u32;
		if(is16) {
			pixel = (u32(outColor.r) >> 3u) | ((u32(outColor.g) >> 3u) << 5u) | ((u32(outColor.b) >> 3u) << 10u) | ((u32(outColor.a) >> 7u) << 15u);
		} else {
			pixel = u32(outColor.r) | (u32(outColor.g) << 8u) | (u32(outColor.b) << 16u) | (u32(outColor.a) << 24u);
		}
		let writeMask = sw(s, 16u);
		pixel = (pixel & writeMask) | (dstPixel & ~writeMask);
		(*cache).fb = pixel;
		(*cache).flags |= 4u;
	}
	if(writeDepth) {
		(*cache).z = srcDepth;
		(*cache).flags |= 10u;
	}
}

fn writeBackPixel(cache : ptr<function, PixelCache>) {
	if(((*cache).flags & 4u) != 0u) {
		switch(param(0u)) {
			case 0x01u, 0x09u, 0x31u: { write24((*cache).fbAddress, (*cache).fb); }
			case 0x02u, 0x0Au, 0x3Au: { write16((*cache).fbAddress, (*cache).fb); }
			default: { write32((*cache).fbAddress, (*cache).fb); }
		}
	}
	if(((*cache).flags & 8u) != 0u) {
		switch(param(3u)) {
			case 0x31u: { write24((*cache).zbAddress, (*cache).z); }
			case 0x32u, 0x3Au: { write16((*cache).zbAddress, (*cache).z); }
			default: { write32((*cache).zbAddress, (*cache).z); }
		}
	}
	(*cache).flags &= ~12u;
}

// One workgroup per tile of the batch's bounding box. CGSH_WebGPU::FlushBatch bins the primitives: the tile's
// list (in drawing order) is upload[binEntries + upload[binOffsets + tile] .. upload[binOffsets + tile + 1]].
// Each invocation (= one pixel) shades the tile's primitives in order.
// Workgroup i (row-major over a 32768-wide grid) draws tile upload[tileOrder + i] (non-empty tiles, longest lists first).
// P: fbPsm, fbAddress, fbWidth, zbPsm, zbAddress, -, -, originX, originY, maxX, maxY, tilesX, binOffsets, binEntries,
// tileOrder, tile count
// Invocations are mapped to 4 x 4 pixel blocks (16 consecutive invocations = one SIMD16 thread on Intel): small
// primitives cover fewer threads than with rows of pixels.
@compute @workgroup_size(${this.drawConfig.tile ** 2})
fn draw(@builtin(local_invocation_index) li : u32, @builtin(workgroup_id) wid : vec3<u32>) {
	let lid = vec2<u32>((li & 3u) | ((li >> 2u) & 4u), ((li >> 2u) & 3u) | ((li >> 3u) & 4u)); // 8 x 8 tile
	let index = wid.y * 32768u + wid.x;
	if(index >= param(15u)) { return; }
	let tile = upload[param(14u) + index];
	let tileX = param(7u) + (tile % param(11u)) * TILE_SIZE;
	let tileY = param(8u) + (tile / param(11u)) * TILE_SIZE;
	let px = tileX + lid.x;
	let py = tileY + lid.y;
	let begin = upload[param(12u) + tile];
	let end = upload[param(12u) + tile + 1u];
	let entries = param(13u);
	let fbPsm = param(0u);
	let zbPsm = param(3u);
	var cache : PixelCache;
	cache.fbAddress = pixelAddress(fbPsm, param(1u), param(2u), px, py);
	cache.zbAddress = pixelAddress(zbPsm, param(4u), param(2u), px, py);
	cache.flags = 0u;
	let bit = lid.y * TILE_SIZE + lid.x;
	for(var chunk = begin; chunk < end; chunk += GROUP_SIZE) {
		let count = min(GROUP_SIZE, end - chunk);
		// Each invocation computes the coverage mask of one primitive over the whole tile...
		if(li < count) {
			let p = upload[entries + chunk + li];
			chunkPrims[li] = p;
			chunkMasks[li] = coverageMask(p, tileX, tileY);
		}
		workgroupBarrier();
		// ...then each invocation shades the primitives covering its pixel, in order.
		for(var k = 0u; k < count; k++) {
			if((chunkMasks[k][bit >> 5u] & (1u << (bit & 31u))) == 0u) { continue; }
			shade(chunkPrims[k], px, py, &cache);
		}
		workgroupBarrier();
	}
	writeBackPixel(&cache);
}
`;
	},

	wgslPresent() {
		return `
@group(0) @binding(0) var<storage, read> mem : array<u32>;
@group(0) @binding(1) var<storage, read> swz : array<u32>;
@group(0) @binding(2) var<uniform> P : array<vec4<u32>, 16>;

${this.wgslMemory("mem[i & WORDMASK]")}

struct VsOut {
	@builtin(position) position : vec4<f32>,
	@location(0) texCoord : vec2<f32>,
};

// One quad per display layer (GSH_VulkanPresent): P[0] = psm, bufAddress, bufWidth, dispWidth;
// P[1] = dispHeight, layerX, layerY, layerWidth; P[2] = layerHeight, ...
@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VsOut {
	let uv = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
	let dispSize = vec2<f32>(f32(P[0].w), f32(P[1].x));
	let layerPos = vec2<f32>(f32(P[1].y), f32(P[1].z));
	let layerSize = vec2<f32>(f32(P[1].w), f32(P[2].x));
	let pos = ((uv * layerSize + layerPos) / dispSize) * 2.0 - 1.0;
	var out : VsOut;
	out.position = vec4<f32>(pos.x, -pos.y, 0.0, 1.0);
	out.texCoord = uv;
	return out;
}

@fragment
fn fs(in : VsOut) -> @location(0) vec4<f32> {
	let psm = P[0].x;
	let layerSize = vec2<f32>(f32(P[1].w), f32(P[2].x));
	let pos = vec2<u32>(in.texCoord * layerSize);
	if((psm == PSMCT16) || (psm == PSMCT16S)) {
		return psm16ToColor(readPixel(psm, P[0].y, P[0].z, pos.x, pos.y));
	}
	// PSMCT24 is read as 32 bits like the Vulkan backend (its alpha byte is whatever is in memory)
	return psm32ToColor(read32(pixelAddress(psm, P[0].y, P[0].z, pos.x, pos.y)));
}
`;
	},
};
