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

	RAMSIZE: 0x400000,
	CLUT_BUFFER_SIZE: 32 * 512 * 4,
	SWIZZLE_TABLES_SIZE: 45056 * 4,
	PARAMS_SIZE: 256,

	device: null,
	errorCount: 0,

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
				this.device = await adapter.requestDevice();
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
		this.memoryCopy = d.createBuffer({ label: "GS memory copy", size: this.RAMSIZE, usage: B.STORAGE | B.COPY_DST });
		this.clut = d.createBuffer({ label: "CLUT", size: this.CLUT_BUFFER_SIZE, usage: B.STORAGE });
		this.swizzle = d.createBuffer({ label: "Swizzle tables", size: this.SWIZZLE_TABLES_SIZE, usage: B.STORAGE | B.COPY_DST });
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

		const computeModule = d.createShaderModule({ label: "GS compute", code: this.wgslCompute() });
		const computePipelineLayout = d.createPipelineLayout({ bindGroupLayouts: [this.computeLayout] });
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

		this.upload = this.prims = this.states = null;
		this.ensureBuffers(16 << 20, 4 << 20, 1 << 20);
	},

	// Upload arena, primitives and states of a command list; grown (and the bind groups remade) on demand.
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
			case this.OP_MEMORY_TO_COPY:
				endPass();
				enc.copyBufferToBuffer(this.memory, cmds[i + 1], this.memoryCopy, cmds[i + 1], cmds[i + 2]);
				i += 3;
				break;
			case this.OP_DISPATCH:
				if (!pass) {
					pass = enc.beginComputePass();
				}
				pass.setPipeline(this.computePipelines[cmds[i + 1]]);
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
		d.queue.submit([enc.finish()]);
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

// Host -> local transfer (GSH_VulkanTransferHost): one invocation per pixel of the stream.
@compute @workgroup_size(256)
fn xferHost(@builtin(global_invocation_id) id : vec3<u32>) {
	let psm = param(0u);
	let bufAddress = param(1u);
	let bufWidth = param(2u);
	let rrw = param(3u);
	let dsax = param(4u);
	let dsay = param(5u);
	let dataOffset = param(6u);
	let pixelCount = param(7u);
	let i = id.x;
	if(i >= pixelCount) { return; }
	let x = ((i % rrw) + dsax) % 2048u;
	let y = ((i / rrw) + dsay) % 2048u;
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
	writePixel(psm, bufAddress, bufWidth, x, y, value);
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
		clut[base + (index & 511u)] = color & 0xFFFFu;
		clut[base + ((index + 256u) & 511u)] = color >> 16u;
	}
}

// Draws (see notes/webgpu.md): compute rasterizer working on GS memory, the per-pixel part follows
// GSH_VulkanDrawDesktop::CreateFragmentShader / GSH_VulkanDrawUtils.
@group(0) @binding(6) var<storage, read> prims : array<u32>;
@group(0) @binding(7) var<storage, read> states : array<u32>;

const PRIM_WORDS = 28u;
const STATE_WORDS = 40u;

const PRIM_TRIANGLE = 0u;
const PRIM_SPRITE = 1u;
const PRIM_LINE = 2u;
const PRIM_POINT = 3u;

var<workgroup> primMask : array<atomic<u32>, 8>;

fn sw(s : u32, i : u32) -> u32 { return states[s + i]; }
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
	if(fromCopy) { return memCopy[i & WORDMASK]; }
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
			// Indexed: CLUT lookup (GSH_VulkanDrawUtils GetClutColor)
			var index = raw;
			if((psm == PSMT4) || (psm == PSMT4HL) || (psm == PSMT4HH)) { index = raw + sw(s, 9u); }
			let clutBase = sw(s, 19u);
			let cpsm = sw(s, 3u);
			if((cpsm == PSMCT16) || (cpsm == PSMCT16S)) {
				return expand16(clut[clutBase + (index & 511u)], tcc, aem, ta0, ta1);
			}
			let lo = clut[clutBase + (index & 0xFFu)];
			let hi = clut[clutBase + (index & 0xFFu) + 256u];
			return unpackColor(lo | (hi << 16u));
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
	if(bits(flags0, 6u, 1u) != 0u) {
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
	if(bits(flags0, 5u, 1u) != 0u) {
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

fn shade(p : u32, px : u32, py : u32) {
	let b = p * PRIM_WORDS;
	let bboxMin = prims[b + 26u];
	let bboxMax = prims[b + 27u];
	if((px < (bboxMin & 0xFFFFu)) || (py < (bboxMin >> 16u)) || (px > (bboxMax & 0xFFFFu)) || (py > (bboxMax >> 16u))) { return; }
	let primType = prims[b + 24u];
	let sx = i32(px) * 16;
	let sy = i32(py) * 16;
	let x0 = bitcast<i32>(prims[b + 0u]);
	let y0 = bitcast<i32>(prims[b + 1u]);
	let x1 = bitcast<i32>(prims[b + 8u]);
	let y1 = bitcast<i32>(prims[b + 9u]);

	// Weights of vertex 1 and 2 (vertex 0 gets the rest).
	var w = vec2<f32>(0.0);
	switch(primType) {
		case 0u: {
			let x2 = bitcast<i32>(prims[b + 16u]);
			let y2 = bitcast<i32>(prims[b + 17u]);
			let area = f32(x1 - x0) * f32(y2 - y0) - f32(y1 - y0) * f32(x2 - x0);
			if(area == 0.0) { return; }
			let o = select(-1, 1, area > 0.0);
			let e0 = o * edgeFunction(x2 - x1, y2 - y1, sx - x1, sy - y1);
			let e1 = o * edgeFunction(x0 - x2, y0 - y2, sx - x2, sy - y2);
			let e2 = o * edgeFunction(x1 - x0, y1 - y0, sx - x0, sy - y0);
			if(!edgeCovers(e0, o * (x2 - x1), o * (y2 - y1)) ||
			   !edgeCovers(e1, o * (x0 - x2), o * (y0 - y2)) ||
			   !edgeCovers(e2, o * (x1 - x0), o * (y1 - y0))) { return; }
			let absArea = abs(area);
			w = vec2<f32>(f32(e1), f32(e2)) / absArea;
		}
		case 1u: {
			if((sx < min(x0, x1)) || (sx >= max(x0, x1)) || (sy < min(y0, y1)) || (sy >= max(y0, y1))) { return; }
			w = vec2<f32>(f32(sx - x0) / f32(x1 - x0), f32(sy - y0) / f32(y1 - y0));
		}
		case 2u: {
			let dx = x1 - x0;
			let dy = y1 - y0;
			if(abs(dx) >= abs(dy)) {
				if((sx < min(x0, x1)) || (sx >= max(x0, x1))) { return; }
				let t = f32(sx - x0) / f32(dx);
				let ly = f32(y0) + t * f32(dy);
				if(i32(floor((ly + 8.0) / 16.0)) != i32(py)) { return; }
				w = vec2<f32>(t, 0.0);
			} else {
				if((sy < min(y0, y1)) || (sy >= max(y0, y1))) { return; }
				let t = f32(sy - y0) / f32(dy);
				let lx = f32(x0) + t * f32(dx);
				if(i32(floor((lx + 8.0) / 16.0)) != i32(px)) { return; }
				w = vec2<f32>(t, 0.0);
			}
		}
		default: { }
	}

	// Attributes
	let z0 = prims[b + 2u];
	var z = z0;
	var color = unpackColor(prims[b + 3u]);
	var st = vec2<f32>(bitcast<f32>(prims[b + 4u]), bitcast<f32>(prims[b + 5u]));
	var q = bitcast<f32>(prims[b + 6u]);
	var fog = bitcast<f32>(prims[b + 7u]);
	if(primType == PRIM_SPRITE) {
		let st1 = vec2<f32>(bitcast<f32>(prims[b + 12u]), bitcast<f32>(prims[b + 13u]));
		st = st + (st1 - st) * w;
	} else if(primType != PRIM_POINT) {
		let c1 = unpackColor(prims[b + 11u]);
		let st1 = vec2<f32>(bitcast<f32>(prims[b + 12u]), bitcast<f32>(prims[b + 13u]));
		let q1 = bitcast<f32>(prims[b + 14u]);
		let f1 = bitcast<f32>(prims[b + 15u]);
		let dz1 = f32(bitcast<i32>(prims[b + 10u] - z0));
		var dz = dz1 * w.x;
		var c = color + (c1 - color) * w.x;
		var stv = st + (st1 - st) * w.x;
		var qv = q + (q1 - q) * w.x;
		var fv = fog + (f1 - fog) * w.x;
		if(primType == PRIM_TRIANGLE) {
			let c2 = unpackColor(prims[b + 19u]);
			let st2 = vec2<f32>(bitcast<f32>(prims[b + 20u]), bitcast<f32>(prims[b + 21u]));
			let q2 = bitcast<f32>(prims[b + 22u]);
			let f2 = bitcast<f32>(prims[b + 23u]);
			dz += f32(bitcast<i32>(prims[b + 18u] - z0)) * w.y;
			c += (c2 - color) * w.y;
			stv += (st2 - st) * w.y;
			qv += (q2 - q) * w.y;
			fv += (f2 - fog) * w.y;
		}
		z = z0 + u32(i32(round(dz)));
		color = c;
		st = stv;
		q = qv;
		fog = fv;
	}

	let s = prims[b + 25u] * STATE_WORDS;
	let flags0 = sw(s, 0u);
	let flags1 = sw(s, 1u);

	// Texture function (TFX), in integers like the GS: (Ct * Cf) >> 7
	var src = vec4<i32>(clamp(color, vec4<f32>(0.0), vec4<f32>(255.0)));
	if(bits(flags0, 0u, 1u) != 0u) {
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
	if(bits(flags0, 11u, 1u) != 0u) {
		let f = i32(clamp(fog, 0.0, 255.0));
		let fogColor = vec3<i32>(i32(sw(s, 20u) & 0xFFu), i32((sw(s, 20u) >> 8u) & 0xFFu), i32((sw(s, 20u) >> 16u) & 0xFFu));
		src = vec4<i32>((src.rgb * f + fogColor * (255 - f)) >> vec3<u32>(8u), src.a);
	}

	var writeColor = true;
	var writeDepth = bits(flags0, 12u, 1u) != 0u;
	var writeAlpha = true;

	let scanMask = bits(flags0, 13u, 2u);
	if(((scanMask == 2u) && ((py & 1u) == 0u)) || ((scanMask == 3u) && ((py & 1u) != 0u))) { return; }

	// Alpha test
	let alphaTest = bits(flags0, 28u, 3u);
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

	let fbPsm = param(0u);
	let zbPsm = param(3u);
	let fbAddress = pixelAddress(fbPsm, param(1u), param(2u), px, py);
	let zbAddress = pixelAddress(zbPsm, param(4u), param(2u), px, py);

	let is16 = (fbPsm == PSMCT16) || (fbPsm == PSMCT16S) || (fbPsm == PSMZ16S);
	let is24 = (fbPsm == PSMCT24) || (fbPsm == PSMZ24) || (fbPsm == PSMCT24_UNK);
	let dstPixel = readFramePixel(fbPsm, fbAddress);
	var dst : vec4<i32>;
	if(is16) {
		dst = vec4<i32>(i32((dstPixel & 0x1Fu) << 3u), i32(((dstPixel >> 5u) & 0x1Fu) << 3u), i32(((dstPixel >> 10u) & 0x1Fu) << 3u), select(0, 0x80, (dstPixel & 0x8000u) != 0u));
	} else {
		dst = vec4<i32>(unpackColor(dstPixel));
		if(is24) { dst.a = 0x80; }
	}

	// Destination alpha test (DATE)
	if(bits(flags1, 2u, 1u) != 0u) {
		var alphaBit = (dstPixel & 0x80000000u) != 0u;
		if(is16) { alphaBit = (dstPixel & 0x8000u) != 0u; }
		if(!is24 && (alphaBit != (bits(flags1, 3u, 1u) != 0u))) { return; }
	}

	// Depth test
	var srcDepth = z;
	switch(zbPsm) {
		case 0x31u: { srcDepth = min(z, 0xFFFFFFu); }
		case 0x32u, 0x3Au: { srcDepth = min(z, 0xFFFFu); }
		default: { }
	}
	let depthTest = bits(flags0, 26u, 2u);
	if(depthTest == 0u) { return; }
	if(depthTest >= 2u) {
		var dstDepth = read32(zbAddress);
		switch(zbPsm) {
			case 0x31u: { dstDepth = dstDepth & 0xFFFFFFu; }
			case 0x32u, 0x3Au: { dstDepth = read16(zbAddress); }
			default: { }
		}
		if((depthTest == 2u) && (srcDepth < dstDepth)) { return; }
		if((depthTest == 3u) && (srcDepth <= dstDepth)) { return; }
	}

	// Alpha blending: ((A - B) * C >> 7) + D
	var outColor = src;
	if(bits(flags0, 15u, 1u) != 0u) {
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
		if(is16) { write16(fbAddress, pixel); } else if(is24) { write24(fbAddress, pixel); } else { write32(fbAddress, pixel); }
	}
	if(writeDepth) {
		switch(zbPsm) {
			case 0x31u: { write24(zbAddress, srcDepth); }
			case 0x32u, 0x3Au: { write16(zbAddress, srcDepth); }
			default: { write32(zbAddress, srcDepth); }
		}
	}
}

// One workgroup per 16x16 tile of the batch's bounding box: test 256 primitive bounding boxes at a time
// against the tile, then each invocation (= one pixel) shades the overlapping primitives in order.
// P: fbPsm, fbAddress, fbWidth, zbPsm, zbAddress, primStart, primCount, originX, originY
@compute @workgroup_size(16, 16)
fn draw(@builtin(local_invocation_id) lid : vec3<u32>, @builtin(local_invocation_index) li : u32,
        @builtin(workgroup_id) wid : vec3<u32>) {
	let tileX = param(7u) + wid.x * 16u;
	let tileY = param(8u) + wid.y * 16u;
	let px = tileX + lid.x;
	let py = tileY + lid.y;
	let primStart = param(5u);
	let primCount = param(6u);
	for(var chunk = 0u; chunk < primCount; chunk += 256u) {
		if(li < 8u) { atomicStore(&primMask[li], 0u); }
		workgroupBarrier();
		let i = chunk + li;
		if(i < primCount) {
			let b = (primStart + i) * PRIM_WORDS;
			let bboxMin = prims[b + 26u];
			let bboxMax = prims[b + 27u];
			if(((bboxMin & 0xFFFFu) <= tileX + 15u) && ((bboxMax & 0xFFFFu) >= tileX) &&
			   ((bboxMin >> 16u) <= tileY + 15u) && ((bboxMax >> 16u) >= tileY)) {
				atomicOr(&primMask[li >> 5u], 1u << (li & 31u));
			}
		}
		workgroupBarrier();
		for(var word = 0u; word < 8u; word++) {
			var mask = atomicLoad(&primMask[word]);
			while(mask != 0u) {
				let bit = firstTrailingBit(mask);
				mask &= mask - 1u;
				shade(primStart + chunk + word * 32u + bit, px, py);
			}
		}
		workgroupBarrier();
	}
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
