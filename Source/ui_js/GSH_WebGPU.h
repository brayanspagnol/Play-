#pragma once

#include <vector>
#include <map>
#include "gs/GSHandler.h"
#include "gs/GsSpriteRegion.h"

//GS renderer on WebGPU, ported from GSH_Vulkan: everything works on a copy of GS local memory kept in a
//GPU storage buffer, so frame buffers, textures and CLUTs alias exactly like on the real GS (formats are
//only views of the same memory). Runs on the GS thread; the GPU work is done by GSH_WebGPU.js (WGSL
//shaders), which this class drives with a command list and an upload arena in wasm memory (see Flush).
//See notes/webgpu.md in ps2-web for the design.
class CGSH_WebGPU : public CGSHandler
{
public:
	CGSH_WebGPU(int width, int height);
	virtual ~CGSH_WebGPU() = default;

	static FactoryFunction GetFactoryFunction(int width, int height);

	//Queued on the GS thread by initVm: creates the device (asynchronous, the GS thread waits suspended).
	void InitializeContext();
	//Output (OffscreenCanvas) size and presentation mode; queued on the GS thread.
	void SetOutput(int width, int height, PRESENTATION_MODE);
	//1: log per-frame statistics (batches, primitives, GPU time of the draws) every 2 s from the GS thread
	void SetDebug(int);

	void SaveState(Framework::CZipArchiveWriter&) override;

	void ProcessHostToLocalTransfer() override;
	void ProcessLocalToHostTransfer() override;
	void ProcessLocalToLocalTransfer() override;
	void ProcessClutTransfer(uint32, uint32) override;

	uint8* GetRam() const override;

	//Called by GSH_WebGPU.js
	void OnDeviceReady(bool success);

protected:
	void WriteRegisterImpl(uint8, uint64) override;
	void InitializeImpl() override;
	void ReleaseImpl() override;
	void ResetImpl() override;
	void FlipImpl(const DISPLAY_INFO&) override;
	void BeginTransferWrite() override;
	void TransferWrite(const uint8*, uint32) override;
	void WriteBackMemoryCache() override;
	void SyncMemoryCache() override;
	void SyncCLUT(const TEX0&) override;

public:
	//Command list shared with GSH_WebGPU.js (keep in sync with OP_* there).
	enum OPCODE : uint32
	{
		OP_UPLOAD_TO_MEMORY = 1,   //uploadOffset, memoryOffset, size
		OP_UPLOAD_TO_SWIZZLE = 2,  //uploadOffset, size
		OP_MEMORY_TO_COPY = 3,     //memoryOffset, size
		OP_DISPATCH = 4,           //pipeline, paramsOffset, groupsX, groupsY, groupsZ
		OP_PRESENT = 5,            //paramsOffset, layerCount
		OP_READBACK = 6,           //memoryOffset, size, heap pointer
		OP_UPLOAD_TO_CLUT = 7,     //uploadOffset, clutOffset, size
	};

	enum PIPELINE : uint32
	{
		PIPELINE_XFER_HOST = 0,
		PIPELINE_XFER_LOCAL = 1,
		PIPELINE_CLUT_LOAD = 2,
		PIPELINE_DRAW = 3,
		PIPELINE_TEX_DECODE = 4,
	};

	//Layouts shared with the draw shader (GSH_WebGPU.js)
	enum
	{
		PRIM_WORDS = 28,  //3 x (x, y (12.4, offset applied), z, rgba, s, t, q, fog), type, state, bbox min, bbox max
		STATE_WORDS = 40, //see SetRenderingContext
		DRAW_TILE_SIZE = 8, //pixels per side of a draw shader tile (workgroup), see GsWgpu.drawConfig.tile
		SUBMIT_PRIMS = 8192, //command lists are submitted when they hold this many primitives (and at flips)
		DRAW_GROUPS_PER_ROW = 32768, //draw dispatches are 2D past this many workgroups (WebGPU limit 65535 per dimension)
		CPU_BLOCK_SIZE = 256, //granularity of m_cpuValid
		TEX_DECODE_TEXELS = 0x400000, //size of the decoded texture buffer in texels (GsWgpu.TEX_DECODE_SIZE / 4)
	};

	//Features of a draw batch, the draw shader is specialized for them (GsWgpu.drawFeatures)
	enum DRAW_FEATURE : uint32
	{
		DRAW_FEATURE_TEXTURE = 1 << 0,
		DRAW_FEATURE_LINEAR = 1 << 1,
		DRAW_FEATURE_MIPLOD = 1 << 2,
		DRAW_FEATURE_CLUT32 = 1 << 3,
		DRAW_FEATURE_MEMCOPY = 1 << 4,
		DRAW_FEATURE_BLEND = 1 << 5,
		DRAW_FEATURE_FOG = 1 << 6,
		DRAW_FEATURE_ALPHATEST = 1 << 7,
		DRAW_FEATURE_DATE = 1 << 8,
		DRAW_FEATURE_DEPTHTEST = 1 << 9,
		DRAW_FEATURE_DEPTHWRITE = 1 << 10,
		DRAW_FEATURE_TRIANGLE = 1 << 11,
		DRAW_FEATURE_SPRITE = 1 << 12,
		DRAW_FEATURE_LINE = 1 << 13,
		//Texture formats: 32/24-bit, 16-bit, 8-bit indexed, 4-bit indexed, 16-bit CLUT
		DRAW_FEATURE_TEX32 = 1 << 14,
		DRAW_FEATURE_TEX16 = 1 << 15,
		DRAW_FEATURE_TEX8 = 1 << 16,
		DRAW_FEATURE_TEX4 = 1 << 17,
		DRAW_FEATURE_CLUT16 = 1 << 18,
		//Some of the batch's textures are read decoded (see DecodeBatchTextures); the texture format bits above are then
		//only those of the textures read from GS memory
		DRAW_FEATURE_TEXDECODED = 1 << 19,
		DRAW_FEATURE_TEXFORMATS = DRAW_FEATURE_CLUT32 | DRAW_FEATURE_TEX32 | DRAW_FEATURE_TEX16 | DRAW_FEATURE_TEX8 |
		                          DRAW_FEATURE_TEX4 | DRAW_FEATURE_CLUT16,
	};

	enum PRIM_TYPE : uint32
	{
		PRIM_TYPE_TRIANGLE = 0,
		PRIM_TYPE_SPRITE = 1,
		PRIM_TYPE_LINE = 2,
		PRIM_TYPE_POINT = 3,
	};

private:
	struct CLUTKEY : public convertible<uint64>
	{
		uint32 idx4 : 1;
		uint32 cbp : 14;
		uint32 cpsm : 4;
		uint32 csm : 1;
		uint32 csa : 5;
		uint32 cbw : 6;
		uint32 cou : 6;
		uint32 cov : 10;
		uint32 reserved : 16;
	};
	static_assert(sizeof(CLUTKEY) == sizeof(uint64), "CLUTKEY too big for an uint64");

	enum
	{
		CLUT_CACHE_SIZE = 32,
		CLUT_SLOTS = 1024, //GsWgpu.CLUT_BUFFER_SIZE / (CLUTENTRYCOUNT * 4)
		PARAMS_SIZE = 256,
		MAX_PENDING_XFERS = 64,
		XFER_WORDS = 9,
		XFER_WHOLE_WORDS = 0x100, //flag in a transfer's psm (see FlushTransfers)
	};

	struct VERTEX_DATA
	{
		int32 x = 0, y = 0;
		uint32 z = 0;
		uint32 color = 0;
		float s = 0, t = 0, q = 1;
		float fog = 0;
	};

	struct BATCH
	{
		uint32 fbPsm = 0;
		uint32 fbAddress = 0;
		uint32 fbWidth = 0;
		uint32 zbPsm = 0;
		uint32 zbAddress = 0;
		//Memory written by the batch (frame buffer, plus depth buffer when written), for texture hazards
		uint32 writeStart = 0;
		uint32 writeEnd = 0;
		uint32 zbWriteStart = 0;
		uint32 zbWriteEnd = 0;
		uint32 primStart = 0;
		uint32 stateStart = 0; //state records are per batch: the batch's are [stateStart, end of m_states)
		uint32 minX = ~0U, minY = ~0U, maxX = 0, maxY = 0;
		uint32 features = 0;
	};

	void ProcessPrim(uint64);
	void VertexKick(uint8, uint64);
	void SetRenderingContext(uint64);
	void Prim_Point();
	void Prim_Line();
	void Prim_Triangle();
	void Prim_Sprite();
	void AddPrim(PRIM_TYPE, const VERTEX_DATA*, uint32 vertexCount);
	void FlushBatch();
	void DecodeBatchTextures();
	void FlushTransfers();
	bool BatchAccesses(uint32, uint32) const;
	bool LoadClutFromCpu(uint32, const TEX0&, const TEXCLUT&);
	void InvalidateCpuMemory(uint32, uint32);
	void CopyMemoryForTexture();
	void CheckSpriteCachedClutInvalidation(const CGsSpriteRect&);

	uint32 AllocUpload(uint32 size, uint32 alignment = 16);
	uint32* AllocParams(uint32& offset);
	void PushCommand(std::initializer_list<uint32>);
	void Dispatch(PIPELINE, uint32 paramsOffset, uint32 groupsX, uint32 groupsY = 1, uint32 groupsZ = 1);
	bool Flush();
	void Readback(uint32 address, uint32 size);

	void UploadSwizzleTables();

	int32 FindCachedClut(const CLUTKEY&) const;
	void ClearClutCache();
	static CLUTKEY MakeCachedClutKey(const TEX0&, const TEXCLUT&);
	uint32 DoClutLoad(const TEX0&, const TEXCLUT&);

	int m_width = 0;
	int m_height = 0;
	bool m_deviceReady = false;

	uint8* m_memoryCache = nullptr;
	bool m_memoryCacheSynced = false; //see SaveState

	std::vector<uint32> m_commands;
	std::vector<uint8> m_upload;

	std::vector<uint8> m_xferBuffer;

public:
	struct PENDING_XFER
	{
		uint32 psm, bufAddress, bufWidth, rrw, dsax, dsay, dataOffset, pixelCount, firstGroup;
		uint32 start, end; //destination memory range
		uint32 rrh;
	};
private:
	void WriteCpuMemory(const PENDING_XFER&, uint32);
	std::vector<PENDING_XFER> m_pendingXfers;

	std::vector<uint8> m_cpuValid; //per CPU_BLOCK_SIZE bytes: m_memoryCache matches GPU memory there
	std::vector<uint32> m_clutUploads; //CLUTs built on the CPU: slot, then CLUTENTRYCOUNT entries
	uint32 m_pendingXferGroups = 0;
	std::vector<uint8> m_pendingXferBlocks; //per CPU_BLOCK_SIZE bytes: written by a pending transfer
	std::vector<uint32> m_xferBlocks;

	CLUTKEY m_clutStates[CLUT_CACHE_SIZE];
	uint32 m_clutSlots[CLUT_CACHE_SIZE] = {}; //slot of the CLUT ring holding each cached CLUT
	uint32 m_nextClutSlot = 0;
	uint32 m_nextClutCacheIndex = 0;
	uint32 m_lastClutFillCacheIndex = -1;
	uint32 m_lastClutFillPointer = 0;
	uint32 m_clutIndex = 0; //slot used by the next draws

	//Draw context (as GSH_Vulkan)
	VERTEX m_vtxBuffer[3];
	uint32 m_vtxCount = 0;
	bool m_pendingPrim = false;
	uint64 m_pendingPrimValue = 0;
	uint32 m_primitiveType = 0;
	PRMODE m_primitiveMode;
	uint32 m_fbBasePtr = 0;
	int32 m_primOfsX = 0;
	int32 m_primOfsY = 0;
	uint32 m_texWidth = 0;
	uint32 m_texHeight = 0;
	uint32 m_scissor[4] = {}; //x0, y0, x1, y1 (inclusive)

	//Optimization for Virtua Fighter 2, Sega Rally 95 (see GSH_Vulkan)
	float m_lastLineU = 0;
	float m_lastLineV = 0;
	float m_clutLineU = 0;
	float m_clutLineV = 0;

	//Primitives and states of the current command list, batches of primitives (see notes/webgpu.md)
	std::vector<uint32> m_prims;
	std::vector<uint32> m_states;
	std::vector<uint64> m_stateAreas; //per state record: sum of its primitives' bounding box areas
	std::vector<uint32> m_stateFormats; //per state record: its DRAW_FEATURE_TEXFORMATS bits
	std::vector<uint32> m_decodeJobs;
	uint32 m_decodeTexels = 0; //decoded texture buffer used by the current command list
	std::vector<uint32> m_binOffsets;
	std::vector<uint32> m_tileOrder;
	uint32 m_state[STATE_WORDS] = {};
	int32 m_stateIndex = -1; //index of m_state in m_states, -1 when not stored yet
	uint32 m_stateFeatures = 0; //DRAW_FEATURE_* used by m_state
	uint32 m_stateReadStart = 0; //memory read by m_state's texture
	uint32 m_stateReadEnd = 0;
	bool m_stateInBatch = false; //m_state's reads are in m_batchReads
	std::vector<std::pair<uint32, uint32>> m_batchReads;
	BATCH m_batch;
	bool m_memoryCopyActive = false;
	uint32 m_memoryCopyAddress = 0;
	uint32 m_memoryCopySize = 0;
	CGsSpriteRegion m_memoryCopyRegion;
};
