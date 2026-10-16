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
	};

	enum PIPELINE : uint32
	{
		PIPELINE_XFER_HOST = 0,
		PIPELINE_XFER_LOCAL = 1,
		PIPELINE_CLUT_LOAD = 2,
		PIPELINE_DRAW = 3,
	};

	//Layouts shared with the draw shader (GSH_WebGPU.js)
	enum
	{
		PRIM_WORDS = 28,  //3 x (x, y (12.4, offset applied), z, rgba, s, t, q, fog), type, state, bbox min, bbox max
		STATE_WORDS = 40, //see SetRenderingContext
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
		PARAMS_SIZE = 256,
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
		uint32 minX = ~0U, minY = ~0U, maxX = 0, maxY = 0;
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
	void CopyMemoryForTexture();
	void CheckSpriteCachedClutInvalidation(const CGsSpriteRect&);

	uint32 AllocUpload(uint32 size, uint32 alignment = 16);
	uint32* AllocParams(uint32& offset);
	void PushCommand(std::initializer_list<uint32>);
	void Dispatch(PIPELINE, uint32 paramsOffset, uint32 groupsX, uint32 groupsY = 1, uint32 groupsZ = 1);
	void Flush();
	void Readback(uint32 address, uint32 size);

	void UploadSwizzleTables();

	int32 FindCachedClut(const CLUTKEY&) const;
	void ClearClutCache();
	static CLUTKEY MakeCachedClutKey(const TEX0&, const TEXCLUT&);
	void DoClutLoad(uint32 clutIndex, const TEX0&, const TEXCLUT&);

	int m_width = 0;
	int m_height = 0;
	bool m_deviceReady = false;

	uint8* m_memoryCache = nullptr;
	bool m_memoryCacheSynced = false; //see SaveState

	std::vector<uint32> m_commands;
	std::vector<uint8> m_upload;

	std::vector<uint8> m_xferBuffer;

	CLUTKEY m_clutStates[CLUT_CACHE_SIZE];
	uint32 m_nextClutCacheIndex = 0;
	uint32 m_lastClutFillCacheIndex = -1;
	uint32 m_lastClutFillPointer = 0;
	uint32 m_clutIndex = 0;

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
	uint32 m_state[STATE_WORDS] = {};
	int32 m_stateIndex = -1; //index of m_state in m_states, -1 when not stored yet
	BATCH m_batch;
	bool m_memoryCopyActive = false;
	uint32 m_memoryCopyAddress = 0;
	uint32 m_memoryCopySize = 0;
	CGsSpriteRegion m_memoryCopyRegion;
};
