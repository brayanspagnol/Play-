#include "GSH_WebGPU.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <emscripten/em_js.h>
#include "AppConfig.h"
#include "gs/GsPixelFormats.h"
#include "gs/GsTransferRange.h"
#include "gs/GsCachedArea.h"

// clang-format off
EM_JS_DEPS(GSH_WebGPU, "$getWasmTableEntry,$callUserCallback");
EM_JS(void, GsWgpu_Init, (void* gs, void* resumeFn, void* readyFn, int width, int height), {
	GsWgpu.init(gs, resumeFn, readyFn, width, height);
});

EM_JS(void, GsWgpu_SetDebug, (int debug), {
	GsWgpu.debug = debug;
});

EM_JS(void, GsWgpu_Resize, (int width, int height), {
	GsWgpu.resize(width, height);
});

EM_JS(int, GsWgpu_Execute, (const void* commands, int commandCount, const void* upload, int uploadSize,
                             const void* prims, int primsSize, const void* states, int statesSize), {
	return GsWgpu.execute(commands, commandCount, upload, uploadSize, prims, primsSize, states, statesSize);
});
// clang-format on

static uint32 MakeColor(uint8 r, uint8 g, uint8 b, uint8 a)
{
	return (a << 24) | (b << 16) | (g << 8) | (r);
}

static uint16 RGBA32ToRGBA16(uint32 inputColor)
{
	uint32 result = 0;
	result |= ((inputColor & 0x000000F8) >> (0 + 3)) << 0;
	result |= ((inputColor & 0x0000F800) >> (8 + 3)) << 5;
	result |= ((inputColor & 0x00F80000) >> (16 + 3)) << 10;
	result |= ((inputColor & 0x80000000) >> 31) << 15;
	return result;
}

static bool RangesOverlap(uint32 start1, uint32 end1, uint32 start2, uint32 end2)
{
	return (start1 < end2) && (start2 < end1);
}

static void OnDeviceReadyCallback(void* gs, int success)
{
	reinterpret_cast<CGSH_WebGPU*>(gs)->OnDeviceReady(success != 0);
}

CGSH_WebGPU::CGSH_WebGPU(int width, int height)
    : m_width(width)
    , m_height(height)
{
}

CGSHandler::FactoryFunction CGSH_WebGPU::GetFactoryFunction(int width, int height)
{
	return [width, height]() { return new CGSH_WebGPU(width, height); };
}

void CGSH_WebGPU::InitializeImpl()
{
	//GPU objects are made by InitializeContext (device creation is asynchronous).
	m_memoryCache = new uint8[RAMSIZE];
	memset(m_memoryCache, 0, RAMSIZE);
	m_cpuValid.assign(RAMSIZE / CPU_BLOCK_SIZE, 1);
	m_pendingXferBlocks.assign(RAMSIZE / CPU_BLOCK_SIZE, 0);
	ClearClutCache();
	UploadSwizzleTables();
	WriteBackMemoryCache();
}

void CGSH_WebGPU::InitializeContext()
{
	GsWgpu_Init(this, reinterpret_cast<void*>(&CGSH_WebGPU::ResumeThreadProc),
	            reinterpret_cast<void*>(&OnDeviceReadyCallback), m_width, m_height);
	SuspendThreadProc();
}

void CGSH_WebGPU::OnDeviceReady(bool success)
{
	m_deviceReady = success;
	if(!success)
	{
		fprintf(stderr, "GSH_WebGPU: no WebGPU device, nothing will be drawn.\n");
		m_commands.clear();
		m_upload.clear();
		return;
	}
	//A first (black) frame tells the page the GS is ready (emulator.js waits for it before booting).
	PushCommand({OP_PRESENT, 0, 0});
	Flush();
}

void CGSH_WebGPU::ReleaseImpl()
{
	m_commands.clear();
	m_upload.clear();
	delete[] m_memoryCache;
	m_memoryCache = nullptr;
}

void CGSH_WebGPU::ResetImpl()
{
	CGSHandler::ResetImpl();
	FlushBatch();
	m_vtxCount = 0;
	m_primitiveType = PRIM_INVALID;
	m_pendingPrim = false;
	m_pendingPrimValue = 0;
	m_memoryCopyActive = false;
	m_xferBuffer.clear();
	ClearClutCache();
	memset(m_memoryCache, 0, RAMSIZE);
	WriteBackMemoryCache();
}

void CGSH_WebGPU::SetDebug(int debug)
{
	SendGSCall([debug]() { GsWgpu_SetDebug(debug); });
}

void CGSH_WebGPU::SetOutput(int width, int height, PRESENTATION_MODE mode)
{
	SendGSCall([this, width, height, mode]() {
		m_width = width;
		m_height = height;
		GsWgpu_Resize(width, height);
		PRESENTATION_PARAMS params;
		params.mode = mode;
		params.windowWidth = width;
		params.windowHeight = height;
		SetPresentationParams(params);
	});
}

uint8* CGSH_WebGPU::GetRam() const
{
	return m_memoryCache;
}

/////////////////////////////////////////////////////////////
// Command list
/////////////////////////////////////////////////////////////

uint32 CGSH_WebGPU::AllocUpload(uint32 size, uint32 alignment)
{
	uint32 offset = (m_upload.size() + alignment - 1) & ~(alignment - 1);
	m_upload.resize(offset + size);
	return offset;
}

uint32* CGSH_WebGPU::AllocParams(uint32& offset)
{
	//Uniform block bound with a dynamic offset (minUniformBufferOffsetAlignment is at most 256).
	offset = AllocUpload(PARAMS_SIZE, PARAMS_SIZE);
	auto params = reinterpret_cast<uint32*>(m_upload.data() + offset);
	memset(params, 0, PARAMS_SIZE);
	return params;
}

void CGSH_WebGPU::PushCommand(std::initializer_list<uint32> words)
{
	m_commands.insert(m_commands.end(), words);
}

void CGSH_WebGPU::Dispatch(PIPELINE pipeline, uint32 paramsOffset, uint32 groupsX, uint32 groupsY, uint32 groupsZ)
{
	PushCommand({OP_DISPATCH, pipeline, paramsOffset, groupsX, groupsY, groupsZ});
}

bool CGSH_WebGPU::Flush()
{
	FlushBatch();
	if(!m_clutUploads.empty())
	{
		//CLUTs built on the CPU go to fresh slots: write them before anything else in the list, one copy per
		//run of consecutive slots.
		std::vector<uint32> uploads;
		uint32 recordSize = CLUTENTRYCOUNT + 1;
		uint32 count = m_clutUploads.size() / recordSize;
		uint32 offset = AllocUpload(count * CLUTENTRYCOUNT * sizeof(uint32), 4);
		auto dst = reinterpret_cast<uint32*>(m_upload.data() + offset);
		uint32 runStart = 0;
		for(uint32 i = 0; i < count; i++)
		{
			const uint32* record = m_clutUploads.data() + i * recordSize;
			memcpy(dst + i * CLUTENTRYCOUNT, record + 1, CLUTENTRYCOUNT * sizeof(uint32));
			bool runEnds = (i + 1 == count) || (m_clutUploads[(i + 1) * recordSize] != record[0] + 1);
			if(runEnds)
			{
				uint32 firstSlot = m_clutUploads[runStart * recordSize];
				uploads.insert(uploads.end(), {OP_UPLOAD_TO_CLUT, offset + runStart * CLUTENTRYCOUNT * 4,
				                               firstSlot * CLUTENTRYCOUNT * 4, (i + 1 - runStart) * CLUTENTRYCOUNT * 4});
				runStart = i + 1;
			}
		}
		m_commands.insert(m_commands.begin(), uploads.begin(), uploads.end());
		m_clutUploads.clear();
	}
	if(!m_deviceReady || m_commands.empty()) return false;
	m_upload.resize((m_upload.size() + 3) & ~3);
	bool gpuBusy = GsWgpu_Execute(m_commands.data(), m_commands.size(), m_upload.data(), m_upload.size(),
	               m_prims.data(), m_prims.size() * sizeof(uint32), m_states.data(), m_states.size() * sizeof(uint32));
	m_commands.clear();
	m_upload.clear();
	m_prims.clear();
	m_states.clear();
	m_stateAreas.clear();
	m_stateFormats.clear();
	m_stateIndex = -1;
	m_batch.primStart = 0;
	m_batch.stateStart = 0;
	m_decodeTexels = 0;
	return gpuBusy;
}

void CGSH_WebGPU::Readback(uint32 address, uint32 size)
{
	assert(m_deviceReady);
	if(!m_deviceReady) return;
	FlushBatch();
	PushCommand({OP_READBACK, address, size, static_cast<uint32>(reinterpret_cast<uintptr_t>(m_memoryCache + address))});
	std::fill(m_cpuValid.begin() + address / CPU_BLOCK_SIZE, m_cpuValid.begin() + (address + size) / CPU_BLOCK_SIZE, 1);
	Flush();
	//The data lands in m_memoryCache when the map completes; the next GS call runs after that.
	SuspendThreadProc();
}

template <typename Storage>
static void AppendSwizzleTable(std::vector<uint32>& tables)
{
	auto offsets = CGsPixelFormats::CPixelIndexor<Storage>::GetPageOffsets();
	tables.insert(tables.end(), offsets, offsets + Storage::PAGEWIDTH * Storage::PAGEHEIGHT);
}

void CGSH_WebGPU::UploadSwizzleTables()
{
	//Same order and offsets as SWZ_* in GSH_WebGPU.js
	std::vector<uint32> tables;
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMCT32>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMZ32>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMCT16>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMCT16S>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMZ16>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMZ16S>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMT8>(tables);
	AppendSwizzleTable<CGsPixelFormats::STORAGEPSMT4>(tables);
	assert(tables.size() == 45056);
	uint32 size = tables.size() * sizeof(uint32);
	uint32 offset = AllocUpload(size);
	memcpy(m_upload.data() + offset, tables.data(), size);
	PushCommand({OP_UPLOAD_TO_SWIZZLE, offset, size});
}

/////////////////////////////////////////////////////////////
// Registers / flip
/////////////////////////////////////////////////////////////

void CGSH_WebGPU::WriteRegisterImpl(uint8 registerId, uint64 data)
{
	//Some games such as Silent Hill 2 don't finish their transfers
	//completely: make sure we push the data to the GS's RAM nevertheless.
	if(!m_xferBuffer.empty() && (registerId != GS_REG_HWREG))
	{
		ProcessHostToLocalTransfer();
	}

	CGSHandler::WriteRegisterImpl(registerId, data);

	switch(registerId)
	{
	case GS_REG_PRIM:
		//Check if previous primitive set was a line
		if(m_primitiveType == PRIM_LINE)
		{
			//Virtua Fighter 2, Sega Rally 95 (and others): draw lines and use the result as CLUT
			//(see GSH_Vulkan::WriteRegisterImpl).
			if(
			    (m_clutLineU != m_lastLineU) ||
			    (m_clutLineV != m_lastLineV))
			{
				ClearClutCache();
			}
			m_clutLineU = m_lastLineU;
			m_clutLineV = m_lastLineV;
		}
		m_pendingPrim = true;
		m_pendingPrimValue = data;
		break;

	case GS_REG_XYZ2:
	case GS_REG_XYZ3:
	case GS_REG_XYZF2:
	case GS_REG_XYZF3:
		VertexKick(registerId, data);
		break;
	}
}

void CGSH_WebGPU::FlipImpl(const DISPLAY_INFO& dispInfo)
{
	FlushBatch();
	uint32 layerCount = 0;
	uint32 firstParams = 0;
	auto viewport = GetPresentationViewport();
	for(const auto& layer : dispInfo.layers)
	{
		if(!layer.enabled) continue;
		uint32 paramsOffset = 0;
		auto params = AllocParams(paramsOffset);
		if(layerCount == 0) firstParams = paramsOffset;
		assert(paramsOffset == firstParams + layerCount * PARAMS_SIZE);
		float blendAlpha = std::min(static_cast<float>(layer.constantAlpha) * 2.f / 255.f, 1.f);
		uint32 blendMode = 0;
		if(layer.useConstantAlpha)
		{
			blendMode = (blendAlpha != 1.0f) ? 2 : 0;
		}
		else
		{
			blendMode = 1;
		}
		params[0] = layer.psm;
		params[1] = layer.bufPtr;
		params[2] = layer.bufWidth;
		params[3] = dispInfo.width;
		params[4] = dispInfo.height;
		params[5] = layer.offsetX;
		params[6] = layer.offsetY;
		params[7] = layer.width;
		params[8] = layer.height;
		params[9] = blendMode;
		memcpy(&params[10], &blendAlpha, sizeof(float));
		params[11] = viewport.offsetX;
		params[12] = viewport.offsetY;
		params[13] = viewport.width;
		params[14] = viewport.height;
		layerCount++;
	}
	PushCommand({OP_PRESENT, firstParams, layerCount});
	if(Flush())
	{
		//Too many frames queued on the GPU: wait (GSH_WebGPU.js resumes the GS thread when one completes).
		SuspendThreadProc();
	}
	CGSHandler::FlipImpl(dispInfo);
}

/////////////////////////////////////////////////////////////
// Transfers
/////////////////////////////////////////////////////////////

void CGSH_WebGPU::BeginTransferWrite()
{
	assert(m_xferBuffer.empty());
	m_xferBuffer.clear();
}

void CGSH_WebGPU::TransferWrite(const uint8* imageData, uint32 length)
{
	m_xferBuffer.insert(m_xferBuffer.end(), imageData, imageData + length);
}

//Calls f(block index) for every 256-byte block of GS memory holding pixels of the rectangle
template <typename Storage, typename F>
static void ForEachBlockOf(uint32 bufAddress, uint32 bufWidth, uint32 x0, uint32 y0, uint32 width, uint32 height, F&& f)
{
	static uint8 dummy[1];
	CGsPixelFormats::CPixelIndexor<Storage> indexor(dummy, bufAddress, std::max<uint32>(bufWidth / 64, 1));
	for(uint32 y = y0 & ~(Storage::BLOCKHEIGHT - 1); y < y0 + height; y += Storage::BLOCKHEIGHT)
	{
		for(uint32 x = x0 & ~(Storage::BLOCKWIDTH - 1); x < x0 + width; x += Storage::BLOCKWIDTH)
		{
			uint32 address = reinterpret_cast<uint8*>(indexor.GetPixelAddress(x % 2048, y % 2048)) - dummy;
			f(address / CGSH_WebGPU::CPU_BLOCK_SIZE);
		}
	}
}

template <typename F>
static bool ForEachBlock(uint32 psm, uint32 bufAddress, uint32 bufWidth, uint32 x0, uint32 y0, uint32 width, uint32 height, F&& f)
{
	switch(psm)
	{
	case CGSHandler::PSMCT32:
	case CGSHandler::PSMCT24:
	case CGSHandler::PSMT8H:
	case CGSHandler::PSMT4HL:
	case CGSHandler::PSMT4HH:
		ForEachBlockOf<CGsPixelFormats::STORAGEPSMCT32>(bufAddress, bufWidth, x0, y0, width, height, f);
		return true;
	case CGSHandler::PSMCT16:
		ForEachBlockOf<CGsPixelFormats::STORAGEPSMCT16>(bufAddress, bufWidth, x0, y0, width, height, f);
		return true;
	case CGSHandler::PSMCT16S:
		ForEachBlockOf<CGsPixelFormats::STORAGEPSMCT16S>(bufAddress, bufWidth, x0, y0, width, height, f);
		return true;
	case CGSHandler::PSMT8:
		ForEachBlockOf<CGsPixelFormats::STORAGEPSMT8>(bufAddress, bufWidth, x0, y0, width, height, f);
		return true;
	case CGSHandler::PSMT4:
		ForEachBlockOf<CGsPixelFormats::STORAGEPSMT4>(bufAddress, bufWidth, x0, y0, width, height, f);
		return true;
	default:
		return false;
	}
}

//Host -> local transfers are queued and dispatched together (one dispatch for many small texture uploads):
//they may run before the draws of the open batch as long as that batch doesn't read or write their memory.
void CGSH_WebGPU::ProcessHostToLocalTransfer()
{
	ClearClutCache();

	auto bltBuf = make_convertible<BITBLTBUF>(m_nReg[GS_REG_BITBLTBUF]);
	auto trxReg = make_convertible<TRXREG>(m_nReg[GS_REG_TRXREG]);
	auto trxPos = make_convertible<TRXPOS>(m_nReg[GS_REG_TRXPOS]);

	uint32 pixelCount = 0;
	switch(bltBuf.nDstPsm)
	{
	default:
		assert(false);
		[[fallthrough]];
	case PSMCT32:
	case PSMZ32:
		pixelCount = m_xferBuffer.size() / 4;
		break;
	case PSMCT24:
	case PSMZ24:
		pixelCount = m_xferBuffer.size() / 3;
		break;
	case PSMCT16:
	case PSMCT16S:
	case PSMZ16S:
		pixelCount = m_xferBuffer.size() / 2;
		break;
	case PSMT8:
	case PSMT8H:
		pixelCount = m_xferBuffer.size();
		break;
	case PSMT4:
	case PSMT4HL:
	case PSMT4HH:
		pixelCount = m_xferBuffer.size() * 2;
		break;
	}

	if(pixelCount != 0)
	{
		auto [dstAddress, dstSize] = GsTransfer::GetDstRange(bltBuf, trxReg, trxPos);
		uint32 dstEnd = dstAddress + dstSize;
		if(BatchAccesses(dstAddress, dstEnd))
		{
			FlushBatch();
		}
		uint32 groupCount = (pixelCount + 255) / 256;
		uint32 rrh = (pixelCount + trxReg.nRRW - 1) / std::max<uint32>(trxReg.nRRW, 1);
		//Blocks written by this transfer (the page ranges of GetDstRange are too coarse: small uploads share pages)
		m_xferBlocks.clear();
		bool knownBlocks = ForEachBlock(bltBuf.nDstPsm, bltBuf.GetDstPtr(), bltBuf.GetDstWidth(), trxPos.nDSAX, trxPos.nDSAY, trxReg.nRRW, rrh,
		                                [&](uint32 block) { m_xferBlocks.push_back(block % (RAMSIZE / CPU_BLOCK_SIZE)); });
		bool overlapsPending = false;
		if(knownBlocks)
		{
			overlapsPending = std::any_of(m_xferBlocks.begin(), m_xferBlocks.end(), [&](uint32 block) { return m_pendingXferBlocks[block] != 0; });
		}
		else
		{
			overlapsPending = std::any_of(m_pendingXfers.begin(), m_pendingXfers.end(),
			                              [&](const PENDING_XFER& xfer) { return RangesOverlap(dstAddress, dstEnd, xfer.start, xfer.end); });
		}
		if(overlapsPending || (m_pendingXfers.size() == MAX_PENDING_XFERS) || ((m_pendingXferGroups + groupCount) > 65535))
		{
			FlushTransfers();
		}

		uint32 dataOffset = AllocUpload(m_xferBuffer.size() + 4, 4);
		memcpy(m_upload.data() + dataOffset, m_xferBuffer.data(), m_xferBuffer.size());

		PENDING_XFER xfer;
		xfer.psm = bltBuf.nDstPsm;
		xfer.bufAddress = bltBuf.GetDstPtr();
		xfer.bufWidth = bltBuf.GetDstWidth();
		xfer.rrw = trxReg.nRRW;
		xfer.dsax = trxPos.nDSAX;
		xfer.dsay = trxPos.nDSAY;
		xfer.dataOffset = dataOffset / 4;
		xfer.pixelCount = pixelCount;
		xfer.firstGroup = m_pendingXferGroups;
		xfer.rrh = rrh;
		xfer.start = dstAddress;
		xfer.end = dstEnd;
		m_pendingXfers.push_back(xfer);
		m_pendingXferGroups += groupCount;
		if(knownBlocks)
		{
			for(uint32 block : m_xferBlocks) m_pendingXferBlocks[block] = 1;
		}
		else
		{
			std::fill(m_pendingXferBlocks.begin() + dstAddress / CPU_BLOCK_SIZE, m_pendingXferBlocks.begin() + std::min<uint32>(dstEnd, RAMSIZE) / CPU_BLOCK_SIZE, 1);
		}

		//Keep the CPU copy of small CLUT-format uploads (CLUTs are built on the CPU from it, see LoadClutFromCpu)
		InvalidateCpuMemory(dstAddress, dstEnd);
		if((m_xferBuffer.size() <= 0x1000) && ((xfer.psm == PSMCT32) || (xfer.psm == PSMCT16) || (xfer.psm == PSMCT16S)))
		{
			WriteCpuMemory(xfer, trxReg.nRRH);
		}
	}

	m_xferBuffer.clear();
}

void CGSH_WebGPU::FlushTransfers()
{
	if(m_pendingXfers.empty()) return;
	//Table of XFER_WORDS words per transfer (layout read by the xferHost shader)
	uint32 tableOffset = AllocUpload(m_pendingXfers.size() * XFER_WORDS * sizeof(uint32), 4);
	auto table = reinterpret_cast<uint32*>(m_upload.data() + tableOffset);
	for(const auto& xfer : m_pendingXfers)
	{
		//PSMT8/PSMT4 rectangles made of whole 16x4/32x4 columns: the shader writes whole words (XFER_WHOLE_WORDS)
		uint32 columnWidth = (xfer.psm == PSMT8) ? 16 : (xfer.psm == PSMT4) ? 32 : 0;
		bool wholeWords = (columnWidth != 0) && ((xfer.dsax % columnWidth) == 0) && ((xfer.rrw % columnWidth) == 0) &&
		                  ((xfer.dsay % 4) == 0) && ((xfer.rrh % 4) == 0) && (xfer.pixelCount == xfer.rrw * xfer.rrh);
		*table++ = xfer.psm | (wholeWords ? XFER_WHOLE_WORDS : 0);
		*table++ = xfer.bufAddress;
		*table++ = xfer.bufWidth;
		*table++ = xfer.rrw;
		*table++ = xfer.dsax;
		*table++ = xfer.dsay;
		*table++ = xfer.dataOffset;
		*table++ = xfer.pixelCount;
		*table++ = xfer.firstGroup;
	}
	uint32 paramsOffset = 0;
	auto params = AllocParams(paramsOffset);
	params[0] = m_pendingXfers.size();
	params[1] = tableOffset / 4;
	PushCommand({OP_DISPATCH, PIPELINE_XFER_HOST, paramsOffset, m_pendingXferGroups, 1, 1});
	std::fill(m_pendingXferBlocks.begin(), m_pendingXferBlocks.end(), 0);
	m_pendingXfers.clear();
	m_pendingXferGroups = 0;
}

//Memory written by the GPU: transfer data remembered for CLUTs is no longer what's there
//The CPU copy of GS memory (m_memoryCache) is only up to date where m_cpuValid says so: everything after a
//readback or a state load, plus small host transfers written on both sides. GPU writes invalidate it.
void CGSH_WebGPU::InvalidateCpuMemory(uint32 start, uint32 end)
{
	start = std::min<uint32>(start, RAMSIZE) / CPU_BLOCK_SIZE;
	end = (std::min<uint32>(end, RAMSIZE) + CPU_BLOCK_SIZE - 1) / CPU_BLOCK_SIZE;
	std::fill(m_cpuValid.begin() + start, m_cpuValid.begin() + end, 0);
}

template <typename Storage>
static void WriteTransferPixels(uint8* ram, uint8* valid, const uint8* data, uint32 bufAddress, uint32 bufWidth,
                                uint32 dsax, uint32 dsay, uint32 rrw, uint32 pixelCount)
{
	CGsPixelFormats::CPixelIndexor<Storage> indexor(ram, bufAddress, bufWidth / 64);
	auto pixels = reinterpret_cast<const typename Storage::Unit*>(data);
	for(uint32 i = 0; i < pixelCount; i++)
	{
		uint32 x = ((i % rrw) + dsax) % 2048;
		uint32 y = ((i / rrw) + dsay) % 2048;
		auto address = indexor.GetPixelAddress(x, y);
		*address = pixels[i];
		valid[(reinterpret_cast<uint8*>(address) - ram) / CGSH_WebGPU::CPU_BLOCK_SIZE] = 1;
	}
}

void CGSH_WebGPU::WriteCpuMemory(const PENDING_XFER& xfer, uint32)
{
	switch(xfer.psm)
	{
	case PSMCT32:
		WriteTransferPixels<CGsPixelFormats::STORAGEPSMCT32>(m_memoryCache, m_cpuValid.data(), m_xferBuffer.data(), xfer.bufAddress, xfer.bufWidth, xfer.dsax, xfer.dsay, xfer.rrw, xfer.pixelCount);
		break;
	case PSMCT16:
		WriteTransferPixels<CGsPixelFormats::STORAGEPSMCT16>(m_memoryCache, m_cpuValid.data(), m_xferBuffer.data(), xfer.bufAddress, xfer.bufWidth, xfer.dsax, xfer.dsay, xfer.rrw, xfer.pixelCount);
		break;
	case PSMCT16S:
		WriteTransferPixels<CGsPixelFormats::STORAGEPSMCT16S>(m_memoryCache, m_cpuValid.data(), m_xferBuffer.data(), xfer.bufAddress, xfer.bufWidth, xfer.dsax, xfer.dsay, xfer.rrw, xfer.pixelCount);
		break;
	}
}

//Whether the open batch reads (textures) or writes (frame/depth buffer) memory in [start, end)
bool CGSH_WebGPU::BatchAccesses(uint32 start, uint32 end) const
{
	if((m_prims.size() / PRIM_WORDS) == m_batch.primStart) return false;
	if(RangesOverlap(start, end, m_batch.writeStart, m_batch.writeEnd) ||
	   RangesOverlap(start, end, m_batch.zbWriteStart, m_batch.zbWriteEnd))
	{
		return true;
	}
	return std::any_of(m_batchReads.begin(), m_batchReads.end(),
	                   [&](const auto& range) { return RangesOverlap(start, end, range.first, range.second); });
}

void CGSH_WebGPU::ProcessLocalToHostTransfer()
{
	bool readsEnabled = CAppConfig::GetInstance().GetPreferenceBoolean(PREF_CGSHANDLER_GS_RAM_READS_ENABLED);
	if(!readsEnabled) return;

	auto bltBuf = make_convertible<BITBLTBUF>(m_nReg[GS_REG_BITBLTBUF]);
	auto trxReg = make_convertible<TRXREG>(m_nReg[GS_REG_TRXREG]);
	auto trxPos = make_convertible<TRXPOS>(m_nReg[GS_REG_TRXPOS]);

	auto [copyBase, copySize] = GsTransfer::GetSrcRange(bltBuf, trxReg, trxPos);
	if(copyBase >= RAMSIZE) return;
	if((copyBase + copySize) > RAMSIZE)
	{
		copySize = RAMSIZE - copyBase;
	}
	//ponytail: always waits for the GPU; Vulkan reuses stale data for recurring reads (LOCAL_TO_HOST_XFER_HISTORY),
	//add that if a game reads back every frame.
	Readback(copyBase, copySize);
}

void CGSH_WebGPU::ProcessLocalToLocalTransfer()
{
	FlushBatch();
	ClearClutCache();

	auto bltBuf = make_convertible<BITBLTBUF>(m_nReg[GS_REG_BITBLTBUF]);
	auto trxReg = make_convertible<TRXREG>(m_nReg[GS_REG_TRXREG]);
	auto trxPos = make_convertible<TRXPOS>(m_nReg[GS_REG_TRXPOS]);

	assert(trxPos.nDIR == 0);

	//Read from a snapshot when source and destination overlap (the Vulkan backend only does it when
	//the base pointers match: Tairyou Jigoku's sewer scrolling).
	auto [srcAddress, srcSize] = GsTransfer::GetSrcRange(bltBuf, trxReg, trxPos);
	auto [dstAddress, dstSize] = GsTransfer::GetDstRange(bltBuf, trxReg, trxPos);
	srcSize = std::min<uint32>(srcSize, RAMSIZE - std::min<uint32>(srcAddress, RAMSIZE));
	InvalidateCpuMemory(dstAddress, dstAddress + dstSize);
	bool useCopy = (srcAddress < (dstAddress + dstSize)) && (dstAddress < (srcAddress + srcSize));
	if(useCopy && (srcSize != 0))
	{
		PushCommand({OP_MEMORY_TO_COPY, srcAddress, srcSize});
	}

	uint32 paramsOffset = 0;
	auto params = AllocParams(paramsOffset);
	params[0] = bltBuf.nSrcPsm;
	params[1] = bltBuf.nDstPsm;
	params[2] = bltBuf.GetSrcPtr();
	params[3] = bltBuf.GetSrcWidth();
	params[4] = bltBuf.GetDstPtr();
	params[5] = bltBuf.GetDstWidth();
	params[6] = trxPos.nSSAX;
	params[7] = trxPos.nSSAY;
	params[8] = trxPos.nDSAX;
	params[9] = trxPos.nDSAY;
	params[10] = trxReg.nRRW;
	params[11] = trxReg.nRRH;
	params[12] = useCopy ? 1 : 0;
	Dispatch(PIPELINE_XFER_LOCAL, paramsOffset, (trxReg.nRRW + 15) / 16, (trxReg.nRRH + 15) / 16);
}

void CGSH_WebGPU::ProcessClutTransfer(uint32, uint32)
{
}

void CGSH_WebGPU::WriteBackMemoryCache()
{
	FlushBatch();
	std::fill(m_cpuValid.begin(), m_cpuValid.end(), 1);
	uint32 offset = AllocUpload(RAMSIZE);
	memcpy(m_upload.data() + offset, m_memoryCache, RAMSIZE);
	PushCommand({OP_UPLOAD_TO_MEMORY, offset, 0, RAMSIZE});
}

void CGSH_WebGPU::SyncMemoryCache()
{
	if(m_memoryCacheSynced)
	{
		m_memoryCacheSynced = false;
		return;
	}
	Readback(0, RAMSIZE);
}

void CGSH_WebGPU::SaveState(Framework::CZipArchiveWriter& archive)
{
	//CGSHandler::SaveState reads GetRam() as soon as its SyncMemoryCache call returned, but here that
	//call only starts the readback: sync first and wait for a call that runs after the GS thread resumed.
	SendGSCall([this]() { SyncMemoryCache(); }, true);
	SendGSCall([this]() { m_memoryCacheSynced = true; }, true);
	CGSHandler::SaveState(archive);
}

/////////////////////////////////////////////////////////////
// CLUT
/////////////////////////////////////////////////////////////

void CGSH_WebGPU::ClearClutCache()
{
	memset(&m_clutStates, 0, sizeof(m_clutStates));
	m_lastClutFillCacheIndex = -1;
	m_lastClutFillPointer = -1;
}

int32 CGSH_WebGPU::FindCachedClut(const CLUTKEY& key) const
{
	for(uint32 i = 0; i < CLUT_CACHE_SIZE; i++)
	{
		if(m_clutStates[i] == key)
		{
			return i;
		}
	}
	return -1;
}

CGSH_WebGPU::CLUTKEY CGSH_WebGPU::MakeCachedClutKey(const TEX0& tex0, const TEXCLUT& texClut)
{
	auto clutKey = make_convertible<CLUTKEY>(0);
	clutKey.idx4 = CGsPixelFormats::IsPsmIDTEX4(tex0.nPsm);
	clutKey.cbp = tex0.nCBP;
	clutKey.cpsm = tex0.nCPSM;
	clutKey.csm = tex0.nCSM;
	clutKey.csa = tex0.nCSA;
	clutKey.cbw = texClut.nCBW;
	clutKey.cou = texClut.nCOU;
	clutKey.cov = texClut.nCOV;
	return clutKey;
}

//CLUTs are loaded into a ring of CLUT_SLOTS slots (the Vulkan backend reuses 32 slots and ends its render
//pass at each load): a slot isn't overwritten before the draws using it ran, so loads don't split batches.
//The load is recorded before the batch's draw dispatch, so it must not depend on what the batch draws.
uint32 CGSH_WebGPU::DoClutLoad(const TEX0& tex0, const TEXCLUT& texClut)
{
	uint32 clutStart = tex0.GetCLUTPtr();
	uint32 clutEnd = clutStart + ((tex0.nCSM == 0) ? 0x2000 : RAMSIZE);
	if(RangesOverlap(clutStart, clutEnd, m_batch.writeStart, m_batch.writeEnd) ||
	   RangesOverlap(clutStart, clutEnd, m_batch.zbWriteStart, m_batch.zbWriteEnd))
	{
		FlushBatch();
	}
	uint32 clutIndex = m_nextClutSlot;
	m_nextClutSlot = (m_nextClutSlot + 1) % CLUT_SLOTS;
	if(m_nextClutSlot == 0)
	{
		//Wrapping around: the next loads overwrite slots that queued draws may use (CPU loaded CLUTs
		//are written at the start of the command list, so submit it).
		Flush();
		ClearClutCache();
	}
	if(LoadClutFromCpu(clutIndex, tex0, texClut))
	{
		return clutIndex;
	}
	FlushTransfers();
	uint32 paramsOffset = 0;
	auto params = AllocParams(paramsOffset);
	params[0] = CGsPixelFormats::IsPsmIDTEX8(tex0.nPsm) ? 1 : 0;
	params[1] = tex0.nCSM;
	params[2] = tex0.nCPSM;
	params[3] = tex0.GetCLUTPtr();
	params[4] = (tex0.nCSM == 0) ? 0x40 : texClut.GetBufWidth();
	params[5] = tex0.nCSA;
	params[6] = texClut.GetOffsetU();
	params[7] = texClut.GetOffsetV();
	params[8] = clutIndex * CLUTENTRYCOUNT;
	Dispatch(PIPELINE_CLUT_LOAD, paramsOffset, 1);
	return clutIndex;
}

//Most CLUTs are uploaded with a small host -> local transfer: when the CPU copy of their memory is up to date,
//build them on the CPU (no GPU dispatch, no need to run the queued transfers first).
template <typename Storage>
static bool ReadClutPixels(uint8* ram, const uint8* valid, uint32 bufAddress, uint32 bufWidth,
                           uint32 x0, uint32 y0, uint32 width, uint32 count, uint32* pixels)
{
	CGsPixelFormats::CPixelIndexor<Storage> indexor(ram, bufAddress, bufWidth / 64);
	for(uint32 i = 0; i < count; i++)
	{
		auto address = indexor.GetPixelAddress(x0 + (i % width), y0 + (i / width));
		if(!valid[(reinterpret_cast<uint8*>(address) - ram) / CGSH_WebGPU::CPU_BLOCK_SIZE]) return false;
		pixels[i] = *address;
	}
	return true;
}

bool CGSH_WebGPU::LoadClutFromCpu(uint32 clutIndex, const TEX0& tex0, const TEXCLUT& texClut)
{
	bool idx8 = CGsPixelFormats::IsPsmIDTEX8(tex0.nPsm);
	uint32 bufWidth = (tex0.nCSM == 0) ? 0x40 : texClut.GetBufWidth();
	uint32 x0 = 0, y0 = 0, width = 0, count = 0;
	if(tex0.nCSM == 0)
	{
		width = idx8 ? 16 : 8;
		count = idx8 ? 256 : 16;
	}
	else
	{
		x0 = texClut.GetOffsetU();
		y0 = texClut.GetOffsetV();
		width = count = idx8 ? 256 : 16;
	}
	uint32 pixels[256];
	bool is32 = (tex0.nCPSM == PSMCT32) || (tex0.nCPSM == PSMCT24);
	bool ok = false;
	switch(tex0.nCPSM)
	{
	case PSMCT32:
	case PSMCT24:
		ok = ReadClutPixels<CGsPixelFormats::STORAGEPSMCT32>(m_memoryCache, m_cpuValid.data(), tex0.GetCLUTPtr(), bufWidth, x0, y0, width, count, pixels);
		break;
	case PSMCT16:
		ok = ReadClutPixels<CGsPixelFormats::STORAGEPSMCT16>(m_memoryCache, m_cpuValid.data(), tex0.GetCLUTPtr(), bufWidth, x0, y0, width, count, pixels);
		break;
	case PSMCT16S:
		ok = ReadClutPixels<CGsPixelFormats::STORAGEPSMCT16S>(m_memoryCache, m_cpuValid.data(), tex0.GetCLUTPtr(), bufWidth, x0, y0, width, count, pixels);
		break;
	}
	if(!ok) return false;

	m_clutUploads.resize(m_clutUploads.size() + CLUTENTRYCOUNT + 1);
	uint32* clut = m_clutUploads.data() + m_clutUploads.size() - CLUTENTRYCOUNT;
	clut[-1] = clutIndex;
	//Same mapping as the clutLoad shader (GSH_VulkanClutLoad)
	for(uint32 i = 0; i < count; i++)
	{
		uint32 index = i;
		if(tex0.nCSM == 0)
		{
			if(idx8)
			{
				index = (index & ~0x18) | ((index & 0x08) << 1) | ((index & 0x10) >> 1);
			}
			else
			{
				index += tex0.nCSA * 16;
			}
		}
		if(is32)
		{
			//Whole colors in the first 256 entries, halves swapped at index >= 256 (see the clutLoad shader)
			clut[index & 255] = (index & 256) ? ((pixels[i] >> 16) | (pixels[i] << 16)) : pixels[i];
		}
		else
		{
			clut[index & 511] = pixels[i];
		}
	}
	return true;
}

void CGSH_WebGPU::SyncCLUT(const TEX0& tex0)
{
	if(!CGsPixelFormats::IsPsmIDTEX(tex0.nPsm)) return;
	if(!ProcessCLD(tex0))
	{
		//We haven't loaded anything, but it's possible that this draw call will reference data from a previous load.
		//Tales of Destiny games will do that, loading 256 CLUT entries and then not loading but referencing the entries.
		if((tex0.nCSM == 0) && CGsPixelFormats::IsPsmIDTEX4(tex0.nPsm) &&
		   (m_lastClutFillCacheIndex != -1) && (m_lastClutFillPointer == tex0.nCBP))
		{
			m_clutIndex = m_clutSlots[m_lastClutFillCacheIndex];
		}
		return;
	}

	auto texClut = make_convertible<TEXCLUT>(m_nReg[GS_REG_TEXCLUT]);

	auto clutKey = MakeCachedClutKey(tex0, texClut);
	int32 clutCacheIndex = FindCachedClut(clutKey);
	if(clutCacheIndex == -1)
	{
		uint32 slot = DoClutLoad(tex0, texClut);
		clutCacheIndex = m_nextClutCacheIndex++;
		m_nextClutCacheIndex %= CLUT_CACHE_SIZE;
		m_clutStates[clutCacheIndex] = clutKey;
		m_clutSlots[clutCacheIndex] = slot;
	}

	m_clutIndex = m_clutSlots[clutCacheIndex];

	if(tex0.nCSM == 0)
	{
		if(CGsPixelFormats::IsPsmIDTEX8(tex0.nPsm))
		{
			m_lastClutFillCacheIndex = clutCacheIndex;
			m_lastClutFillPointer = tex0.nCBP;
		}
		else if(m_lastClutFillPointer != tex0.nCBP)
		{
			//If we have a IDTEX4 texture, but it's from a different place, invalidate
			m_lastClutFillCacheIndex = -1;
		}
	}
	else
	{
		m_lastClutFillCacheIndex = -1;
	}
}

/////////////////////////////////////////////////////////////
// Draws (primitive assembly follows GSH_Vulkan; rasterized by the draw shader in GSH_WebGPU.js)
/////////////////////////////////////////////////////////////

//State record bits (STATE_WORDS words, read by the draw shader)
enum
{
	STATE_FLAGS0 = 0,
	STATE_FLAGS1 = 1,
	STATE_TEX_PSM = 2,
	STATE_CLUT_PSM = 3,
	STATE_TEX_BUFADDR = 4,
	STATE_TEX_BUFWIDTH = 5,
	STATE_TEX_WIDTH = 6,
	STATE_TEX_HEIGHT = 7,
	STATE_TEX_MIPLEVEL = 8,
	STATE_TEX_CSA = 9,
	STATE_TEX_A0 = 10,
	STATE_TEX_A1 = 11,
	STATE_CLAMP = 12, //minU, minV, maxU, maxV
	STATE_FB_WRITEMASK = 16,
	STATE_ALPHA_FIX = 17,
	STATE_ALPHA_REF = 18,
	STATE_CLUT_BASE = 19,
	STATE_FOG_COLOR = 20,
	STATE_MIP_MAX = 21,
	STATE_MIP_L = 22,
	STATE_MIP_K = 23,
	STATE_MIP_BUFS = 24, //(address, width) x 6
	STATE_TEX_DECODED = 36, //1 + offset of the decoded texture in the decoded texture buffer, 0: not decoded
};

enum
{
	FLAGS1_MEMORY_COPY = 1 << 4,
};

static uint32 AreaSize(uint32 psm, uint32 width, uint32 height)
{
	CGsCachedArea area;
	area.SetArea(psm, 0, width, height);
	return area.GetSize();
}

void CGSH_WebGPU::ProcessPrim(uint64 data)
{
	unsigned int newPrimitiveType = static_cast<unsigned int>(data & 0x07);
	m_primitiveType = newPrimitiveType;
	switch(m_primitiveType)
	{
	case PRIM_POINT:
		m_vtxCount = 1;
		break;
	case PRIM_LINE:
	case PRIM_LINESTRIP:
		m_vtxCount = 2;
		break;
	case PRIM_TRIANGLE:
	case PRIM_TRIANGLESTRIP:
	case PRIM_TRIANGLEFAN:
		m_vtxCount = 3;
		break;
	case PRIM_SPRITE:
		m_vtxCount = 2;
		break;
	}
}

void CGSH_WebGPU::VertexKick(uint8 registerId, uint64 data)
{
	if(m_pendingPrim)
	{
		m_pendingPrim = false;
		ProcessPrim(m_pendingPrimValue);
	}

	if(m_vtxCount == 0) return;

	bool drawingKick = (registerId == GS_REG_XYZ2) || (registerId == GS_REG_XYZF2);
	bool fog = (registerId == GS_REG_XYZF2) || (registerId == GS_REG_XYZF3);

	if(!m_drawEnabled) drawingKick = false;

	if(fog)
	{
		m_vtxBuffer[m_vtxCount - 1].position = data & 0x00FFFFFFFFFFFFFFULL;
		m_vtxBuffer[m_vtxCount - 1].rgbaq = m_nReg[GS_REG_RGBAQ];
		m_vtxBuffer[m_vtxCount - 1].uv = m_nReg[GS_REG_UV];
		m_vtxBuffer[m_vtxCount - 1].st = m_nReg[GS_REG_ST];
		m_vtxBuffer[m_vtxCount - 1].fog = static_cast<uint8>(data >> 56);
	}
	else
	{
		m_vtxBuffer[m_vtxCount - 1].position = data;
		m_vtxBuffer[m_vtxCount - 1].rgbaq = m_nReg[GS_REG_RGBAQ];
		m_vtxBuffer[m_vtxCount - 1].uv = m_nReg[GS_REG_UV];
		m_vtxBuffer[m_vtxCount - 1].st = m_nReg[GS_REG_ST];
		m_vtxBuffer[m_vtxCount - 1].fog = static_cast<uint8>(m_nReg[GS_REG_FOG] >> 56);
	}

	m_vtxCount--;

	if(m_vtxCount == 0)
	{
		if((m_nReg[GS_REG_PRMODECONT] & 1) != 0)
		{
			m_primitiveMode <<= m_nReg[GS_REG_PRIM];
		}
		else
		{
			m_primitiveMode <<= m_nReg[GS_REG_PRMODE];
		}

		if(drawingKick)
		{
			SetRenderingContext(m_primitiveMode);
		}

		switch(m_primitiveType)
		{
		case PRIM_POINT:
			if(drawingKick) Prim_Point();
			m_vtxCount = 1;
			break;
		case PRIM_LINE:
			if(drawingKick) Prim_Line();
			m_vtxCount = 2;
			break;
		case PRIM_LINESTRIP:
			if(drawingKick) Prim_Line();
			memcpy(&m_vtxBuffer[1], &m_vtxBuffer[0], sizeof(VERTEX));
			m_vtxCount = 1;
			break;
		case PRIM_TRIANGLE:
			if(drawingKick) Prim_Triangle();
			m_vtxCount = 3;
			break;
		case PRIM_TRIANGLESTRIP:
			if(drawingKick) Prim_Triangle();
			memcpy(&m_vtxBuffer[2], &m_vtxBuffer[1], sizeof(VERTEX));
			memcpy(&m_vtxBuffer[1], &m_vtxBuffer[0], sizeof(VERTEX));
			m_vtxCount = 1;
			break;
		case PRIM_TRIANGLEFAN:
			if(drawingKick) Prim_Triangle();
			memcpy(&m_vtxBuffer[1], &m_vtxBuffer[0], sizeof(VERTEX));
			m_vtxCount = 1;
			break;
		case PRIM_SPRITE:
			if(drawingKick) Prim_Sprite();
			m_vtxCount = 2;
			break;
		}
	}
}

static std::pair<uint32, uint32> GetMipLevelInfo(uint32 level, const CGSHandler::MIPTBP1& miptbp1, const CGSHandler::MIPTBP2& miptbp2)
{
	switch(level)
	{
	default:
		assert(false);
		return std::pair<uint32, uint32>(0, 0);
	case 1:
		return std::pair<uint32, uint32>(miptbp1.GetTbp1(), miptbp1.GetTbw1());
	case 2:
		return std::pair<uint32, uint32>(miptbp1.GetTbp2(), miptbp1.GetTbw2());
	case 3:
		return std::pair<uint32, uint32>(miptbp1.GetTbp3(), miptbp1.GetTbw3());
	case 4:
		return std::pair<uint32, uint32>(miptbp2.GetTbp4(), miptbp2.GetTbw4());
	case 5:
		return std::pair<uint32, uint32>(miptbp2.GetTbp5(), miptbp2.GetTbw5());
	case 6:
		return std::pair<uint32, uint32>(miptbp2.GetTbp6(), miptbp2.GetTbw6());
	}
}

//Port of CGSH_Vulkan::SetRenderingContext: the pipeline caps and push constants become a state record
//(STATE_*), the frame/depth buffer become the batch.
void CGSH_WebGPU::SetRenderingContext(uint64 primReg)
{
	auto prim = make_convertible<PRMODE>(primReg);

	unsigned int context = prim.nContext;

	auto offset = make_convertible<XYOFFSET>(m_nReg[GS_REG_XYOFFSET_1 + context]);
	auto frame = make_convertible<FRAME>(m_nReg[GS_REG_FRAME_1 + context]);
	auto zbuf = make_convertible<ZBUF>(m_nReg[GS_REG_ZBUF_1 + context]);
	auto tex0 = make_convertible<TEX0>(m_nReg[GS_REG_TEX0_1 + context]);
	auto tex1 = make_convertible<TEX1>(m_nReg[GS_REG_TEX1_1 + context]);
	auto miptbp1 = make_convertible<MIPTBP1>(m_nReg[GS_REG_MIPTBP1_1 + context]);
	auto miptbp2 = make_convertible<MIPTBP2>(m_nReg[GS_REG_MIPTBP2_1 + context]);
	auto clamp = make_convertible<CLAMP>(m_nReg[GS_REG_CLAMP_1 + context]);
	auto alpha = make_convertible<ALPHA>(m_nReg[GS_REG_ALPHA_1 + context]);
	auto scissor = make_convertible<SCISSOR>(m_nReg[GS_REG_SCISSOR_1 + context]);
	auto test = make_convertible<TEST>(m_nReg[GS_REG_TEST_1 + context]);
	auto texA = make_convertible<TEXA>(m_nReg[GS_REG_TEXA]);
	auto fogCol = make_convertible<FOGCOL>(m_nReg[GS_REG_FOGCOL]);
	auto scanMask = m_nReg[GS_REG_SCANMSK] & 3;
	auto colClamp = m_nReg[GS_REG_COLCLAMP] & 1;
	auto fba = m_nReg[GS_REG_FBA_1 + context] & 1;

	uint32 state[STATE_WORDS] = {};
	bool hasTexture = prim.nTexture != 0;

	bool writeDepth = (zbuf.nMask == 0) && (test.nDepthEnabled != 0); //Depth test disabled -> no writes to depth buffer
	uint32 depthTest = test.nDepthMethod;
	if(!test.nDepthEnabled)
	{
		depthTest = DEPTH_TEST_ALWAYS;
	}
	//Disable depth writes if both frame buffer and z buffer share the same address
	if(frame.GetBasePtr() == zbuf.GetBasePtr())
	{
		writeDepth = false;
	}

	uint32 alphaTest = test.nAlphaMethod;
	uint32 alphaFail = test.nAlphaFail;
	if(!test.nAlphaEnabled)
	{
		alphaTest = ALPHA_TEST_ALWAYS;
	}

	uint32 fbWriteMask = ~frame.nMask;

	//Convert alpha testing to write masking if possible
	if((alphaTest == ALPHA_TEST_NEVER) && (alphaFail == ALPHA_TEST_FAIL_FBONLY))
	{
		alphaTest = ALPHA_TEST_ALWAYS;
		writeDepth = false;
	}
	if((alphaTest == ALPHA_TEST_NEVER) && (alphaFail == ALPHA_TEST_FAIL_RGBONLY))
	{
		alphaTest = ALPHA_TEST_ALWAYS;
		writeDepth = false;
		fbWriteMask &= 0x00FFFFFF;
	}

	switch(frame.nPsm)
	{
	case PSMCT24:
	case PSMZ24:
		fbWriteMask = fbWriteMask & 0x00FFFFFF;
		break;
	case PSMCT16:
	case PSMCT16S:
	case PSMZ16S:
		fbWriteMask = RGBA32ToRGBA16(fbWriteMask) & 0xFFFF;
		break;
	default:
		break;
	}

	uint32 flags0 = 0;
	uint32 flags1 = 0;
	bool needsTextureCopy = false;

	if(hasTexture)
	{
		uint32 texBufPtr = tex0.GetBufPtr();
		uint32 texBufWidth = tex0.GetBufWidth();
		uint32 texMipLevel = 0;
		bool dynamicMipLod = false;

		bool minLinear = false;
		bool magLinear = false;
		switch(tex1.nMinFilter)
		{
		case MIN_FILTER_LINEAR:
		case MIN_FILTER_LINEAR_MIP_NEAREST:
		case MIN_FILTER_LINEAR_MIP_LINEAR:
			minLinear = true;
			break;
		default:
			break;
		}
		magLinear = (tex1.nMagFilter == MAG_FILTER_LINEAR);

		//Ignore min filter if we don't have mipmap levels.
		if(tex1.nMaxMip == 0)
		{
			minLinear = magLinear;
		}
		else if(tex1.nMinFilter >= MIN_FILTER_NEAREST_MIP_NEAREST)
		{
			if(tex1.nLODMethod == LOD_CALC_STATIC)
			{
				int k = trunc(tex1.GetK());
				texMipLevel = std::clamp<int>(k, 0, tex1.nMaxMip);
				if(texMipLevel != 0)
				{
					auto mipLevelInfo = GetMipLevelInfo(texMipLevel, miptbp1, miptbp2);
					texBufPtr = mipLevelInfo.first;
					texBufWidth = mipLevelInfo.second;
				}
			}
			else
			{
				dynamicMipLod = true;
				for(int i = 1; i <= 6; i++)
				{
					auto info = (i <= tex1.nMaxMip) ? GetMipLevelInfo(i, miptbp1, miptbp2) : std::pair<uint32, uint32>(0, 0);
					state[STATE_MIP_BUFS + (i - 1) * 2 + 0] = info.first;
					state[STATE_MIP_BUFS + (i - 1) * 2 + 1] = info.second;
				}
				float lodK = tex1.GetK();
				state[STATE_MIP_MAX] = tex1.nMaxMip;
				state[STATE_MIP_L] = tex1.nLODL;
				memcpy(&state[STATE_MIP_K], &lodK, sizeof(float));
			}
		}

		flags0 |= 1 << 0;
		flags0 |= (tex0.nColorComp & 1) << 1;
		flags0 |= (texA.nAEM & 1) << 2;
		flags0 |= (tex0.nFunction & 3) << 3;
		flags0 |= ((minLinear && magLinear) ? 1 : 0) << 5;
		flags0 |= (dynamicMipLod ? 1 : 0) << 6;
		flags0 |= (clamp.nWMS & 3) << 7;
		flags0 |= (clamp.nWMT & 3) << 9;
		flags1 |= (prim.nUseUV ? 1 : 0) << 5;

		state[STATE_TEX_PSM] = tex0.nPsm;
		state[STATE_CLUT_PSM] = CGsPixelFormats::IsPsmIDTEX(tex0.nPsm) ? tex0.nCPSM : 0;
		state[STATE_TEX_BUFADDR] = texBufPtr;
		state[STATE_TEX_BUFWIDTH] = texBufWidth;
		state[STATE_TEX_WIDTH] = tex0.GetWidth();
		state[STATE_TEX_HEIGHT] = tex0.GetHeight();
		state[STATE_TEX_MIPLEVEL] = texMipLevel;
		state[STATE_TEX_CSA] = tex0.nCSA * 0x10;
		state[STATE_TEX_A0] = texA.nTA0;
		state[STATE_TEX_A1] = texA.nTA1;
		state[STATE_CLAMP + 0] = clamp.GetMinU();
		state[STATE_CLAMP + 1] = clamp.GetMinV();
		state[STATE_CLAMP + 2] = clamp.GetMaxU();
		state[STATE_CLAMP + 3] = clamp.GetMaxV();
		state[STATE_CLUT_BASE] = CGsPixelFormats::IsPsmIDTEX(tex0.nPsm) ? m_clutIndex * CLUTENTRYCOUNT : 0;

		//Primitives sampling the buffer they draw to read a copy of the memory (GSH_Vulkan's textureUseMemoryCopy,
		//there for sprites only: other primitives rely on fragment interlock there; here they would race with the
		//writes of other tiles)
		{
			//Texture vs the frame/depth buffer area this state writes (an upper byte texture under a 24-bit buffer is
			//never written)
			uint32 texStart = tex0.GetBufPtr();
			uint32 texEnd = texStart + AreaSize(tex0.nPsm, tex0.GetBufWidth(), tex0.GetHeight());
			uint32 height = scissor.scay1 + 1;
			uint32 frameBufPtr = frame.GetBasePtr();
			uint32 depthBufPtr = zbuf.GetBasePtr();
			bool isTexUpperBytePsm = CGsPixelFormats::IsPsmUpperByte(tex0.nPsm);
			needsTextureCopy |= RangesOverlap(texStart, texEnd, frameBufPtr, frameBufPtr + AreaSize(frame.nPsm, frame.GetWidth(), height)) &&
			                    !(isTexUpperBytePsm && CGsPixelFormats::IsPsm24Bits(frame.nPsm));
			if(writeDepth)
			{
				needsTextureCopy |= RangesOverlap(texStart, texEnd, depthBufPtr, depthBufPtr + AreaSize(zbuf.nPsm | 0x30, frame.GetWidth(), height)) &&
				                    !(isTexUpperBytePsm && CGsPixelFormats::IsPsm24Bits(zbuf.nPsm));
			}
			if(needsTextureCopy)
			{
				flags1 |= FLAGS1_MEMORY_COPY;
			}
		}
	}

	flags0 |= (prim.nFog ? 1 : 0) << 11;
	flags0 |= (writeDepth ? 1 : 0) << 12;
	flags0 |= scanMask << 13;
	if(prim.nAlpha)
	{
		flags0 |= 1 << 15;
		flags0 |= (alpha.nA & 3) << 16;
		flags0 |= (alpha.nB & 3) << 18;
		flags0 |= (alpha.nC & 3) << 20;
		flags0 |= (alpha.nD & 3) << 22;
		state[STATE_ALPHA_FIX] = alpha.nFix;
	}
	flags0 |= colClamp << 24;
	flags0 |= fba << 25;
	flags0 |= (depthTest & 3) << 26;
	flags0 |= (alphaTest & 7) << 28;
	flags1 |= (alphaFail & 3) << 0;
	flags1 |= (test.nDestAlphaEnabled ? 1 : 0) << 2;
	flags1 |= (test.nDestAlphaMode ? 1 : 0) << 3;

	state[STATE_FLAGS0] = flags0;
	state[STATE_FLAGS1] = flags1;
	state[STATE_FB_WRITEMASK] = fbWriteMask;
	state[STATE_ALPHA_REF] = test.nAlphaRef;
	if(prim.nFog)
	{
		state[STATE_FOG_COLOR] = fogCol.nFCR | (fogCol.nFCG << 8) | (fogCol.nFCB << 16);
	}

	m_scissor[0] = scissor.scax0;
	m_scissor[1] = scissor.scay0;
	m_scissor[2] = scissor.scax1;
	m_scissor[3] = scissor.scay1;

	//Frame/depth buffer: one batch per pair
	uint32 fbPsm = frame.nPsm;
	uint32 zbPsm = zbuf.nPsm | 0x30;
	if(
	    (m_batch.fbPsm != fbPsm) || (m_batch.fbAddress != frame.GetBasePtr()) || (m_batch.fbWidth != frame.GetWidth()) ||
	    (m_batch.zbPsm != zbPsm) || (m_batch.zbAddress != zbuf.GetBasePtr()))
	{
		FlushBatch();
		m_batch.fbPsm = fbPsm;
		m_batch.fbAddress = frame.GetBasePtr();
		m_batch.fbWidth = frame.GetWidth();
		m_batch.zbPsm = zbPsm;
		m_batch.zbAddress = zbuf.GetBasePtr();
		m_batch.writeStart = m_batch.writeEnd = m_batch.fbAddress;
		m_batch.zbWriteStart = m_batch.zbWriteEnd = m_batch.zbAddress;
	}
	//Memory this state can write (grows with the scissor). If the batch's earlier primitives read textures there,
	//later primitives writing it would race with those reads (other tiles): start a new batch.
	uint32 height = scissor.scay1 + 1;
	uint32 writeEnd = std::max(m_batch.writeEnd, m_batch.fbAddress + AreaSize(fbPsm, frame.GetWidth(), height));
	uint32 zbWriteEnd = writeDepth ? std::max(m_batch.zbWriteEnd, m_batch.zbAddress + AreaSize(zbPsm, frame.GetWidth(), height)) : m_batch.zbWriteEnd;
	auto readsOverlap = [&](uint32 start, uint32 end) {
		return std::any_of(m_batchReads.begin(), m_batchReads.end(),
		                   [&](const auto& range) { return RangesOverlap(start, end, range.first, range.second); });
	};
	if(((writeEnd != m_batch.writeEnd) && readsOverlap(m_batch.writeStart, writeEnd)) ||
	   ((zbWriteEnd != m_batch.zbWriteEnd) && readsOverlap(m_batch.zbWriteStart, zbWriteEnd)))
	{
		FlushBatch();
	}
	m_batch.writeEnd = std::max(m_batch.writeEnd, writeEnd);
	m_batch.zbWriteEnd = std::max(m_batch.zbWriteEnd, zbWriteEnd);

	if(memcmp(state, m_state, sizeof(state)) != 0)
	{
		bool batchHasPrims = (m_prims.size() / PRIM_WORDS) != m_batch.primStart;
		if(hasTexture && batchHasPrims)
		{
			//A texture that overlaps memory written by this batch must see those writes: start a new dispatch
			//(same guarantee as the barrier between draws in the Vulkan backend).
			uint32 texStart = tex0.GetBufPtr();
			uint32 texEnd = texStart + AreaSize(tex0.nPsm, tex0.GetBufWidth(), tex0.GetHeight());
			if(RangesOverlap(texStart, texEnd, m_batch.writeStart, m_batch.writeEnd) ||
			   RangesOverlap(texStart, texEnd, m_batch.zbWriteStart, m_batch.zbWriteEnd))
			{
				FlushBatch();
			}
		}
		if(needsTextureCopy)
		{
			CGsCachedArea textureArea;
			textureArea.SetArea(tex0.nPsm, tex0.nBufPtr, tex0.GetBufWidth(), tex0.GetHeight());
			m_memoryCopyAddress = tex0.GetBufPtr();
			m_memoryCopySize = std::min<uint32>(textureArea.GetSize(), RAMSIZE - m_memoryCopyAddress);
			CopyMemoryForTexture();
		}
		memcpy(m_state, state, sizeof(state));
		m_stateIndex = -1;
		m_stateInBatch = false;
		m_stateReadStart = m_stateReadEnd = 0;
		if(hasTexture)
		{
			//Memory the texture (all its mip levels) can read
			m_stateReadStart = tex0.GetBufPtr();
			m_stateReadEnd = m_stateReadStart + AreaSize(tex0.nPsm, tex0.GetBufWidth(), tex0.GetHeight());
			for(uint32 level = 1; level <= tex1.nMaxMip; level++)
			{
				auto info = GetMipLevelInfo(level, miptbp1, miptbp2);
				m_stateReadStart = std::min(m_stateReadStart, info.first);
				m_stateReadEnd = std::max(m_stateReadEnd, info.first + AreaSize(tex0.nPsm, info.second, std::max(tex0.GetHeight() >> level, 1U)));
			}
		}
		uint32 features = 0;
		if(hasTexture)
		{
			features |= DRAW_FEATURE_TEXTURE;
			if(flags0 & (1 << 5)) features |= DRAW_FEATURE_LINEAR;
			if(flags0 & (1 << 6)) features |= DRAW_FEATURE_MIPLOD;
			if(CGsPixelFormats::IsPsmIDTEX(tex0.nPsm))
			{
				features |= ((tex0.nCPSM == PSMCT16) || (tex0.nCPSM == PSMCT16S)) ? DRAW_FEATURE_CLUT16 : DRAW_FEATURE_CLUT32;
			}
			switch(tex0.nPsm)
			{
			case PSMCT16:
			case PSMCT16S:
			case PSMZ16:
			case PSMZ16S:
				features |= DRAW_FEATURE_TEX16;
				break;
			case PSMT8:
			case PSMT8H:
				features |= DRAW_FEATURE_TEX8;
				break;
			case PSMT4:
			case PSMT4HL:
			case PSMT4HH:
				features |= DRAW_FEATURE_TEX4;
				break;
			default:
				features |= DRAW_FEATURE_TEX32;
				break;
			}
			if(needsTextureCopy) features |= DRAW_FEATURE_MEMCOPY;
		}
		if(prim.nAlpha) features |= DRAW_FEATURE_BLEND;
		if(prim.nFog) features |= DRAW_FEATURE_FOG;
		if(alphaTest != ALPHA_TEST_ALWAYS) features |= DRAW_FEATURE_ALPHATEST;
		if(test.nDestAlphaEnabled) features |= DRAW_FEATURE_DATE;
		if((depthTest == DEPTH_TEST_GEQUAL) || (depthTest == DEPTH_TEST_GREATER)) features |= DRAW_FEATURE_DEPTHTEST;
		if(writeDepth) features |= DRAW_FEATURE_DEPTHWRITE;
		m_stateFeatures = features;
	}

	m_fbBasePtr = frame.GetBasePtr();
	m_primOfsX = offset.nOffsetX;
	m_primOfsY = offset.nOffsetY;
	m_texWidth = tex0.GetWidth();
	m_texHeight = tex0.GetHeight();
}

void CGSH_WebGPU::CopyMemoryForTexture()
{
	FlushBatch();
	FlushTransfers();
	PushCommand({OP_MEMORY_TO_COPY, m_memoryCopyAddress, m_memoryCopySize});
	m_memoryCopyRegion.Reset();
}

void CGSH_WebGPU::Prim_Point()
{
	auto xyz = make_convertible<XYZ>(m_vtxBuffer[0].position);
	auto rgbaq = make_convertible<RGBAQ>(m_vtxBuffer[0].rgbaq);

	VERTEX_DATA vertex;
	vertex.x = static_cast<int32>(xyz.nX) - m_primOfsX;
	vertex.y = static_cast<int32>(xyz.nY) - m_primOfsY;
	vertex.z = xyz.nZ;
	vertex.color = MakeColor(rgbaq.nR, rgbaq.nG, rgbaq.nB, rgbaq.nA);
	vertex.fog = m_vtxBuffer[0].fog;
	AddPrim(PRIM_TYPE_POINT, &vertex, 1);
}

void CGSH_WebGPU::Prim_Line()
{
	VERTEX_DATA vertices[2];
	for(int i = 0; i < 2; i++)
	{
		const auto& src = m_vtxBuffer[1 - i];
		auto xyz = make_convertible<XYZ>(src.position);
		auto rgbaq = make_convertible<RGBAQ>(src.rgbaq);
		auto& dst = vertices[i];
		dst.x = static_cast<int32>(xyz.nX) - m_primOfsX;
		dst.y = static_cast<int32>(xyz.nY) - m_primOfsY;
		dst.z = xyz.nZ;
		dst.color = MakeColor(rgbaq.nR, rgbaq.nG, rgbaq.nB, rgbaq.nA);
		dst.fog = src.fog;
		if(m_primitiveMode.nTexture)
		{
			if(m_primitiveMode.nUseUV)
			{
				auto uv = make_convertible<UV>(src.uv);
				dst.s = uv.GetU();
				dst.t = uv.GetV();
			}
			else
			{
				auto st = make_convertible<ST>(src.st);
				dst.s = st.nS;
				dst.t = st.nT;
				dst.q = rgbaq.nQ;
			}
		}
	}
	if(m_primitiveMode.nTexture && m_primitiveMode.nUseUV)
	{
		m_lastLineU = vertices[0].s / static_cast<float>(m_texWidth);
		m_lastLineV = vertices[0].t / static_cast<float>(m_texHeight);
	}
	AddPrim(PRIM_TYPE_LINE, vertices, 2);
}

void CGSH_WebGPU::Prim_Triangle()
{
	VERTEX_DATA vertices[3];
	for(int i = 0; i < 3; i++)
	{
		const auto& src = m_vtxBuffer[2 - i];
		auto xyz = make_convertible<XYZ>(src.position);
		auto rgbaq = make_convertible<RGBAQ>(src.rgbaq);
		auto& dst = vertices[i];
		dst.x = static_cast<int32>(xyz.nX) - m_primOfsX;
		dst.y = static_cast<int32>(xyz.nY) - m_primOfsY;
		dst.z = xyz.nZ;
		dst.color = MakeColor(rgbaq.nR, rgbaq.nG, rgbaq.nB, rgbaq.nA);
		dst.fog = src.fog;
		if(m_primitiveMode.nTexture)
		{
			if(m_primitiveMode.nUseUV)
			{
				auto uv = make_convertible<UV>(src.uv);
				dst.s = uv.GetU();
				dst.t = uv.GetV();
			}
			else
			{
				auto st = make_convertible<ST>(src.st);
				dst.s = st.nS;
				dst.t = st.nT;
				dst.q = rgbaq.nQ;
			}
		}
	}
	if(m_primitiveMode.nShading == 0)
	{
		//Flat shaded triangles use the last color set
		vertices[0].color = vertices[1].color = vertices[2].color;
	}
	AddPrim(PRIM_TYPE_TRIANGLE, vertices, 3);
}

void CGSH_WebGPU::Prim_Sprite()
{
	VERTEX_DATA vertices[2];
	auto rgbaq = make_convertible<RGBAQ>(m_vtxBuffer[0].rgbaq);
	auto xyzLast = make_convertible<XYZ>(m_vtxBuffer[0].position);
	for(int i = 0; i < 2; i++)
	{
		const auto& src = m_vtxBuffer[1 - i];
		auto xyz = make_convertible<XYZ>(src.position);
		auto& dst = vertices[i];
		dst.x = static_cast<int32>(xyz.nX) - m_primOfsX;
		dst.y = static_cast<int32>(xyz.nY) - m_primOfsY;
		dst.z = xyzLast.nZ;
		dst.color = MakeColor(rgbaq.nR, rgbaq.nG, rgbaq.nB, rgbaq.nA);
		dst.fog = m_vtxBuffer[0].fog;
		if(m_primitiveMode.nTexture)
		{
			if(m_primitiveMode.nUseUV)
			{
				auto uv = make_convertible<UV>(src.uv);
				dst.s = uv.GetU();
				dst.t = uv.GetV();
			}
			else
			{
				auto st = make_convertible<ST>(src.st);
				float q = make_convertible<RGBAQ>(src.rgbaq).nQ;
				if(q == 0) q = 1;
				dst.s = st.nS / q;
				dst.t = st.nT / q;
			}
		}
	}

	CGsSpriteRect rect(vertices[0].x / 16.f, vertices[0].y / 16.f, vertices[1].x / 16.f, vertices[1].y / 16.f);
	CheckSpriteCachedClutInvalidation(rect);

	if(m_state[STATE_FLAGS1] & FLAGS1_MEMORY_COPY)
	{
		//Some games use tiny sprites for full screen effects that sample their own frame buffer
		//(MGS3, MK: Shaolin Monks, Tales of Legendia): the copy stays valid until a sprite overlaps
		//an area drawn since it was made (see CDraw::AddVertices).
		if(m_memoryCopyRegion.Intersects(rect))
		{
			CopyMemoryForTexture();
		}
		m_memoryCopyRegion.Insert(rect);
	}

	AddPrim(PRIM_TYPE_SPRITE, vertices, 2);
}

void CGSH_WebGPU::CheckSpriteCachedClutInvalidation(const CGsSpriteRect& rect)
{
	//Gran Turismo 4 writes to an area in GS RAM then uses it as a CLUT (see GSH_Vulkan).
	if((rect.left == 0) && (rect.top == 0) && (rect.GetWidth() == 8) && (rect.GetHeight() == 16))
	{
		for(uint32 i = 0; i < CLUT_CACHE_SIZE; i++)
		{
			uint32 cbpBasePtr = m_clutStates[i].cbp * 0x100;
			if(cbpBasePtr == m_fbBasePtr)
			{
				memset(&m_clutStates, 0, sizeof(m_clutStates));
				FlushBatch();
				break;
			}
		}
	}
}

void CGSH_WebGPU::AddPrim(PRIM_TYPE type, const VERTEX_DATA* vertices, uint32 vertexCount)
{
	int32 minX = INT32_MAX, minY = INT32_MAX, maxX = INT32_MIN, maxY = INT32_MIN;
	for(uint32 i = 0; i < vertexCount; i++)
	{
		minX = std::min(minX, vertices[i].x);
		minY = std::min(minY, vertices[i].y);
		maxX = std::max(maxX, vertices[i].x);
		maxY = std::max(maxY, vertices[i].y);
	}
	//Pixels whose sample point (integer coordinate) can be covered
	switch(type)
	{
	case PRIM_TYPE_SPRITE:
		minX = (minX + 15) >> 4;
		minY = (minY + 15) >> 4;
		maxX = ((maxX + 15) >> 4) - 1;
		maxY = ((maxY + 15) >> 4) - 1;
		break;
	case PRIM_TYPE_POINT:
		minX = maxX = (minX + 8) >> 4;
		minY = maxY = (minY + 8) >> 4;
		break;
	case PRIM_TYPE_LINE:
		minX = minX >> 4;
		minY = minY >> 4;
		maxX = (maxX + 8) >> 4;
		maxY = (maxY + 8) >> 4;
		break;
	default:
		minX = (minX + 15) >> 4;
		minY = (minY + 15) >> 4;
		maxX = maxX >> 4;
		maxY = maxY >> 4;
		break;
	}
	minX = std::max<int32>(minX, m_scissor[0]);
	minY = std::max<int32>(minY, m_scissor[1]);
	maxX = std::min<int32>(maxX, std::min<uint32>(m_scissor[2], 2047));
	maxY = std::min<int32>(maxY, std::min<uint32>(m_scissor[3], 2047));
	if((minX > maxX) || (minY > maxY)) return;

	if(m_stateIndex < 0)
	{
		m_stateIndex = m_states.size() / STATE_WORDS;
		m_states.insert(m_states.end(), std::begin(m_state), std::end(m_state));
		m_stateAreas.push_back(0);
		m_stateFormats.push_back(m_stateFeatures & DRAW_FEATURE_TEXFORMATS);
	}
	m_stateAreas[m_stateIndex] += static_cast<uint64>(maxX - minX + 1) * (maxY - minY + 1);
	if(!m_stateInBatch)
	{
		m_stateInBatch = true;
		if(m_stateReadEnd != m_stateReadStart)
		{
			m_batchReads.emplace_back(m_stateReadStart, m_stateReadEnd);
		}
	}

	size_t base = m_prims.size();
	m_prims.resize(base + PRIM_WORDS);
	uint32* prim = m_prims.data() + base;
	for(uint32 i = 0; i < 3; i++)
	{
		const auto& vertex = vertices[std::min(i, vertexCount - 1)];
		uint32* dst = prim + i * 8;
		dst[0] = static_cast<uint32>(vertex.x);
		dst[1] = static_cast<uint32>(vertex.y);
		dst[2] = vertex.z;
		dst[3] = vertex.color;
		memcpy(&dst[4], &vertex.s, sizeof(float));
		memcpy(&dst[5], &vertex.t, sizeof(float));
		memcpy(&dst[6], &vertex.q, sizeof(float));
		memcpy(&dst[7], &vertex.fog, sizeof(float));
	}
	static const uint32 primTypeFeatures[] = {DRAW_FEATURE_TRIANGLE, DRAW_FEATURE_SPRITE, DRAW_FEATURE_LINE, 0};
	m_batch.features |= m_stateFeatures | primTypeFeatures[type];
	prim[24] = type;
	prim[25] = m_stateIndex;
	prim[26] = minX | (minY << 16);
	prim[27] = maxX | (maxY << 16);

	m_batch.minX = std::min<uint32>(m_batch.minX, minX);
	m_batch.minY = std::min<uint32>(m_batch.minY, minY);
	m_batch.maxX = std::max<uint32>(m_batch.maxX, maxX);
	m_batch.maxY = std::max<uint32>(m_batch.maxY, maxY);
}

//Whether a triangle can't cover any pixel of the tile whose first pixel is (tileX, tileY): all the tile's sample
//points are outside one of its edges (conservative). Same edge functions as the draw shader (12.4 coordinates).
static bool TriangleMissesTile(const uint32* prim, int32 tileX, int32 tileY)
{
	int64 x0 = static_cast<int32>(prim[0]), y0 = static_cast<int32>(prim[1]);
	int64 x1 = static_cast<int32>(prim[8]), y1 = static_cast<int32>(prim[9]);
	int64 x2 = static_cast<int32>(prim[16]), y2 = static_cast<int32>(prim[17]);
	int64 area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
	if(area == 0) return true;
	int64 o = (area > 0) ? 1 : -1;
	int64 minX = tileX * 16, maxX = (tileX + CGSH_WebGPU::DRAW_TILE_SIZE - 1) * 16;
	int64 minY = tileY * 16, maxY = (tileY + CGSH_WebGPU::DRAW_TILE_SIZE - 1) * 16;
	const int64 ax[3] = {x1, x2, x0}, ay[3] = {y1, y2, y0}, bx[3] = {x2, x0, x1}, by[3] = {y2, y0, y1};
	for(uint32 i = 0; i < 3; i++)
	{
		int64 dx = o * (bx[i] - ax[i]), dy = o * (by[i] - ay[i]);
		//e = dx * (sy - ay) - dy * (sx - ax), largest at this corner of the tile
		int64 sx = (dy < 0) ? maxX : minX;
		int64 sy = (dx > 0) ? maxY : minY;
		if((dx * (sy - ay[i]) - dy * (sx - ax[i])) < 0) return true;
	}
	return false;
}

void CGSH_WebGPU::FlushBatch()
{
	FlushTransfers();
	uint32 primEnd = m_prims.size() / PRIM_WORDS;
	m_batchReads.clear();
	m_stateInBatch = false;
	if(primEnd == m_batch.primStart) return;

	//Bin the primitives per tile (in drawing order), so that each workgroup of the draw shader only walks
	//the primitives that can touch its tile.
	uint32 originX = m_batch.minX & ~15U;
	uint32 originY = m_batch.minY & ~15U;
	uint32 tilesX = (m_batch.maxX - originX) / DRAW_TILE_SIZE + 1;
	uint32 tilesY = (m_batch.maxY - originY) / DRAW_TILE_SIZE + 1;
	uint32 tileCount = tilesX * tilesY;
	m_binOffsets.assign(tileCount + 1, 0);
	for(uint32 p = m_batch.primStart; p < primEnd; p++)
	{
		const uint32* prim = m_prims.data() + p * PRIM_WORDS;
		uint32 tx0 = ((prim[26] & 0xFFFF) - originX) / DRAW_TILE_SIZE, ty0 = ((prim[26] >> 16) - originY) / DRAW_TILE_SIZE;
		uint32 tx1 = ((prim[27] & 0xFFFF) - originX) / DRAW_TILE_SIZE, ty1 = ((prim[27] >> 16) - originY) / DRAW_TILE_SIZE;
		bool test = (prim[24] == PRIM_TYPE_TRIANGLE) && ((tx0 != tx1) || (ty0 != ty1));
		for(uint32 ty = ty0; ty <= ty1; ty++)
		{
			for(uint32 tx = tx0; tx <= tx1; tx++)
			{
				if(test && TriangleMissesTile(prim, originX + tx * DRAW_TILE_SIZE, originY + ty * DRAW_TILE_SIZE)) continue;
				m_binOffsets[ty * tilesX + tx + 1]++;
			}
		}
	}
	//Workgroups only for the tiles with primitives, the longest lists first: a tile's list is walked by one
	//workgroup, so a long list started late would run alone at the end of the dispatch.
	m_tileOrder.clear();
	for(uint32 t = 0; t < tileCount; t++)
	{
		if(m_binOffsets[t + 1] != 0) m_tileOrder.push_back(t);
	}
	std::stable_sort(m_tileOrder.begin(), m_tileOrder.end(),
	                 [&](uint32 a, uint32 b) { return m_binOffsets[a + 1] > m_binOffsets[b + 1]; });
	for(uint32 t = 0; t < tileCount; t++)
	{
		m_binOffsets[t + 1] += m_binOffsets[t];
	}
	uint32 entryCount = m_binOffsets[tileCount];
	uint32 orderCount = m_tileOrder.size();
	uint32 binOffset = AllocUpload((tileCount + 1 + entryCount + orderCount) * sizeof(uint32), 4);
	auto bins = reinterpret_cast<uint32*>(m_upload.data() + binOffset);
	memcpy(bins, m_binOffsets.data(), (tileCount + 1) * sizeof(uint32));
	uint32* entries = bins + tileCount + 1;
	memcpy(entries + entryCount, m_tileOrder.data(), orderCount * sizeof(uint32));
	for(uint32 p = m_batch.primStart; p < primEnd; p++)
	{
		const uint32* prim = m_prims.data() + p * PRIM_WORDS;
		uint32 tx0 = ((prim[26] & 0xFFFF) - originX) / DRAW_TILE_SIZE, ty0 = ((prim[26] >> 16) - originY) / DRAW_TILE_SIZE;
		uint32 tx1 = ((prim[27] & 0xFFFF) - originX) / DRAW_TILE_SIZE, ty1 = ((prim[27] >> 16) - originY) / DRAW_TILE_SIZE;
		bool test = (prim[24] == PRIM_TYPE_TRIANGLE) && ((tx0 != tx1) || (ty0 != ty1));
		for(uint32 ty = ty0; ty <= ty1; ty++)
		{
			for(uint32 tx = tx0; tx <= tx1; tx++)
			{
				if(test && TriangleMissesTile(prim, originX + tx * DRAW_TILE_SIZE, originY + ty * DRAW_TILE_SIZE)) continue;
				entries[m_binOffsets[ty * tilesX + tx]++] = p;
			}
		}
	}

	DecodeBatchTextures();

	uint32 paramsOffset = 0;
	auto params = AllocParams(paramsOffset);
	params[0] = m_batch.fbPsm;
	params[1] = m_batch.fbAddress;
	params[2] = m_batch.fbWidth;
	params[3] = m_batch.zbPsm;
	params[4] = m_batch.zbAddress;
	params[7] = originX;
	params[8] = originY;
	params[9] = m_batch.maxX;
	params[10] = m_batch.maxY;
	params[11] = tilesX;
	params[12] = binOffset / 4;
	params[13] = binOffset / 4 + tileCount + 1;
	params[14] = binOffset / 4 + tileCount + 1 + entryCount;
	params[15] = orderCount;
	uint32 groupsX = std::min<uint32>(orderCount, DRAW_GROUPS_PER_ROW);
	uint32 groupsY = (orderCount + DRAW_GROUPS_PER_ROW - 1) / DRAW_GROUPS_PER_ROW;
	PushCommand({OP_DISPATCH, PIPELINE_DRAW | (m_batch.features << 8), paramsOffset, groupsX, groupsY, 1});
	InvalidateCpuMemory(m_batch.writeStart, m_batch.writeEnd);
	InvalidateCpuMemory(m_batch.zbWriteStart, m_batch.zbWriteEnd);
	m_drawCallCount++;

	m_batch.primStart = primEnd;
	//Next primitives get new state records (their decoded textures are the next batch's)
	m_batch.stateStart = m_states.size() / STATE_WORDS;
	m_stateIndex = -1;
	m_batch.minX = m_batch.minY = ~0U;
	m_batch.maxX = m_batch.maxY = 0;
	m_batch.features = 0;
	m_batch.writeStart = m_batch.writeEnd = m_batch.fbAddress;
	m_batch.zbWriteStart = m_batch.zbWriteEnd = m_batch.zbAddress;
	//Submit big frames in parts: the GPU starts on the first draws while the rest of the frame is processed
	if((m_prims.size() / PRIM_WORDS) >= SUBMIT_PRIMS)
	{
		Flush();
	}
}

//Textures the batch reads a lot are decoded first, by a dispatch before the draw: texels as 32-bit colors (CLUT and
//TEXA applied) in the decoded texture buffer, where the draw shader reads them with one load instead of swizzle table,
//GS memory and CLUT reads. The batch can't write them (see SetRenderingContext), so they stay valid for its whole
//dispatch. Only level 0 of repeat/clamp textures read from GS memory (their coordinates stay in the texture).
void CGSH_WebGPU::DecodeBatchTextures()
{
	if(!(m_batch.features & DRAW_FEATURE_TEXTURE)) return;
	static const uint32 keyWords[] = {STATE_TEX_PSM, STATE_CLUT_PSM, STATE_TEX_BUFADDR, STATE_TEX_BUFWIDTH, STATE_TEX_WIDTH,
	                                  STATE_TEX_HEIGHT, STATE_TEX_CSA, STATE_TEX_A0, STATE_TEX_A1, STATE_CLUT_BASE};
	struct TEXTURE
	{
		uint32 state; //first state record using it
		uint64 area;  //bounding box area of the primitives using it
		uint32 offset;
	};
	std::vector<TEXTURE> textures;
	uint32 stateEnd = m_states.size() / STATE_WORDS;
	std::vector<int32> stateTexture(stateEnd - m_batch.stateStart, -1);
	auto sameTexture = [&](const uint32* a, const uint32* b) {
		if(((a[STATE_FLAGS0] ^ b[STATE_FLAGS0]) & (1 << 2)) != 0) return false; //AEM
		return std::all_of(std::begin(keyWords), std::end(keyWords), [&](uint32 w) { return a[w] == b[w]; });
	};
	for(uint32 i = m_batch.stateStart; i < stateEnd; i++)
	{
		const uint32* state = m_states.data() + i * STATE_WORDS;
		uint32 flags0 = state[STATE_FLAGS0];
		bool decodable = (flags0 & 1) && !(flags0 & (1 << 6)) && (state[STATE_TEX_MIPLEVEL] == 0) &&
		                 (((flags0 >> 7) & 3) <= 1) && (((flags0 >> 9) & 3) <= 1) && !(state[STATE_FLAGS1] & FLAGS1_MEMORY_COPY);
		if(!decodable) continue;
		auto texture = std::find_if(textures.begin(), textures.end(),
		                            [&](const TEXTURE& t) { return sameTexture(m_states.data() + t.state * STATE_WORDS, state); });
		if(texture == textures.end())
		{
			texture = textures.insert(textures.end(), {i, 0, 0});
		}
		texture->area += m_stateAreas[i];
		stateTexture[i - m_batch.stateStart] = texture - textures.begin();
	}
	//Worth it when the primitives cover more pixels than the texture has texels
	uint32 jobCount = 0;
	uint32 groups = 0;
	m_decodeJobs.clear();
	for(auto& texture : textures)
	{
		const uint32* state = m_states.data() + texture.state * STATE_WORDS;
		uint32 texels = state[STATE_TEX_WIDTH] * state[STATE_TEX_HEIGHT];
		if((texture.area < texels) || ((m_decodeTexels + texels) > TEX_DECODE_TEXELS)) continue;
		texture.offset = m_decodeTexels + 1;
		m_decodeTexels += texels;
		m_decodeJobs.insert(m_decodeJobs.end(), {texture.state, texture.offset - 1, state[STATE_TEX_WIDTH], texels, groups});
		groups += (texels + 255) / 256;
		jobCount++;
	}
	if(jobCount == 0) return;
	uint32 formats = 0;
	for(uint32 i = m_batch.stateStart; i < stateEnd; i++)
	{
		int32 texture = stateTexture[i - m_batch.stateStart];
		uint32 offset = (texture >= 0) ? textures[texture].offset : 0;
		m_states[i * STATE_WORDS + STATE_TEX_DECODED] = offset;
		if(offset == 0) formats |= m_stateFormats[i];
	}
	m_batch.features = (m_batch.features & ~DRAW_FEATURE_TEXFORMATS) | formats | DRAW_FEATURE_TEXDECODED;

	uint32 tableOffset = AllocUpload(m_decodeJobs.size() * sizeof(uint32), 4);
	memcpy(m_upload.data() + tableOffset, m_decodeJobs.data(), m_decodeJobs.size() * sizeof(uint32));
	uint32 paramsOffset = 0;
	auto params = AllocParams(paramsOffset);
	params[0] = jobCount;
	params[1] = tableOffset / 4;
	Dispatch(PIPELINE_TEX_DECODE, paramsOffset, groups);
}
