#pragma once

#include <climits>
#include <cstring>
#include "Types.h"
#include "Convertible.h"
#include "Vpu.h"
#include "../uint128.h"
#include "../Profiler.h"
#include "zip/ZipArchiveWriter.h"
#include "zip/ZipArchiveReader.h"
#include "SimdDefs.h"

#ifdef FRAMEWORK_SIMD_USE_SSE
#include <emmintrin.h>
#endif

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#ifdef _MSC_VER
#define FRAMEWORK_SAFE_BUFFERS __declspec(safebuffers)
#else
#define FRAMEWORK_SAFE_BUFFERS
#endif

class CINTC;
class CVu1Thread;

class CVif
{
public:
	//ROLE_FRONT/ROLE_BACK are used when VU1 runs on its own thread (see Vu1Thread.h)
	enum ROLE
	{
		ROLE_NORMAL,
		ROLE_FRONT,
		ROLE_BACK,
	};

	enum
	{
		REGS0_START = 0x10003800,
		VIF0_STAT = 0x10003800,
		VIF0_FBRST = 0x10003810,
		VIF0_ERR = 0x10003820,
		VIF0_MARK = 0x10003830,
		VIF0_CYCLE = 0x10003840,
		VIF0_MODE = 0x10003850,
		VIF0_NUM = 0x10003860,
		VIF0_MASK = 0x10003870,
		VIF0_CODE = 0x10003880,
		VIF0_R0 = 0x10003900,
		VIF0_R1 = 0x10003910,
		VIF0_R2 = 0x10003920,
		VIF0_R3 = 0x10003930,
		REGS0_END = 0x10003A00,

		REGS1_START = 0x10003C00,
		VIF1_STAT = 0x10003C00,
		VIF1_FBRST = 0x10003C10,
		VIF1_ERR = 0x10003C20,
		VIF1_MARK = 0x10003C30,
		VIF1_CYCLE = 0x10003C40,
		VIF1_MODE = 0x10003C50,
		VIF1_NUM = 0x10003C60,
		VIF1_MASK = 0x10003C70,
		VIF1_CODE = 0x10003C80,
		VIF1_R0 = 0x10003D00,
		VIF1_R1 = 0x10003D10,
		VIF1_R2 = 0x10003D20,
		VIF1_R3 = 0x10003D30,
		REGS1_END = 0x10003E00,

		VIF0_FIFO_START = 0x10004000,
		VIF0_FIFO_END = 0x10004FFF,
		VIF1_FIFO_START = 0x10005000,
		VIF1_FIFO_END = 0x10005FFF,
	};

	CVif(unsigned int, CVpu&, CINTC&, uint8*, uint8*);
	virtual ~CVif() = default;

	virtual void Reset();
	uint32 GetRegister(uint32);
	void SetRegister(uint32, uint32);
	void CountTicks(uint32);
	virtual void SaveState(Framework::CZipArchiveWriter&);
	virtual void LoadState(Framework::CZipArchiveReader&);

	virtual uint32 GetTOP() const;
	virtual uint32 GetITOP() const;

	virtual uint32 ReceiveDMA(uint32, uint32, uint32, bool);

	bool IsWaitingForProgramEnd() const;

	void SetRole(ROLE, CVu1Thread*);
	//Front: pushes the qwords consumed since the last call to the VU1 thread
	void ForwardConsumedData();
	//Back: decodes data forwarded by the front, returns false if it is stalled (GIF busy)
	void BeginForwardedData(uint8*, uint32, bool);
	bool ProcessForwardedData();
	//Copies the decoder state (used when switching VU1 threading on or off)
	virtual void CopyStateFrom(const CVif&);
	void CopyRowFrom(const CVif&);

protected:
	enum
	{
		CODE_CMD_NOP = 0x00,
		CODE_CMD_STCYCL = 0x01,
		CODE_CMD_OFFSET = 0x02, //VIF1 only
		CODE_CMD_BASE = 0x03,   //VIF1 only
		CODE_CMD_ITOP = 0x04,
		CODE_CMD_STMOD = 0x05,
		CODE_CMD_MSKPATH3 = 0x06, //VIF1 only
		CODE_CMD_MARK = 0x07,
		CODE_CMD_FLUSHE = 0x10,
		CODE_CMD_FLUSH = 0x11,  //VIF1 only
		CODE_CMD_FLUSHA = 0x13, //VIF1 only
		CODE_CMD_MSCAL = 0x14,
		CODE_CMD_MSCALF = 0x15,
		CODE_CMD_MSCNT = 0x17,
		CODE_CMD_STMASK = 0x20,
		CODE_CMD_STROW = 0x30,
		CODE_CMD_STCOL = 0x31,
		CODE_CMD_MPG = 0x4A,
		CODE_CMD_DIRECT = 0x50,   //VIF1 only
		CODE_CMD_DIRECTHL = 0x51, //VIF1 only
	};

	enum
	{
		FBRST_RST = 0x01,
		FBRST_FBK = 0x02,
		FBRST_STP = 0x04,
		FBRST_STC = 0x08
	};

	enum
	{
		STAT_VPS_IDLE = 0x00,
		STAT_VPS_WAITING = 0x01,
		STAT_VPS_DECODING = 0x02,
		STAT_VPS_TRANSFER = 0x03,
		STAT_VPS_MASK = 0x03,

		STAT_FDR = 0x00800000,
	};

	enum
	{
		FIFO_SIZE = 0x100
	};

	class CFifoStream
	{
	public:
		CFifoStream(uint8*, uint8*);
		virtual ~CFifoStream() = default;

		void Reset();

		inline uint32 GetAvailableReadBytes() const
		{
			return GetRemainingDmaTransferSize() + (BUFFERSIZE - m_bufferPosition);
		}

		inline uint32 GetRemainingDmaTransferSize() const
		{
			return m_endAddress - m_nextAddress;
		}

		//The GetAvailableReadBytes() bytes left are contiguous at GetDirectPointer(): no tag to skip, and
		//the buffer (if partly read) was loaded from the current source
		inline bool CanReadDirect() const
		{
			return !m_tagIncluded && ((m_bufferPosition == BUFFERSIZE) || ((m_nextAddress - m_startAddress) >= BUFFERSIZE));
		}

		inline void Read(void* buffer, uint32 size)
		{
			assert(m_source != nullptr);
			assert(buffer != nullptr);
			uint8* readBuffer = reinterpret_cast<uint8*>(buffer);
			while(size != 0)
			{
				SyncBuffer();
				uint32 read = std::min<uint32>(size, BUFFERSIZE - m_bufferPosition);
				memcpy(readBuffer, reinterpret_cast<uint8*>(&m_buffer) + m_bufferPosition, read);
				readBuffer += read;
				m_bufferPosition += read;
				size -= read;
			}
		}

		template <size_t ValueSize>
		inline FRAMEWORK_SAFE_BUFFERS void ReadValue(void* buffer)
		{
			//Hyper-specialized version of Read that allows the compiler to better
			//optimize the code. Will only work with reads of at most 16 bytes.
			static_assert(ValueSize <= 0x10);
			assert(m_source != nullptr);
			assert(buffer != nullptr);
			uint32 availableBufferSize = BUFFERSIZE - m_bufferPosition;
			uint8* readBuffer = reinterpret_cast<uint8*>(buffer);
			if(ValueSize <= availableBufferSize)
			{
				memcpy(readBuffer, reinterpret_cast<uint8*>(&m_buffer) + m_bufferPosition, ValueSize);
				m_bufferPosition += ValueSize;
			}
			else if(!m_tagIncluded || (ValueSize <= 8))
			{
				uint128 qw[2];
				qw[0] = m_buffer;
				uint32 startPosition = m_bufferPosition;
				bool hasTag = m_tagIncluded;
				m_bufferPosition = BUFFERSIZE;
				SyncBuffer();
				qw[1] = m_buffer;
				if(hasTag)
				{
					qw[1].nD0 = qw[1].nD1;
				}
				m_bufferPosition += (ValueSize - availableBufferSize);
				memcpy(readBuffer, reinterpret_cast<uint8*>(&qw) + startPosition, ValueSize);
			}
			else
			{
				Read(buffer, ValueSize);
			}
		}

		//Same as Read without copying (front decoder)
		inline void Skip(uint32 size)
		{
			uint32 bufferAvailable = BUFFERSIZE - m_bufferPosition;
			if(size <= bufferAvailable)
			{
				m_bufferPosition += size;
				return;
			}
			size -= bufferAvailable;
			m_bufferPosition = BUFFERSIZE;
			if(!m_tagIncluded)
			{
				//Only the last qword touched needs to be loaded in the buffer
				uint32 wholeQwords = (size - 1) / BUFFERSIZE;
				m_nextAddress += wholeQwords * BUFFERSIZE;
				size -= wholeQwords * BUFFERSIZE;
			}
			while(size != 0)
			{
				SyncBuffer();
				uint32 amount = std::min<uint32>(size, BUFFERSIZE - m_bufferPosition);
				m_bufferPosition += amount;
				size -= amount;
			}
		}

		void Flush();
		inline void Align32();
		void SetDmaParams(uint32, uint32, bool);
		void SetFifoParams(uint8*, uint32);
		void SetRawParams(uint8*, uint32, bool);
		//Returns the qwords loaded from the source since the previous call
		bool TakeLoaded(const uint8*&, uint32&, bool&);

		uint8* GetDirectPointer() const;
		void Advance(uint32);

		uint128 GetBuffer() const;
		void SetBuffer(uint128);

		uint32 GetBufferPosition() const;
		void SetBufferPosition(uint32);

	private:
		inline void SyncBuffer()
		{
			assert(m_bufferPosition <= BUFFERSIZE);
			if(m_bufferPosition >= BUFFERSIZE)
			{
				assert(m_nextAddress < m_endAddress);
				m_buffer = *reinterpret_cast<uint128*>(&m_source[m_nextAddress]);
				m_nextAddress += 0x10;
				m_bufferPosition = 0;
				if(m_tagIncluded)
				{
					//Skip next 8 bytes
					m_tagIncluded = false;
					m_bufferPosition += 8;
				}
			}
		}

		enum
		{
			BUFFERSIZE = 0x10
		};

		uint8* m_ram = nullptr;
		uint8* m_spr = nullptr;

		alignas(16) uint128 m_buffer = {};
		uint32 m_bufferPosition = BUFFERSIZE;
		uint32 m_startAddress = 0;
		uint32 m_nextAddress = 0;
		uint32 m_endAddress = 0;
		bool m_tagIncluded = false;
		uint8* m_source = nullptr;
		uint32 m_forwardAddress = 0;
		bool m_initialTagIncluded = false;
	};

	typedef CFifoStream StreamType;

	struct STAT : public convertible<uint32>
	{
		unsigned int nVPS : 2;
		unsigned int nVEW : 1;
		unsigned int nVGW : 1; //VIF1 only
		unsigned int nReserved0 : 2;
		unsigned int nMRK : 1;
		unsigned int nDBF : 1;
		unsigned int nVSS : 1;
		unsigned int nVFS : 1;
		unsigned int nVIS : 1;
		unsigned int nINT : 1;
		unsigned int nER0 : 1;
		unsigned int nER1 : 1;
		unsigned int nReserved2 : 9;
		unsigned int nFDR : 1; //VIF1 only
		unsigned int nFQC : 4;
		unsigned int nReserved3 : 4;
	};
	static_assert(sizeof(STAT) == sizeof(uint32), "Size of STAT struct must be 4 bytes.");

	struct ERR : public convertible<uint32>
	{
		unsigned int nMII : 1;
		unsigned int nME0 : 1;
		unsigned int nME1 : 1;
		unsigned int nReserved : 29;
	};
	static_assert(sizeof(ERR) == sizeof(uint32), "Size of ERR struct must be 4 bytes.");

	struct CYCLE : public convertible<uint32>
	{
		unsigned int nCL : 8;
		unsigned int nWL : 8;
		unsigned int reserved : 16;
	};
	static_assert(sizeof(CYCLE) == sizeof(uint32), "Size of CYCLE struct must be 4 bytes.");

	struct CODE : public convertible<uint32>
	{
		unsigned int nIMM : 16;
		unsigned int nNUM : 8;
		unsigned int nCMD : 7;
		unsigned int nI : 1;
	};
	static_assert(sizeof(CODE) == sizeof(uint32), "Size of CODE struct must be 4 bytes.");

	enum ADDMODE
	{
		MODE_NORMAL = 0,
		MODE_OFFSET = 1,
		MODE_DIFFERENCE = 2
	};

	enum MASKOP
	{
		MASK_DATA = 0,
		MASK_ROW = 1,
		MASK_COL = 2,
		MASK_MASK = 3
	};

	void ProcessFifoWrite(uint32, uint32);

	void ProcessPacket(StreamType&);
	virtual void ExecuteCommand(StreamType&, CODE);
	virtual void Cmd_UNPACK(StreamType&, CODE, uint32);

	void Cmd_MPG(StreamType&, CODE);
	void Cmd_STROW(StreamType&, CODE);
	void Cmd_STCOL(StreamType&, CODE);
	void Cmd_STMASK(StreamType&, CODE);

	inline uint32 GetColMaskOp(unsigned int) const;

	inline bool IsVuReadyForVif() const
	{
		//The front never waits for VU1: the back executes everything in order.
		//The back never sees VU1 running (programs run to completion), but a program stopped by
		//a T/D bit must not stall it forever.
		switch(m_role)
		{
		case ROLE_FRONT:
			return true;
		case ROLE_BACK:
			return !m_vpu.IsVuRunning();
		default:
			return m_vpu.IsVuReady();
		}
	}

	void ExecuteMicroProgram(uint32);
	void SkipUnpack(StreamType&, CODE);

	inline bool Unpack_S32(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < 4) return false;

		uint32 value;
		stream.ReadValue<4>(&value);

		for(unsigned int i = 0; i < 4; i++)
		{
			result.nV[i] = value;
		}

		return true;
	}

	template <bool zeroExtend>
	inline bool Unpack_S16(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < 2) return false;

		uint16 value;
		stream.ReadValue<2>(&value);

		uint32 temp = value;
		if(!zeroExtend)
		{
			temp = static_cast<int16>(temp);
		}

		for(unsigned int i = 0; i < 4; i++)
		{
			result.nV[i] = temp;
		}

		return true;
	}

	template <bool zeroExtend>
	inline bool Unpack_S8(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < 1) return false;

		uint8 value;
		stream.ReadValue<1>(&value);

		uint32 temp = value;
		if(!zeroExtend)
		{
			temp = static_cast<int8>(temp);
		}

		for(unsigned int i = 0; i < 4; i++)
		{
			result.nV[i] = temp;
		}

		return true;
	}

	inline bool Unpack_V45(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < 2) return false;

		uint16 value = 0;
		stream.ReadValue<2>(&value);

		result.nV0 = ((value >> 0) & 0x1F) << 3;
		result.nV1 = ((value >> 5) & 0x1F) << 3;
		result.nV2 = ((value >> 10) & 0x1F) << 3;
		result.nV3 = ((value >> 15) & 0x01) << 7;

		return true;
	}

	template <unsigned int fields>
	inline bool Unpack_V32(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < (fields * 4)) return false;
		stream.ReadValue<fields * 4>(&result);
		return true;
	}

	template <unsigned int fields, bool zeroExtend>
	inline bool Unpack_V16(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < (fields * 2)) return false;

		uint16 values[fields];
		stream.ReadValue<fields * 2>(values);

		for(unsigned int i = 0; i < fields; i++)
		{
			uint32 temp = values[i];
			if(!zeroExtend)
			{
				temp = static_cast<int16>(temp);
			}

			result.nV[i] = temp;
		}

		return true;
	}

	template <unsigned int fields, bool zeroExtend>
	inline bool Unpack_V8(StreamType& stream, uint128& result)
	{
		if(stream.GetAvailableReadBytes() < (fields)) return false;

		uint8 values[fields];
		stream.ReadValue<fields>(values);

		for(unsigned int i = 0; i < fields; i++)
		{
			uint32 temp = values[i];
			if(!zeroExtend)
			{
				temp = static_cast<int8>(temp);
			}

			result.nV[i] = temp;
		}

		return true;
	}

	template <uint8 dataType, bool usn>
	bool Unpack_ReadValue(StreamType& stream, uint128& writeValue)
	{
		bool success = false;
		switch(dataType)
		{
		case 0x00:
			//S-32
			success = Unpack_S32(stream, writeValue);
			break;
		case 0x01:
			//S-16
			success = Unpack_S16<usn>(stream, writeValue);
			break;
		case 0x02:
			//S-8
			success = Unpack_S8<usn>(stream, writeValue);
			break;
		case 0x04:
			//V2-32
			success = Unpack_V32<2>(stream, writeValue);
			break;
		case 0x05:
			//V2-16
			success = Unpack_V16<2, usn>(stream, writeValue);
			break;
		case 0x06:
			//V2-8
			success = Unpack_V8<2, usn>(stream, writeValue);
			break;
		case 0x08:
			//V3-32
			success = Unpack_V32<3>(stream, writeValue);
			break;
		case 0x09:
			//V3-16
			success = Unpack_V16<3, usn>(stream, writeValue);
			break;
		case 0x0A:
			//V3-8
			success = Unpack_V8<3, usn>(stream, writeValue);
			break;
		case 0x0C:
			//V4-32
			success = Unpack_V32<4>(stream, writeValue);
			break;
		case 0x0D:
			//V4-16
			success = Unpack_V16<4, usn>(stream, writeValue);
			break;
		case 0x0E:
			//V4-8
			success = Unpack_V8<4, usn>(stream, writeValue);
			break;
		case 0x0F:
			//V4-5
			success = Unpack_V45(stream, writeValue);
			break;
		default:
			assert(0);
			break;
		}
		return success;
	}

#ifdef __wasm_simd128__
	static constexpr uint32 UnpackElementSize(uint8 dataType)
	{
		//S-32, S-16, S-8, -, V2-32, V2-16, V2-8, -, V3-32, V3-16, V3-8, -, V4-32, V4-16, V4-8, V4-5 (0: invalid)
		constexpr uint8 sizes[16] = {4, 2, 1, 0, 8, 4, 2, 0, 12, 6, 3, 0, 16, 8, 4, 2};
		return sizes[dataType & 0x0F];
	}

	//One unpacked vector, same values as Unpack_ReadValue (missing fields are 0)
	template <uint8 dataType, bool usn>
	static inline v128_t Unpack_LoadSimd(const uint8* src)
	{
		auto extend16 = [](v128_t v) { return usn ? wasm_u32x4_extend_low_u16x8(v) : wasm_i32x4_extend_low_i16x8(v); };
		auto extend8 = [&](v128_t v) { return extend16(usn ? wasm_u16x8_extend_low_u8x16(v) : wasm_i16x8_extend_low_i8x16(v)); };
		const v128_t zero = wasm_i32x4_const_splat(0);
		switch(dataType)
		{
		case 0x00:
			return wasm_v128_load32_splat(src);
		case 0x01:
		{
			uint16 value;
			memcpy(&value, src, 2);
			return wasm_i32x4_splat(usn ? static_cast<int32>(value) : static_cast<int32>(static_cast<int16>(value)));
		}
		case 0x02:
			return wasm_i32x4_splat(usn ? static_cast<int32>(src[0]) : static_cast<int32>(static_cast<int8>(src[0])));
		case 0x04:
			return wasm_v128_load64_zero(src);
		case 0x05:
			return extend16(wasm_v128_load32_zero(src));
		case 0x06:
			return extend8(wasm_v128_load16_lane(src, zero, 0));
		case 0x08:
			return wasm_v128_load32_lane(src + 8, wasm_v128_load64_zero(src), 2);
		case 0x09:
			return extend16(wasm_v128_load16_lane(src + 4, wasm_v128_load32_zero(src), 2));
		case 0x0A:
			return extend8(wasm_v128_load8_lane(src + 2, wasm_v128_load16_lane(src, zero, 0), 2));
		case 0x0C:
			return wasm_v128_load(src);
		case 0x0D:
			return extend16(wasm_v128_load64_zero(src));
		case 0x0E:
			return extend8(wasm_v128_load32_zero(src));
		case 0x0F:
		default:
		{
			uint16 value;
			memcpy(&value, src, 2);
			return wasm_u32x4_make(((value >> 0) & 0x1F) << 3, ((value >> 5) & 0x1F) << 3, ((value >> 10) & 0x1F) << 3, ((value >> 15) & 0x01) << 7);
		}
		}
	}

	//count consecutive vectors read and written (no skipping or filling between them), same results
	//as count iterations of the loop in Unpack. writeTick: m_writeTick of the first one.
	template <uint8 dataType, bool useMask, uint8 mode, bool usn>
	void UnpackRunSimd(const uint8* src, uint32 count, uint32 dstAddr, uint8* vuMem, uint32 vuMemSize, uint32 writeTick)
	{
		constexpr uint32 elementSize = UnpackElementSize(dataType);
		v128_t row = wasm_v128_load(m_R);
		v128_t dataLanes[4], rowLanes[4], colLanes[4], keepLanes[4];
		if(useMask)
		{
			for(uint32 col = 0; col < 4; col++)
			{
				uint32 colMask = GetColMaskOp(col);
				auto lanes = [&](uint32 op) {
					auto lane = [&](uint32 i) { return (((colMask >> (i * 2)) & 3) == op) ? -1 : 0; };
					return wasm_i32x4_make(lane(0), lane(1), lane(2), lane(3));
				};
				dataLanes[col] = lanes(MASK_DATA);
				rowLanes[col] = lanes(MASK_ROW);
				colLanes[col] = wasm_v128_and(lanes(MASK_COL), wasm_i32x4_splat(m_C[col]));
				keepLanes[col] = lanes(MASK_MASK);
			}
		}
		for(uint32 i = 0; i < count; i++)
		{
			v128_t value = Unpack_LoadSimd<dataType, usn>(src);
			src += elementSize;
			uint8* dst = vuMem + dstAddr;
			dstAddr = (dstAddr + 0x10) & (vuMemSize - 1);
			if(!useMask)
			{
				if(mode == MODE_OFFSET)
				{
					value = wasm_i32x4_add(value, row);
				}
				else if(mode == MODE_DIFFERENCE)
				{
					row = wasm_i32x4_add(value, row);
					value = row;
				}
				wasm_v128_store(dst, value);
			}
			else
			{
				uint32 col = std::min<uint32>(writeTick + i, 3);
				if(mode == MODE_OFFSET)
				{
					value = wasm_i32x4_add(value, row);
				}
				else if(mode == MODE_DIFFERENCE)
				{
					value = wasm_i32x4_add(value, row);
					row = wasm_v128_bitselect(value, row, dataLanes[col]);
				}
				v128_t result = wasm_v128_or(wasm_v128_or(wasm_v128_and(value, dataLanes[col]), wasm_v128_and(row, rowLanes[col])), colLanes[col]);
				result = wasm_v128_bitselect(wasm_v128_load(dst), result, keepLanes[col]);
				wasm_v128_store(dst, result);
			}
		}
		if(mode == MODE_DIFFERENCE)
		{
			wasm_v128_store(m_R, row);
		}
	}
#endif

	template <uint8 dataType, bool clGreaterEqualWl, bool useMask, uint8 mode, bool usn>
	void Unpack(StreamType& stream, CODE nCommand, uint32 nDstAddr)
	{
		assert((nCommand.nCMD & 0x60) == 0x60);

		const auto vuMem = m_vpu.GetVuMemory();
		const auto vuMemSize = m_vpu.GetVuMemorySize();
		uint32 cl = m_CYCLE.nCL;
		uint32 wl = m_CYCLE.nWL;
		if(wl == 0)
		{
			wl = UINT_MAX;
			cl = 0;
		}

		if(m_NUM == nCommand.nNUM)
		{
			m_readTick = 0;
			m_writeTick = 0;
		}

		uint32 currentNum = (m_NUM == 0) ? 256 : m_NUM;
		uint32 codeNum = (m_CODE.nNUM == 0) ? 256 : m_CODE.nNUM;
		uint32 transfered = codeNum - currentNum;

		if(cl > wl)
		{
			nDstAddr += cl * (transfered / wl) + (transfered % wl);
		}
		else
		{
			nDstAddr += transfered;
		}

		nDstAddr *= 0x10;
		assert(nDstAddr < vuMemSize);
		nDstAddr &= (vuMemSize - 1);

		while(currentNum != 0)
		{
#ifdef __wasm_simd128__
			//Whole run of vectors that are read and written, straight from the source
			if constexpr(UnpackElementSize(dataType) != 0)
			{
				constexpr uint32 elementSize = UnpackElementSize(dataType);
				bool readPhase = clGreaterEqualWl ? (m_readTick < wl) : (m_writeTick < cl);
				if(readPhase && stream.CanReadDirect())
				{
					uint32 count = clGreaterEqualWl ? (wl - m_readTick) : (cl - m_writeTick);
					count = std::min<uint32>(count, currentNum);
					count = std::min<uint32>(count, stream.GetAvailableReadBytes() / elementSize);
					if(count != 0)
					{
						UnpackRunSimd<dataType, useMask, mode, usn>(stream.GetDirectPointer(), count, nDstAddr, vuMem, vuMemSize, m_writeTick);
						stream.Skip(count * elementSize);
						currentNum -= count;
						m_writeTick = std::min<uint32>(m_writeTick + count, wl);
						m_readTick = std::min<uint32>(m_readTick + count, cl);
						if(clGreaterEqualWl ? (m_readTick == cl) : (m_writeTick == wl))
						{
							m_writeTick = 0;
							m_readTick = 0;
						}
						nDstAddr = (nDstAddr + (count * 0x10)) & (vuMemSize - 1);
						continue;
					}
				}
			}
#endif
			bool mustWrite = false;
			uint128 writeValue;
			memset(&writeValue, 0, sizeof(writeValue));

			if(clGreaterEqualWl)
			{
				if(m_readTick < wl)
				{
					bool success = Unpack_ReadValue<dataType, usn>(stream, writeValue);
					if(!success) break;
					mustWrite = true;
				}
			}
			else
			{
				if(m_writeTick < cl)
				{
					bool success = Unpack_ReadValue<dataType, usn>(stream, writeValue);
					if(!success) break;
				}

				mustWrite = true;
			}

			if(mustWrite)
			{
				auto dst = reinterpret_cast<uint128*>(vuMem + nDstAddr);
				uint32 col = (m_writeTick > 3) ? 3 : m_writeTick;
				uint32 colMask = useMask ? GetColMaskOp(col) : 0;

				if((colMask == 0) && (mode == 0))
				{
					(*dst) = writeValue;
				}
#ifdef FRAMEWORK_SIMD_USE_SSE
				else if((colMask == 0) && (mode == MODE_OFFSET))
				{
					*reinterpret_cast<__m128i*>(dst) = _mm_add_epi32(*reinterpret_cast<__m128i*>(&writeValue), *reinterpret_cast<__m128i*>(&m_R));
				}
#endif
				else
				{
					for(unsigned int i = 0; i < 4; i++)
					{
						uint32 maskOp = useMask ? (colMask >> (i * 2)) & 0x3 : MASK_DATA;

						if(maskOp == MASK_DATA)
						{
							if(mode == MODE_OFFSET)
							{
								writeValue.nV[i] += m_R[i];
							}
							else if(mode == MODE_DIFFERENCE)
							{
								writeValue.nV[i] += m_R[i];
								m_R[i] = writeValue.nV[i];
							}

							dst->nV[i] = writeValue.nV[i];
						}
						else if(maskOp == MASK_ROW)
						{
							dst->nV[i] = m_R[i];
						}
						else if(maskOp == MASK_COL)
						{
							dst->nV[i] = m_C[col];
						}
						else if(maskOp == MASK_MASK)
						{
							//Don't write anything
						}
						else
						{
							assert(0);
						}
					}
				}

				currentNum--;
			}

			if(clGreaterEqualWl)
			{
				m_writeTick = std::min<uint32>(m_writeTick + 1, wl);
				m_readTick = std::min<uint32>(m_readTick + 1, cl);

				if(m_readTick == cl)
				{
					m_writeTick = 0;
					m_readTick = 0;
				}
			}
			else
			{
				m_writeTick = std::min<uint32>(m_writeTick + 1, wl);
				m_readTick = std::min<uint32>(m_readTick + 1, cl);

				if(m_writeTick == wl)
				{
					m_writeTick = 0;
					m_readTick = 0;
				}
			}

			nDstAddr += 0x10;
			nDstAddr &= (vuMemSize - 1);
		}

		if(currentNum != 0)
		{
			m_STAT.nVPS = 1;
		}
		else
		{
			stream.Align32();
			m_STAT.nVPS = 0;
		}

		m_NUM = static_cast<uint8>(currentNum);
	}

	typedef void (CVif::*Unpacker)(StreamType&, CODE, uint32);

	enum
	{
		MAX_UNPACKERS = 0x200
	};

	virtual void PrepareMicroProgram();
	void StartMicroProgram(uint32);
	void StartDelayedMicroProgram(uint32);
	bool ResumeDelayedMicroProgram();

	void DisassembleCommand(CODE);
	void DisassembleGet(uint32);
	void DisassembleSet(uint32, uint32);

	unsigned int m_number = 0;
	ROLE m_role = ROLE_NORMAL;
	CVu1Thread* m_vu1Thread = nullptr;
	CVpu& m_vpu;
	CINTC& m_intc;
	uint8* m_ram = nullptr;
	uint8* m_spr = nullptr;
	CFifoStream m_stream;
	Unpacker m_unpacker[MAX_UNPACKERS];

	uint8 m_fifoBuffer[FIFO_SIZE];
	uint32 m_fifoIndex = 0;

	STAT m_STAT;
	ERR m_ERR;
	CYCLE m_CYCLE;
	CODE m_CODE;
	uint8 m_NUM;
	uint32 m_MODE;
	alignas(16) uint32 m_R[4];
	uint32 m_C[4];
	uint32 m_MASK;
	uint32 m_MARK;
	uint32 m_ITOP;
	uint32 m_ITOPS;
	uint32 m_readTick;
	uint32 m_writeTick;
	uint32 m_pendingMicroProgram;
	uint32 m_incomingFifoDelay;
	int32 m_interruptDelayTicks;

	CProfiler::ZoneHandle m_vifProfilerZone = 0;
};
