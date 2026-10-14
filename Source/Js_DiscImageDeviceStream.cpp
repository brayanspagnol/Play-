#include "Js_DiscImageDeviceStream.h"
#include <stdexcept>
#include <cassert>
#include <emscripten.h>
#include <unistd.h>

static std::string g_discImageUrl;

void Js_SetDiscImageUrl(const std::string& url)
{
	g_discImageUrl = url;
}

// Runs on the calling (emulation) thread: reads [position, position + size) from
// the URL in 1 MB blocks through a small LRU kept in this worker's global scope.
// Returns 0 on failure so the caller can fall back to the main-thread path.
EM_JS(int, Js_ReadDiscUrl, (const char* urlPtr, uint32 positionLow, uint32 positionHigh, void* buffer, uint32 size), {
	const url = UTF8ToString(urlPtr);
	let d = globalThis.__ps2Disc;
	if(!d || d.url !== url) d = globalThis.__ps2Disc = {url : url, blocks : new Map()};
	const BS = 1 << 20, MAX_BLOCKS = 48;
	let pos = (positionLow >>> 0) + (positionHigh >>> 0) * 4294967296;
	let dst = buffer >>> 0;
	let left = size >>> 0;
	while(left > 0)
	{
		const bi = Math.floor(pos / BS);
		let b = d.blocks.get(bi);
		if(b)
		{
			d.blocks.delete(bi);
			d.blocks.set(bi, b);
		}
		else
		{
			const start = bi * BS;
			for(let attempt = 0;; attempt++)
			{
				try
				{
					const x = new XMLHttpRequest();
					x.open("GET", url, false);
					x.responseType = "arraybuffer";
					x.setRequestHeader("Range", "bytes=" + start + "-" + (start + BS - 1));
					x.send();
					if(x.status !== 206) throw new Error("HTTP " + x.status);
					b = new Uint8Array(x.response);
					break;
				}
				catch(e)
				{
					if(attempt >= 2)
					{
						console.error("Disc read failed at " + start + ": " + e);
						return 0;
					}
				}
			}
			d.blocks.set(bi, b);
			if(d.blocks.size > MAX_BLOCKS) d.blocks.delete(d.blocks.keys().next().value);
		}
		const off = pos - bi * BS;
		const n = Math.min(left, b.length - off);
		if(n <= 0) return 0;
		HEAPU8.set(b.subarray(off, off + n), dst);
		dst += n;
		pos += n;
		left -= n;
	}
	return 1;
});

void CJsDiscImageDeviceStream::Seek(int64 position, Framework::STREAM_SEEK_DIRECTION whence)
{
	switch(whence)
	{
	case Framework::STREAM_SEEK_SET:
		m_position = position;
		break;
	case Framework::STREAM_SEEK_CUR:
		m_position += position;
		break;
	case Framework::STREAM_SEEK_END:
	{
		uint32 positionLo = MAIN_THREAD_EM_ASM_INT({return Module.discImageDevice.getFileSize()});
		uint32 positionHi = MAIN_THREAD_EM_ASM_INT({return Module.discImageDevice.getFileSize() / 4294967296});
		m_position = static_cast<uint64>(positionLo) | (static_cast<uint64>(positionHi) << 32);
	}
	break;
	}
}

uint64 CJsDiscImageDeviceStream::Tell()
{
	return m_position;
}

uint64 CJsDiscImageDeviceStream::Read(void* buffer, uint64 size)
{
	if(size == 0) return 0;

	assert(size <= std::numeric_limits<uint32>::max());

	uint32 positionLow = static_cast<uint32>(m_position);
	uint32 positionHigh = static_cast<uint32>(m_position >> 32);

	if(!g_discImageUrl.empty() && Js_ReadDiscUrl(g_discImageUrl.c_str(), positionLow, positionHigh, buffer, static_cast<uint32>(size)))
	{
		m_position += size;
		return size;
	}

	MAIN_THREAD_EM_ASM({
		let posLow = $1 >>> 0;
		let posHigh = $2 >>> 0;
		let position = posLow + (posHigh * 4294967296);
		Module.discImageDevice.read($0, position, $3);
	},
	                   buffer, positionLow, positionHigh, static_cast<uint32>(size));
	while(!MAIN_THREAD_EM_ASM_INT({return Module.discImageDevice.isDone()}))
	{
		usleep(100);
	}
	m_position += size;
	return size;
}

uint64 CJsDiscImageDeviceStream::Write(const void*, uint64)
{
	throw std::runtime_error("Not supported.");
}

bool CJsDiscImageDeviceStream::IsEOF()
{
	throw std::runtime_error("Not supported.");
}

void CJsDiscImageDeviceStream::Flush()
{
}
