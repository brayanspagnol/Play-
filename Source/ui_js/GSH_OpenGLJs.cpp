#include "GSH_OpenGLJs.h"
#include <algorithm>
#include <cassert>
#include <emscripten/em_js.h>
#include <emscripten/html5_webgl.h>

// clang-format off
EM_JS(void, GsJs_CreateCanvas, (int width, int height), {
	// Registered under a selector so emscripten_webgl_create_context can find it on this thread.
	specialHTMLTargets["#gsCanvas"] = new OffscreenCanvas(width, height);
});

EM_JS(void, GsJs_ResizeCanvas, (int width, int height), {
	GLctx.canvas.width = width;
	GLctx.canvas.height = height;
});

EM_JS(void, GsJs_Present, (), {
	// Only complete frames ever reach the page (no cleared or half-drawn buffer can be shown).
	var bitmap = GLctx.canvas.transferToImageBitmap();
	postMessage({ cmd: 'callHandler', handler: 'onGsFrame', args: [bitmap] }, [bitmap]);
});
// clang-format on

CGSH_OpenGLJs::CGSH_OpenGLJs(int width, int height)
    : m_width(width)
    , m_height(height)
{
}

CGSH_OpenGL::FactoryFunction CGSH_OpenGLJs::GetFactoryFunction(int width, int height)
{
	return [width, height]() { return new CGSH_OpenGLJs(width, height); };
}

void CGSH_OpenGLJs::InitializeImpl()
{
	//Deferred to InitializeContext (see header).
}

void CGSH_OpenGLJs::InitializeContext()
{
	GsJs_CreateCanvas(m_width, m_height);
	EmscriptenWebGLContextAttributes attr;
	emscripten_webgl_init_context_attributes(&attr);
	attr.majorVersion = 2;
	attr.minorVersion = 0;
	attr.alpha = false;
	attr.proxyContextToMainThread = EMSCRIPTEN_WEBGL_CONTEXT_PROXY_DISALLOW;
	m_context = emscripten_webgl_create_context("#gsCanvas", &attr);
	assert(m_context > 0);
	emscripten_webgl_make_context_current(m_context);
	CGSH_OpenGL::InitializeImpl();
}

void CGSH_OpenGLJs::ReleaseImpl()
{
	CGSH_OpenGL::ReleaseImpl();
}

void CGSH_OpenGLJs::SetOutput(int width, int height, PRESENTATION_MODE mode)
{
	SendGSCall([this, width, height, mode]() {
		m_width = width;
		m_height = height;
		if(m_context > 0) GsJs_ResizeCanvas(width, height);
		PRESENTATION_PARAMS params;
		params.mode = mode;
		params.windowWidth = width;
		params.windowHeight = height;
		SetPresentationParams(params);
		//Reloads the resolution factor/texture filtering preferences and drops the framebuffers made with the old ones
		NotifyPreferencesChangedImpl();
	});
}

void CGSH_OpenGLJs::PresentBackbuffer()
{
	GsJs_Present();
}

void CGSH_OpenGLJs::SetFrameSkip(int count)
{
	m_frameSkip = std::max(count, 0);
}

uint32 CGSH_OpenGLJs::GetSkippedFrames() const
{
	return m_skippedFrames;
}

//GS thread, at each flip. A game frame ends when the game shows another buffer (double buffering, the usual
//case); for games that keep one display buffer, every flip ends one. The drawing state changes only at those
//boundaries, so a shown frame is never partly drawn: a skipped frame's buffer (stale) is not presented and the
//page keeps the previous picture.
void CGSH_OpenGLJs::FlipImpl(const DISPLAY_INFO& dispInfo)
{
	uint32 displayPtr = dispInfo.layers[0].bufPtr;
	bool swapped = (displayPtr != m_lastDisplayPtr);
	m_lastDisplayPtr = displayPtr;
	m_flipsSinceSwap = swapped ? 0 : std::min<uint32>(m_flipsSinceSwap + 1, 1000);
	bool boundary = swapped || (m_flipsSinceSwap > 4);
	bool drawn = m_drawEnabled;
	if(boundary)
	{
		uint32 skip = m_frameSkip.load(std::memory_order_relaxed);
		if(skip == 0)
			m_skipLeft = 0;
		else if(m_skipLeft != 0)
			m_skipLeft--;
		else if(drawn)
			m_skipLeft = skip;
		m_drawEnabled = (m_skipLeft == 0);
	}
	if(drawn)
	{
		CGSH_OpenGL::FlipImpl(dispInfo);
	}
	else
	{
		m_skippedFrames++;
		CGSHandler::FlipImpl(dispInfo);
	}
}
