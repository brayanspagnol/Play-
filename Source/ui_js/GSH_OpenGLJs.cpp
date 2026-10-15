#include "GSH_OpenGLJs.h"
#include <cassert>
#include <emscripten/em_js.h>
#include <emscripten/html5_webgl.h>

// clang-format off
EM_JS(void, GsJs_CreateCanvas, (int width, int height), {
	// Registered under a selector so emscripten_webgl_create_context can find it on this thread.
	specialHTMLTargets["#gsCanvas"] = new OffscreenCanvas(width, height);
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

void CGSH_OpenGLJs::PresentBackbuffer()
{
	GsJs_Present();
}
