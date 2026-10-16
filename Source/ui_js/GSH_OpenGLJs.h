#pragma once

#include <emscripten/threading.h>
#include "gs/GSH_OpenGL/GSH_OpenGL.h"

//GS on its own WebGL2 context: the context lives on an OffscreenCanvas created by the GS thread, so
//GL calls run on that thread instead of being proxied to the browser main thread. Every presented
//frame is handed to the page as an ImageBitmap (Module.onGsFrame(bitmap) on the main thread).
//Creating a WebGL context in a worker needs the main thread (Chromium sets it up there), so it can't
//happen in InitializeImpl, which the main thread waits for: InitializeContext is queued on the GS
//thread afterwards (initVm) and the first frame it presents tells the page the GS is ready.
class CGSH_OpenGLJs : public CGSH_OpenGL
{
public:
	CGSH_OpenGLJs(int width, int height);
	virtual ~CGSH_OpenGLJs() = default;

	static FactoryFunction GetFactoryFunction(int width, int height);

	void InitializeImpl() override;
	void InitializeContext();
	//Output (OffscreenCanvas) size and presentation mode, plus a reload of the GS preferences
	//(resolution factor, bilinear filtering, widescreen); queued on the GS thread.
	void SetOutput(int width, int height, PRESENTATION_MODE);
	void ReleaseImpl() override;
	void PresentBackbuffer() override;

private:
	int m_width = 0;
	int m_height = 0;
	EMSCRIPTEN_WEBGL_CONTEXT_HANDLE m_context = 0;
};
