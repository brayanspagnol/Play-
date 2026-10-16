cmake_minimum_required(VERSION 3.18)

set(CMAKE_MODULE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/../../deps/Dependencies/cmake-modules
	${CMAKE_MODULE_PATH}
)
include(Header)

project(Play)

if(NOT TARGET PlayCore)
	add_subdirectory(
		${CMAKE_CURRENT_SOURCE_DIR}/../
		${CMAKE_CURRENT_BINARY_DIR}/Source
	)
endif()
list(APPEND UI_JS_PROJECT_LIBS PlayCore)

if(NOT TARGET ui_shared)
	add_subdirectory(
		${CMAKE_CURRENT_SOURCE_DIR}/../ui_shared
		${CMAKE_CURRENT_BINARY_DIR}/ui_shared
	)
endif()
list(APPEND UI_JS_PROJECT_LIBS ui_shared)

if(NOT TARGET gsh_opengl)
	add_subdirectory(
		${CMAKE_CURRENT_SOURCE_DIR}/../gs/GSH_OpenGL
		${CMAKE_CURRENT_BINARY_DIR}/gs/GSH_OpenGL
	)
endif()
list(APPEND UI_JS_PROJECT_LIBS gsh_opengl)

add_executable(Play
	GSH_OpenGLJs.cpp
	GSH_OpenGLJs.h
	GSH_WebGPU.cpp
	GSH_WebGPU.h
	InputProviderEmscripten.cpp
	InputProviderEmscripten.h
	Main.cpp
	Ps2VmJs.cpp
	Ps2VmJs.h
	SH_WebAudio.cpp
	SH_WebAudio.h
)
target_link_libraries(Play PUBLIC ${UI_JS_PROJECT_LIBS})

target_link_options(Play PRIVATE "--bind")
target_link_options(Play PRIVATE "-sMODULARIZE=1")
target_link_options(Play PRIVATE "-sEXPORT_ES6=1")
target_link_options(Play PRIVATE "-sUSE_ES6_IMPORT_META=0")
target_link_options(Play PRIVATE "-sEXPORTED_FUNCTIONS=['_main', '_initVm', '_EmptyBlockHandler', '_MemoryUtils_GetByteProxy', '_MemoryUtils_GetHalfProxy', '_MemoryUtils_GetWordProxy', '_MemoryUtils_GetDoubleProxy', '_MemoryUtils_SetByteProxy', '_MemoryUtils_SetHalfProxy', '_MemoryUtils_SetWordProxy', '_MemoryUtils_SetDoubleProxy', '_LWL_Proxy', '_LWR_Proxy', '_LDL_Proxy', '_LDR_Proxy', '_SWL_Proxy', '_SWR_Proxy', '_SDL_Proxy', '_SDR_Proxy']")
target_link_options(Play PRIVATE "-sEXPORTED_RUNTIME_METHODS=['ccall', 'FS']")
target_link_options(Play PRIVATE "-sFORCE_FILESYSTEM")
target_link_options(Play PRIVATE "-sALLOW_TABLE_GROWTH")
target_link_options(Play PRIVATE "-sALLOW_MEMORY_GROWTH")
#target_link_options(Play PRIVATE "-sASSERTIONS=2")
target_link_options(Play PRIVATE "-sEXPORT_NAME=Play")
target_link_options(Play PRIVATE "-sPTHREAD_POOL_SIZE=6")
target_link_options(Play PRIVATE "-sWASM_BIGINT")
target_link_options(Play PRIVATE "-sOFFSCREEN_FRAMEBUFFER=1")
target_link_options(Play PRIVATE "-sENVIRONMENT=web,worker")
target_link_options(Play PRIVATE "-sUSE_WEBGL2=1")
#Per-game settings (idle loops, FP modes, patches) read by CGameConfig from the app resources path
target_link_options(Play PRIVATE "--embed-file=${CMAKE_CURRENT_SOURCE_DIR}/../../GameConfig.xml@/usr/local/share/GameConfig.xml")
set_property(TARGET Play APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../GameConfig.xml)
#Audio output: AudioWorklet + time-stretching (see SH_WebAudio.h)
target_link_options(Play PRIVATE "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/WebAudio.js")
set_property(TARGET Play APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/WebAudio.js)
#WebGPU GS renderer (GPU side, WGSL shaders), see GSH_WebGPU.h
target_link_options(Play PRIVATE "--pre-js=${CMAKE_CURRENT_SOURCE_DIR}/GSH_WebGPU.js")
set_property(TARGET Play APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/GSH_WebGPU.js)
