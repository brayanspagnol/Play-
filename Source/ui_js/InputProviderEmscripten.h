#pragma once

#include "input/InputProvider.h"
#include "ControllerInfo.h"
#include <emscripten/html5.h>

// One provider for keyboard + browser gamepads (Gamepad API, "standard" layout).
// Each PS2 control has one binding target (keyId = PS2 button index); the provider
// merges both sources into it, so keyboard and gamepad work at the same time:
// - buttons: pressed if pressed on the keyboard OR on any connected gamepad;
// - analog axes: gamepad stick value when outside the deadzone, else keyboard keys.
// PollGamepads() must run on the browser main thread (Main.cpp calls it every
// animation frame); connecting/disconnecting a gamepad is picked up on the next poll.
class CInputProviderEmscripten : public CInputProvider
{
public:
	uint32 GetId() const override;
	std::string GetTargetDescription(const BINDINGTARGET&) const override;

	static BINDINGTARGET MakeBindingTarget(PS2::CControllerInfo::BUTTON);

	void OnKeyDown(const EM_UTF8*);
	void OnKeyUp(const EM_UTF8*);
	void PollGamepads();

private:
	void OnKey(const EM_UTF8*, bool);
	void Publish(PS2::CControllerInfo::BUTTON);

	enum
	{
		AXIS_NEUTRAL = 0x7F,
	};

	// Keyboard: buttons use keyNeg only; axes use keyNeg (toward 0x00) / keyPos (toward 0xFF).
	bool m_keyNeg[PS2::CControllerInfo::MAX_BUTTONS] = {};
	bool m_keyPos[PS2::CControllerInfo::MAX_BUTTONS] = {};
	uint32 m_padValue[PS2::CControllerInfo::MAX_BUTTONS] = {AXIS_NEUTRAL, AXIS_NEUTRAL, AXIS_NEUTRAL, AXIS_NEUTRAL};
	uint32 m_lastValue[PS2::CControllerInfo::MAX_BUTTONS] = {AXIS_NEUTRAL, AXIS_NEUTRAL, AXIS_NEUTRAL, AXIS_NEUTRAL};
};
