#include "InputProviderEmscripten.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iterator>

using PS2::CControllerInfo;

constexpr uint32 PROVIDER_ID = 'EmSc';
constexpr double STICK_DEADZONE = 0.15;
constexpr double TRIGGER_THRESHOLD = 0.3;

namespace
{
	// Same keys as the original Play! web build (which used "Key1" etc. for the digit keys:
	// KeyboardEvent.code is "Digit1", so L1/L2/L3/R1/R2/R3 never worked from the keyboard).
	const CInputProviderEmscripten::KEYBINDING g_defaultKeyMap[] =
	    {
	        {"Enter", CControllerInfo::START, false},
	        {"Backspace", CControllerInfo::SELECT, false},
	        {"ArrowLeft", CControllerInfo::DPAD_LEFT, false},
	        {"ArrowRight", CControllerInfo::DPAD_RIGHT, false},
	        {"ArrowUp", CControllerInfo::DPAD_UP, false},
	        {"ArrowDown", CControllerInfo::DPAD_DOWN, false},
	        {"KeyA", CControllerInfo::SQUARE, false},
	        {"KeyZ", CControllerInfo::CROSS, false},
	        {"KeyS", CControllerInfo::TRIANGLE, false},
	        {"KeyX", CControllerInfo::CIRCLE, false},
	        {"Digit1", CControllerInfo::L1, false},
	        {"Digit2", CControllerInfo::L2, false},
	        {"Digit3", CControllerInfo::L3, false},
	        {"Digit8", CControllerInfo::R1, false},
	        {"Digit9", CControllerInfo::R2, false},
	        {"Digit0", CControllerInfo::R3, false},
	        {"KeyF", CControllerInfo::ANALOG_LEFT_X, false},
	        {"KeyH", CControllerInfo::ANALOG_LEFT_X, true},
	        {"KeyT", CControllerInfo::ANALOG_LEFT_Y, false},
	        {"KeyG", CControllerInfo::ANALOG_LEFT_Y, true},
	        {"KeyJ", CControllerInfo::ANALOG_RIGHT_X, false},
	        {"KeyL", CControllerInfo::ANALOG_RIGHT_X, true},
	        {"KeyI", CControllerInfo::ANALOG_RIGHT_Y, false},
	        {"KeyK", CControllerInfo::ANALOG_RIGHT_Y, true},
	};

	// W3C "standard" gamepad button index -> PS2 button (16, home/guide, unbound).
	const CControllerInfo::BUTTON g_defaultPadButtonMap[] =
	    {
	        CControllerInfo::CROSS,      // 0  A / Cross
	        CControllerInfo::CIRCLE,     // 1  B / Circle
	        CControllerInfo::SQUARE,     // 2  X / Square
	        CControllerInfo::TRIANGLE,   // 3  Y / Triangle
	        CControllerInfo::L1,         // 4  LB
	        CControllerInfo::R1,         // 5  RB
	        CControllerInfo::L2,         // 6  LT
	        CControllerInfo::R2,         // 7  RT
	        CControllerInfo::SELECT,     // 8  Back / Share
	        CControllerInfo::START,      // 9  Start / Options
	        CControllerInfo::L3,         // 10 left stick click
	        CControllerInfo::R3,         // 11 right stick click
	        CControllerInfo::DPAD_UP,    // 12
	        CControllerInfo::DPAD_DOWN,  // 13
	        CControllerInfo::DPAD_LEFT,  // 14
	        CControllerInfo::DPAD_RIGHT, // 15
	};

	uint32 AxisToPs2(double v)
	{
		long value = std::lround(127.0 + v * 128.0);
		return static_cast<uint32>(std::max(0L, std::min(255L, value)));
	}

	// Radial deadzone, rescaled so the stick still reaches full range.
	void ReadStick(double x, double y, uint32& outX, uint32& outY)
	{
		double mag = std::sqrt(x * x + y * y);
		if(mag < STICK_DEADZONE)
		{
			outX = outY = 0x7F;
			return;
		}
		double scale = std::min(1.0, (mag - STICK_DEADZONE) / (1.0 - STICK_DEADZONE)) / mag;
		outX = AxisToPs2(x * scale);
		outY = AxisToPs2(y * scale);
	}
}

CInputProviderEmscripten::CInputProviderEmscripten()
    : m_keyBindings(std::begin(g_defaultKeyMap), std::end(g_defaultKeyMap))
{
	std::fill(std::begin(m_padButtons), std::end(m_padButtons), -1);
	std::copy(std::begin(g_defaultPadButtonMap), std::end(g_defaultPadButtonMap), std::begin(m_padButtons));
}

void CInputProviderEmscripten::SetBindings(std::vector<KEYBINDING> keyBindings, const int (&padButtons)[MAX_PAD_BUTTONS])
{
	m_keyBindings = std::move(keyBindings);
	for(int i = 0; i < MAX_PAD_BUTTONS; i++)
	{
		int button = padButtons[i];
		m_padButtons[i] = (button >= 0 && button < CControllerInfo::MAX_BUTTONS && !CControllerInfo::IsAxis(static_cast<CControllerInfo::BUTTON>(button))) ? button : -1;
	}
	std::fill(std::begin(m_keyNeg), std::end(m_keyNeg), false);
	std::fill(std::begin(m_keyPos), std::end(m_keyPos), false);
	for(uint32 b = 0; b < CControllerInfo::MAX_BUTTONS; b++)
	{
		Publish(static_cast<CControllerInfo::BUTTON>(b));
	}
}

void CInputProviderEmscripten::SetGamepadEnabled(bool enabled)
{
	m_gamepadEnabled = enabled;
}

uint32 CInputProviderEmscripten::GetId() const
{
	return PROVIDER_ID;
}

std::string CInputProviderEmscripten::GetTargetDescription(const BINDINGTARGET& target) const
{
	return (target.keyId < CControllerInfo::MAX_BUTTONS) ? CControllerInfo::m_buttonName[target.keyId] : std::string();
}

BINDINGTARGET CInputProviderEmscripten::MakeBindingTarget(CControllerInfo::BUTTON button)
{
	auto keyType = CControllerInfo::IsAxis(button) ? BINDINGTARGET::KEYTYPE::AXIS : BINDINGTARGET::KEYTYPE::BUTTON;
	return BINDINGTARGET(PROVIDER_ID, DeviceIdType{{0}}, button, keyType);
}

void CInputProviderEmscripten::OnKeyDown(const EM_UTF8* code)
{
	OnKey(code, true);
}

void CInputProviderEmscripten::OnKeyUp(const EM_UTF8* code)
{
	OnKey(code, false);
}

void CInputProviderEmscripten::OnKey(const EM_UTF8* code, bool pressed)
{
	// A key may drive several controls (and a control may have several keys).
	for(const auto& key : m_keyBindings)
	{
		if(key.code != code) continue;
		(key.positive ? m_keyPos : m_keyNeg)[key.button] = pressed;
		Publish(key.button);
	}
}

void CInputProviderEmscripten::PollGamepads()
{
	uint32 padValue[CControllerInfo::MAX_BUTTONS] = {0x7F, 0x7F, 0x7F, 0x7F};

	if(m_gamepadEnabled && emscripten_sample_gamepad_data() == EMSCRIPTEN_RESULT_SUCCESS)
	{
		int count = emscripten_get_num_gamepads();
		for(int i = 0; i < count; i++)
		{
			EmscriptenGamepadEvent state;
			if(emscripten_get_gamepad_status(i, &state) != EMSCRIPTEN_RESULT_SUCCESS) continue;
			if(!state.connected) continue;

			// All connected gamepads drive pad 1 (buttons OR'ed, first moved stick wins).
			int buttonCount = std::min<int>(state.numButtons, MAX_PAD_BUTTONS);
			for(int b = 0; b < buttonCount; b++)
			{
				if((m_padButtons[b] >= 0) && (state.digitalButton[b] || (state.analogButton[b] > TRIGGER_THRESHOLD)))
				{
					padValue[m_padButtons[b]] = 1;
				}
			}
			if(state.numAxes >= 4)
			{
				uint32 lx, ly, rx, ry;
				ReadStick(state.axis[0], state.axis[1], lx, ly);
				ReadStick(state.axis[2], state.axis[3], rx, ry);
				if(padValue[CControllerInfo::ANALOG_LEFT_X] == 0x7F && padValue[CControllerInfo::ANALOG_LEFT_Y] == 0x7F)
				{
					padValue[CControllerInfo::ANALOG_LEFT_X] = lx;
					padValue[CControllerInfo::ANALOG_LEFT_Y] = ly;
				}
				if(padValue[CControllerInfo::ANALOG_RIGHT_X] == 0x7F && padValue[CControllerInfo::ANALOG_RIGHT_Y] == 0x7F)
				{
					padValue[CControllerInfo::ANALOG_RIGHT_X] = rx;
					padValue[CControllerInfo::ANALOG_RIGHT_Y] = ry;
				}
			}
		}
	}

	for(uint32 b = 0; b < CControllerInfo::MAX_BUTTONS; b++)
	{
		if(m_padValue[b] == padValue[b]) continue;
		m_padValue[b] = padValue[b];
		Publish(static_cast<CControllerInfo::BUTTON>(b));
	}
}

void CInputProviderEmscripten::Publish(CControllerInfo::BUTTON button)
{
	uint32 value = 0;
	if(CControllerInfo::IsAxis(button))
	{
		value = m_padValue[button];
		if(value == AXIS_NEUTRAL && (m_keyNeg[button] != m_keyPos[button]))
		{
			value = m_keyNeg[button] ? 0x00 : 0xFF;
		}
	}
	else
	{
		value = (m_keyNeg[button] || m_padValue[button]) ? 1 : 0;
	}
	if(value == m_lastValue[button]) return;
	m_lastValue[button] = value;
	OnInput(MakeBindingTarget(button), value);
}
