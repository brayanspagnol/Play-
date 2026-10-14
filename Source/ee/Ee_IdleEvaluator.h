#pragma once

#include "Types.h"

namespace Ee
{
	class CIdleEvaluator
	{
	public:
		void Reset();
		bool IsIdle() const;

		enum EVENT
		{
			EVENT_INTERRUPT,
			EVENT_WAITSEMA,
			EVENT_SIGNALSEMA,
			EVENT_ROTATETHREADREADYQUEUE,
			EVENT_CHANGETHREAD,
		};

		void NotifyEvent(EVENT, uint32, uint32 = 0);

		//Tight rotation detection: 'executedCycles' is the running count of EE cycles actually
		//executed (idle-skipped cycles excluded). The resulting idle state only lasts until the
		//next execution chunk starts (BeginExecution), so threads doing real work are never throttled.
		void NotifyRotateThreadReadyQueue(uint32 threadIdBefore, uint32 threadIdAfter, uint32 executedCycles);
		void BeginExecution();

	private:
		//For Shadow of the Colossus
		struct STRATEGY_SEMACHECKER
		{
			void Reset();
			void NotifyEvent(EVENT, uint32, uint32);

			int32 m_lastWaitSema = -1;
			uint32 m_lastSemaWaitCounter = 0;
			bool m_isIdle = false;
		} m_semaChecker;

		//For Atelier Elie & Marie: Activated when RotateThreadReadyQueue didn't have any effect
		struct STRATEGY_SELFTHREADROTATE
		{
			void Reset();
			void NotifyEvent(EVENT, uint32, uint32);

			uint32 m_threadId = -1;
			uint32 m_selfRotateThreadCount = 0;
			bool m_isIdle = false;
		} m_selfThreadRotate;

		//For GTA San Andreas: threads at the same priority yielding to each other with
		//RotateThreadReadyQueue (an idle spinner plus a thread polling a flag) while frequent
		//interrupts keep resetting the other strategies' counters. Rotations a few cycles apart
		//mean nothing else happens on the EE until an interrupt makes progress possible.
		struct STRATEGY_TIGHTROTATE
		{
			enum
			{
				TIGHT_CYCLES = 256,
				THRESHOLD = 16,
			};

			void Reset();

			uint32 m_lastCycles = 0;
			uint32 m_streak = 0;
			bool m_interrupted = false;
			bool m_isIdle = false;
		} m_tightRotate;

		//For Ys: Ark of Napishtim
		struct STRATEGY_THREADROTATEBOUNCE
		{
			void Reset();
			void NotifyEvent(EVENT, uint32, uint32);

			uint32 m_threadPairFirst = -1;
			uint32 m_threadPairSecond = -1;
			uint32 m_bounceCount = 0;
			bool m_isIdle = false;
		} m_threadRotateBounce;
	};
}
