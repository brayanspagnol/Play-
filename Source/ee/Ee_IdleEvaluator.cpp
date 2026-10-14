#include "Ee_IdleEvaluator.h"

using namespace Ee;

void CIdleEvaluator::Reset()
{
	m_semaChecker.Reset();
	m_selfThreadRotate.Reset();
	m_threadRotateBounce.Reset();
	m_tightRotate.Reset();
}

bool CIdleEvaluator::IsIdle() const
{
	return m_semaChecker.m_isIdle ||
	       m_selfThreadRotate.m_isIdle ||
	       m_threadRotateBounce.m_isIdle ||
	       m_tightRotate.m_isIdle;
}

void CIdleEvaluator::NotifyEvent(EVENT eventType, uint32 arg0, uint32 arg1)
{
	m_semaChecker.NotifyEvent(eventType, arg0, arg1);
	m_selfThreadRotate.NotifyEvent(eventType, arg0, arg1);
	m_threadRotateBounce.NotifyEvent(eventType, arg0, arg1);
	if(eventType == EVENT_INTERRUPT)
	{
		m_tightRotate.m_interrupted = true;
	}
	m_tightRotate.m_isIdle = false;
}

void CIdleEvaluator::NotifyRotateThreadReadyQueue(uint32 threadIdBefore, uint32 threadIdAfter, uint32 executedCycles)
{
	NotifyEvent(EVENT_ROTATETHREADREADYQUEUE, threadIdBefore, threadIdAfter);

	auto& strategy = m_tightRotate;
	uint32 elapsed = executedCycles - strategy.m_lastCycles;
	strategy.m_lastCycles = executedCycles;
	if(elapsed <= STRATEGY_TIGHTROTATE::TIGHT_CYCLES)
	{
		strategy.m_streak++;
	}
	else if(!strategy.m_interrupted)
	{
		//Real work happened since the last rotation (interrupt handler cycles don't count)
		strategy.m_streak = 0;
	}
	strategy.m_interrupted = false;
	strategy.m_isIdle = (strategy.m_streak >= STRATEGY_TIGHTROTATE::THRESHOLD);
}

void CIdleEvaluator::BeginExecution()
{
	m_tightRotate.m_isIdle = false;
}

void CIdleEvaluator::STRATEGY_TIGHTROTATE::Reset()
{
	m_lastCycles = 0;
	m_streak = 0;
	m_interrupted = false;
	m_isIdle = false;
}

void CIdleEvaluator::STRATEGY_SEMACHECKER::Reset()
{
	m_lastWaitSema = -1;
	m_lastSemaWaitCounter = 0;
	m_isIdle = false;
}

void CIdleEvaluator::STRATEGY_SEMACHECKER::NotifyEvent(EVENT eventType, uint32 arg0, uint32 arg1)
{
	switch(eventType)
	{
	case EVENT_WAITSEMA:
	case EVENT_SIGNALSEMA:
		if(m_lastWaitSema == arg0)
		{
			m_lastSemaWaitCounter++;
			if(m_lastSemaWaitCounter >= 100)
			{
				m_isIdle = true;
			}
		}
		else
		{
			m_lastSemaWaitCounter = 0;
			m_isIdle = false;
		}
		m_lastWaitSema = arg0;
		break;
	case EVENT_INTERRUPT:
	case EVENT_CHANGETHREAD:
		m_lastSemaWaitCounter = 0;
		m_lastWaitSema = -1;
		m_isIdle = false;
		break;
	default:
		break;
	}
}

void CIdleEvaluator::STRATEGY_SELFTHREADROTATE::Reset()
{
	m_threadId = -1;
	m_selfRotateThreadCount = 0;
	m_isIdle = false;
}

void CIdleEvaluator::STRATEGY_SELFTHREADROTATE::NotifyEvent(EVENT eventType, uint32 arg0, uint32 arg1)
{
	switch(eventType)
	{
	case EVENT_ROTATETHREADREADYQUEUE:
		if(arg0 == arg1)
		{
			m_selfRotateThreadCount++;
			m_threadId = arg0;
			if(m_selfRotateThreadCount > 500)
			{
				m_isIdle = true;
			}
		}
		else
		{
			m_selfRotateThreadCount = 0;
			m_threadId = -1;
			m_isIdle = false;
		}
		break;
	case EVENT_CHANGETHREAD:
		if(arg0 != m_threadId)
		{
			m_selfRotateThreadCount = 0;
			m_isIdle = false;
		}
		break;
	case EVENT_INTERRUPT:
		m_selfRotateThreadCount = 0;
		m_isIdle = false;
		break;
	default:
		break;
	}
}

void CIdleEvaluator::STRATEGY_THREADROTATEBOUNCE::Reset()
{
	m_threadPairFirst = -1;
	m_threadPairSecond = -1;
	m_bounceCount = 0;
	m_isIdle = false;
}

void CIdleEvaluator::STRATEGY_THREADROTATEBOUNCE::NotifyEvent(EVENT eventType, uint32 arg0, uint32 arg1)
{
	switch(eventType)
	{
	case EVENT_ROTATETHREADREADYQUEUE:
	{
		bool hasFirstThread = (arg0 == m_threadPairFirst) || (arg1 == m_threadPairFirst);
		bool hasSecondThread = (arg0 == m_threadPairSecond) || (arg1 == m_threadPairSecond);
		if(hasFirstThread && hasSecondThread)
		{
			m_bounceCount++;
		}
		else
		{
			m_bounceCount = 0;
		}
		m_threadPairFirst = arg0;
		m_threadPairSecond = arg1;
	}
	break;
	case EVENT_CHANGETHREAD:
		if(arg0 != m_threadPairFirst && arg0 != m_threadPairSecond)
		{
			m_bounceCount = 0;
		}
		break;
	case EVENT_INTERRUPT:
		m_bounceCount = 0;
		break;
	default:
		break;
	}
	m_isIdle = (m_bounceCount > 1000);
}
