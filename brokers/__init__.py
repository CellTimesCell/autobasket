"""
AutoBasket - Brokers Module
===========================
Abstraction layer for different execution modes:
- Paper Trading (virtual money)
- Real Money (Kalshi API)
"""

from .base_broker import BaseBroker, BrokerMode, BetResult
from .paper_broker import PaperBroker
from .kalshi_broker import KalshiBroker

__all__ = ['BaseBroker', 'BrokerMode', 'BetResult', 'PaperBroker', 'KalshiBroker']
