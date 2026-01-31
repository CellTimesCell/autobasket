"""
AutoBasket - Paper Trading Broker
=================================
Virtual money broker for testing and simulation.
No real money is risked - all bets are simulated.
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Optional

from .base_broker import (
    BaseBroker, BrokerMode, BetResult, BetStatus, Position
)

logger = logging.getLogger(__name__)


class PaperBroker(BaseBroker):
    """
    Paper trading broker for simulation.

    Features:
    - Simulates bet placement with instant confirmation
    - Tracks virtual balance
    - No real money risked
    - Perfect for testing strategies

    Usage:
        broker = PaperBroker(initial_balance=200.0)
        broker.connect()

        result = broker.place_bet(
            game_id="12345",
            home_team="Lakers",
            away_team="Warriors",
            side="home",
            amount=10.0,
            odds=1.85
        )

        if result.success:
            print(f"Bet placed: {result.bet_id}")

        # When game finishes:
        profit = broker.settle_bet(result.bet_id, won=True, home_score=110, away_score=105)
    """

    def __init__(self, initial_balance: float = 200.0):
        super().__init__(mode=BrokerMode.PAPER, initial_balance=initial_balance)
        self._simulated_delay = 0.0  # Optional delay to simulate API latency

    def connect(self) -> bool:
        """Connect to paper trading (always succeeds)"""
        self._connected = True
        logger.info("📝 Paper trading broker connected")
        logger.info(f"💰 Starting balance: ${self._balance:.2f}")
        return True

    def disconnect(self):
        """Disconnect from paper trading"""
        self._connected = False
        logger.info("📝 Paper trading broker disconnected")

    def get_balance(self) -> float:
        """Get current virtual balance"""
        return self._balance

    def place_bet(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        side: str,
        amount: float,
        odds: float,
        **kwargs
    ) -> BetResult:
        """
        Place a virtual bet.

        The bet is immediately confirmed (no real API call).
        Balance is reduced by the bet amount.
        """
        if not self._connected:
            return BetResult(
                success=False,
                message="Broker not connected"
            )

        # Validate
        if amount <= 0:
            return BetResult(
                success=False,
                message="Bet amount must be positive"
            )

        if amount > self._balance:
            return BetResult(
                success=False,
                message=f"Insufficient balance. Have ${self._balance:.2f}, need ${amount:.2f}"
            )

        if side not in ['home', 'away']:
            return BetResult(
                success=False,
                message=f"Invalid side: {side}. Must be 'home' or 'away'"
            )

        if odds <= 1.0:
            return BetResult(
                success=False,
                message=f"Invalid odds: {odds}. Must be > 1.0"
            )

        # Generate bet ID
        bet_id = f"PAPER_{uuid.uuid4().hex[:8].upper()}"

        # Deduct from balance
        self._balance -= amount

        # Create position
        position = Position(
            bet_id=bet_id,
            game_id=str(game_id),
            home_team=home_team,
            away_team=away_team,
            side=side,
            amount=amount,
            odds=odds,
            status=BetStatus.CONFIRMED,
            placed_at=datetime.now()
        )

        self.positions[bet_id] = position

        bet_team = home_team if side == 'home' else away_team
        logger.info(f"📝 PAPER BET: ${amount:.2f} on {bet_team} @ {odds:.2f}")
        logger.info(f"   Potential win: ${amount * (odds - 1):.2f}")
        logger.info(f"   Balance: ${self._balance:.2f}")

        return BetResult(
            success=True,
            bet_id=bet_id,
            message="Paper bet placed successfully",
            executed_amount=amount,
            executed_odds=odds,
            commission=0.0
        )

    def cancel_bet(self, bet_id: str) -> bool:
        """
        Cancel a paper bet.

        In paper trading, bets can always be cancelled before settlement.
        """
        if bet_id not in self.positions:
            logger.warning(f"Bet {bet_id} not found")
            return False

        position = self.positions[bet_id]

        if position.status not in [BetStatus.PENDING, BetStatus.CONFIRMED]:
            logger.warning(f"Bet {bet_id} cannot be cancelled (status: {position.status})")
            return False

        # Refund the amount
        self._balance += position.amount
        position.status = BetStatus.CANCELLED

        # Move to history
        del self.positions[bet_id]
        self.history.append(position)

        logger.info(f"📝 Paper bet {bet_id} cancelled, ${position.amount:.2f} refunded")
        return True

    def settle_bet(
        self,
        bet_id: str,
        won: bool,
        home_score: int,
        away_score: int
    ) -> float:
        """
        Settle a paper bet.

        Args:
            bet_id: The bet to settle
            won: Whether the bet won
            home_score: Final home score
            away_score: Final away score

        Returns:
            Profit/loss amount
        """
        if bet_id not in self.positions:
            logger.warning(f"Bet {bet_id} not found for settlement")
            return 0.0

        position = self.positions[bet_id]

        if position.status not in [BetStatus.PENDING, BetStatus.CONFIRMED]:
            logger.warning(f"Bet {bet_id} already settled")
            return position.profit

        # Calculate profit/loss
        if won:
            profit = position.amount * (position.odds - 1)
            self._balance += position.amount + profit  # Return stake + profit
            position.status = BetStatus.WON
            position.profit = profit
            emoji = "✅"
        else:
            profit = -position.amount
            position.status = BetStatus.LOST
            position.profit = profit
            emoji = "❌"

        position.settled_at = datetime.now()
        position.final_score = f"{away_score}-{home_score}"

        # Move to history
        del self.positions[bet_id]
        self.history.append(position)

        bet_team = position.home_team if position.side == 'home' else position.away_team
        logger.info(f"{emoji} PAPER BET SETTLED: {bet_team}")
        logger.info(f"   Score: {position.away_team} {away_score} - {home_score} {position.home_team}")
        logger.info(f"   P&L: ${profit:+.2f}")
        logger.info(f"   Balance: ${self._balance:.2f}")

        return profit

    def simulate_market_conditions(
        self,
        slippage: float = 0.0,
        rejection_rate: float = 0.0
    ):
        """
        Configure simulation parameters.

        Args:
            slippage: Maximum odds slippage (0.0-0.1)
            rejection_rate: Probability of bet rejection (0.0-1.0)
        """
        # TODO: Implement market simulation for more realistic paper trading
        pass

    def export_history(self) -> list:
        """Export betting history as list of dicts"""
        return [
            {
                'bet_id': p.bet_id,
                'game_id': p.game_id,
                'home_team': p.home_team,
                'away_team': p.away_team,
                'side': p.side,
                'amount': p.amount,
                'odds': p.odds,
                'status': p.status.value,
                'placed_at': p.placed_at.isoformat(),
                'settled_at': p.settled_at.isoformat() if p.settled_at else None,
                'profit': p.profit,
                'final_score': p.final_score
            }
            for p in self.history
        ]
