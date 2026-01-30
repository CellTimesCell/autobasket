"""
AutoBasket - Live Game Monitor
==============================
Отслеживание игр в реальном времени
Сравнение прогноза с реальным ходом игры
Детекция аномалий для live-ставок

Аномалии которые отслеживаем:
- Аутсайдер лидирует с большим отрывом
- Фаворит неожиданно проигрывает
- Резкий momentum shift (10+ очков за 3 минуты)
- Команда "проснулась" после плохого старта
- Травма ключевого игрока во время матча
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from nba_api.live.nba.endpoints import scoreboard, boxscore
    NBA_LIVE_AVAILABLE = True
except ImportError:
    NBA_LIVE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GameStatus(Enum):
    """Статус игры"""
    SCHEDULED = "scheduled"
    PREGAME = "pregame"
    LIVE_Q1 = "q1"
    LIVE_Q2 = "q2"
    HALFTIME = "halftime"
    LIVE_Q3 = "q3"
    LIVE_Q4 = "q4"
    OVERTIME = "overtime"
    FINAL = "final"
    POSTPONED = "postponed"


class AnomalyType(Enum):
    """Типы аномалий в live-игре"""
    UNDERDOG_LEADING = "underdog_leading"          # Аутсайдер лидирует
    FAVORITE_STRUGGLING = "favorite_struggling"    # Фаворит проигрывает
    MOMENTUM_SHIFT = "momentum_shift"              # Резкое изменение momentum
    COMEBACK_IN_PROGRESS = "comeback_in_progress"  # Камбэк в процессе
    BLOWOUT_DEVELOPING = "blowout_developing"      # Разгром развивается
    INJURY_IMPACT = "injury_impact"                # Травма влияет на игру
    UNEXPECTED_MARGIN = "unexpected_margin"        # Неожиданный счёт
    QUARTER_ANOMALY = "quarter_anomaly"            # Аномалия в четверти
    LIVE_BET_OPPORTUNITY = "live_bet_opportunity"  # Возможность для live-ставки


@dataclass
class LiveAnomaly:
    """Обнаруженная аномалия в игре"""
    anomaly_type: AnomalyType
    game_id: str
    timestamp: datetime
    
    # Детали
    description: str
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # Уверенность в аномалии 0-1
    
    # Данные для анализа
    expected_margin: float       # Ожидаемый счёт
    actual_margin: float         # Реальный счёт
    deviation: float             # Отклонение от ожидания
    
    # Ставка (если есть opportunity)
    bet_opportunity: bool = False
    recommended_side: str = ""   # "home" or "away"
    recommended_odds: float = 0.0
    edge_estimate: float = 0.0
    
    # Контекст
    quarter: int = 0
    time_remaining: str = ""
    home_team: str = ""
    away_team: str = ""
    current_score: str = ""


@dataclass
class LiveScore:
    """Текущий счет"""
    home_score: int = 0
    away_score: int = 0
    
    home_q1: int = 0
    home_q2: int = 0
    home_q3: int = 0
    home_q4: int = 0
    home_ot: int = 0
    
    away_q1: int = 0
    away_q2: int = 0
    away_q3: int = 0
    away_q4: int = 0
    away_ot: int = 0
    
    quarter: int = 0
    time_remaining: str = ""
    
    @property
    def total(self) -> int:
        return self.home_score + self.away_score
    
    @property
    def margin(self) -> int:
        return self.home_score - self.away_score


@dataclass
class LiveGameState:
    """Состояние живой игры"""
    game_id: str
    home_team: str
    away_team: str
    status: GameStatus
    
    # Текущий счет
    score: LiveScore = field(default_factory=LiveScore)
    
    # Время
    start_time: datetime = None
    last_update: datetime = None
    
    # Наш прогноз (для сравнения)
    predicted_home_prob: float = 0.5
    predicted_margin: float = 0.0
    predicted_total: float = 220.0
    bet_placed: bool = False
    bet_side: str = ""  # "home", "away", "over", "under"
    bet_amount: float = 0.0
    
    # Расчеты в реальном времени
    live_home_win_prob: float = 0.5
    is_on_track: bool = True
    deviation_from_prediction: float = 0.0
    
    # История обновлений
    score_history: List[Dict] = field(default_factory=list)


@dataclass
class LiveAlert:
    """Алерт о событии в игре"""
    timestamp: datetime
    game_id: str
    alert_type: str  # "momentum_shift", "blowout", "close_game", "injury", "bet_at_risk"
    message: str
    severity: str  # "info", "warning", "critical"
    data: Dict = field(default_factory=dict)


class AnomalyDetector:
    """
    Детектор аномалий в live-играх
    
    Ищет ситуации как твой пример с Lakers vs Dallas:
    - Аутсайдер неожиданно лидирует
    - Фаворит проигрывает во 2-м тайме
    - Momentum shift (команда "проснулась")
    
    Использование:
        detector = AnomalyDetector()
        anomalies = detector.detect_anomalies(game_state, pre_game_prediction)
        
        for anomaly in anomalies:
            if anomaly.bet_opportunity:
                print(f"🎯 Live bet opportunity: {anomaly.recommended_side}")
    """
    
    # Пороги для детекции
    UNDERDOG_LEAD_THRESHOLD = 6       # Аутсайдер лидирует на 6+ очков
    FAVORITE_STRUGGLE_THRESHOLD = -5  # Фаворит проигрывает 5+ очков
    MOMENTUM_SHIFT_POINTS = 10        # 10 очков подряд без ответа
    COMEBACK_THRESHOLD = 8            # Сократили отставание на 8+ очков
    BLOWOUT_THRESHOLD = 15            # Разгром 15+ очков
    
    # Важность по четвертям (Q2 - самое важное для live ставок)
    QUARTER_WEIGHTS = {
        1: 0.7,   # Q1 - рано судить
        2: 1.0,   # Q2 - лучшее время для live ставок (как в твоём примере)
        3: 0.9,   # Q3 - всё ещё хорошо
        4: 0.6    # Q4 - поздно, odds уже adjusted
    }
    
    def __init__(self):
        # История аномалий (чтобы не дублировать)
        self.detected_anomalies: Dict[str, List[LiveAnomaly]] = {}
        
        # Cooldown - не спамим одинаковые аномалии
        self.cooldowns: Dict[str, datetime] = {}
        self.cooldown_minutes = 5
    
    def detect_anomalies(
        self,
        game: LiveGameState,
        pre_game_odds: Dict = None,
        historical_h2h: Dict = None
    ) -> List[LiveAnomaly]:
        """
        Главный метод - детектирует все аномалии в игре
        
        Args:
            game: Текущее состояние игры
            pre_game_odds: Pre-game odds (implied probability)
            historical_h2h: История встреч команд
        
        Returns:
            Список обнаруженных аномалий
        """
        anomalies = []
        
        # Пропускаем если игра не live
        if game.status not in [GameStatus.LIVE_Q1, GameStatus.LIVE_Q2, 
                               GameStatus.LIVE_Q3, GameStatus.LIVE_Q4,
                               GameStatus.HALFTIME]:
            return anomalies
        
        # Получаем quarter weight
        quarter = game.score.quarter if game.score.quarter > 0 else 2
        q_weight = self.QUARTER_WEIGHTS.get(quarter, 0.8)
        
        # Pre-game prediction (если не передали, используем из game)
        expected_home_prob = pre_game_odds.get('home_prob', game.predicted_home_prob) if pre_game_odds else game.predicted_home_prob
        
        # Кто был фаворитом?
        home_was_favorite = expected_home_prob > 0.55
        away_was_favorite = expected_home_prob < 0.45
        
        # Текущий margin (positive = home winning)
        current_margin = game.score.margin
        
        # 1. UNDERDOG LEADING - Аутсайдер лидирует
        if home_was_favorite and current_margin < -self.UNDERDOG_LEAD_THRESHOLD:
            # Away (аутсайдер) лидирует против фаворита
            anomaly = self._create_underdog_anomaly(
                game, "away", expected_home_prob, current_margin, q_weight
            )
            if anomaly and self._check_cooldown(game.game_id, "underdog_away"):
                anomalies.append(anomaly)
        
        elif away_was_favorite and current_margin > self.UNDERDOG_LEAD_THRESHOLD:
            # Home (аутсайдер) лидирует против фаворита
            anomaly = self._create_underdog_anomaly(
                game, "home", expected_home_prob, current_margin, q_weight
            )
            if anomaly and self._check_cooldown(game.game_id, "underdog_home"):
                anomalies.append(anomaly)
        
        # 2. FAVORITE STRUGGLING - Фаворит в беде
        if home_was_favorite and current_margin < self.FAVORITE_STRUGGLE_THRESHOLD:
            anomaly = self._create_favorite_struggling_anomaly(
                game, "home", expected_home_prob, current_margin, q_weight
            )
            if anomaly and self._check_cooldown(game.game_id, "fav_struggle"):
                anomalies.append(anomaly)
        
        # 3. MOMENTUM SHIFT - Проверяем историю счёта
        momentum_anomaly = self._detect_momentum_shift(game, q_weight)
        if momentum_anomaly and self._check_cooldown(game.game_id, "momentum"):
            anomalies.append(momentum_anomaly)
        
        # 4. UNEXPECTED MARGIN - Счёт сильно отличается от ожидания
        expected_margin = (expected_home_prob - 0.5) * 15  # Примерная конвертация
        margin_deviation = abs(current_margin - expected_margin)
        
        if margin_deviation > 10 and quarter >= 2:
            anomaly = self._create_unexpected_margin_anomaly(
                game, expected_margin, current_margin, margin_deviation, q_weight
            )
            if anomaly and self._check_cooldown(game.game_id, "margin"):
                anomalies.append(anomaly)
        
        # 5. LIVE BET OPPORTUNITY - Комбинированная оценка
        if anomalies:
            best_opportunity = self._evaluate_live_bet_opportunity(
                game, anomalies, expected_home_prob, q_weight
            )
            if best_opportunity:
                anomalies.append(best_opportunity)
        
        # Сохраняем
        if game.game_id not in self.detected_anomalies:
            self.detected_anomalies[game.game_id] = []
        self.detected_anomalies[game.game_id].extend(anomalies)
        
        return anomalies
    
    def _check_cooldown(self, game_id: str, anomaly_key: str) -> bool:
        """Проверяет cooldown для аномалии"""
        key = f"{game_id}_{anomaly_key}"
        
        if key in self.cooldowns:
            if datetime.now() - self.cooldowns[key] < timedelta(minutes=self.cooldown_minutes):
                return False
        
        self.cooldowns[key] = datetime.now()
        return True
    
    def _create_underdog_anomaly(
        self, game: LiveGameState, underdog_side: str,
        expected_prob: float, margin: int, q_weight: float
    ) -> Optional[LiveAnomaly]:
        """Создаёт аномалию 'аутсайдер лидирует'"""
        
        if underdog_side == "away":
            underdog_team = game.away_team
            favorite_team = game.home_team
            lead = abs(margin)
            underdog_pre_prob = 1 - expected_prob
        else:
            underdog_team = game.home_team
            favorite_team = game.away_team
            lead = margin
            underdog_pre_prob = expected_prob
        
        # Рассчитываем edge
        # Если аутсайдер был 35% и лидирует на 8 очков во 2-м тайме,
        # его реальная вероятность выиграть уже ~55-60%
        live_prob_estimate = 0.5 + (lead / 30)  # Грубая оценка
        live_prob_estimate = min(0.75, max(0.25, live_prob_estimate))
        
        # Edge = live probability - pre-game odds
        edge = live_prob_estimate - underdog_pre_prob
        
        # Только если edge > 5%
        if edge < 0.05:
            return None
        
        severity = "high" if lead >= 10 and game.score.quarter == 2 else "medium"
        confidence = min(0.9, 0.5 + (lead / 20) + (q_weight * 0.2))
        
        return LiveAnomaly(
            anomaly_type=AnomalyType.UNDERDOG_LEADING,
            game_id=game.game_id,
            timestamp=datetime.now(),
            description=f"🔥 АНОМАЛИЯ: {underdog_team} (аутсайдер) лидирует +{lead} против {favorite_team}!",
            severity=severity,
            confidence=confidence,
            expected_margin=-lead if underdog_side == "away" else lead,
            actual_margin=margin,
            deviation=abs(margin),
            bet_opportunity=edge > 0.08,
            recommended_side=underdog_side,
            recommended_odds=1 / (1 - underdog_pre_prob + 0.05),  # Примерные live odds
            edge_estimate=edge,
            quarter=game.score.quarter,
            time_remaining=game.score.time_remaining,
            home_team=game.home_team,
            away_team=game.away_team,
            current_score=f"{game.score.away_score}-{game.score.home_score}"
        )
    
    def _create_favorite_struggling_anomaly(
        self, game: LiveGameState, favorite_side: str,
        expected_prob: float, margin: int, q_weight: float
    ) -> Optional[LiveAnomaly]:
        """Создаёт аномалию 'фаворит в беде'"""
        
        favorite_team = game.home_team if favorite_side == "home" else game.away_team
        opponent_team = game.away_team if favorite_side == "home" else game.home_team
        deficit = abs(margin)
        
        severity = "high" if deficit >= 8 else "medium"
        confidence = min(0.85, 0.4 + (deficit / 15) + (q_weight * 0.2))
        
        return LiveAnomaly(
            anomaly_type=AnomalyType.FAVORITE_STRUGGLING,
            game_id=game.game_id,
            timestamp=datetime.now(),
            description=f"⚠️ {favorite_team} (фаворит {expected_prob:.0%}) проигрывает {deficit} очков!",
            severity=severity,
            confidence=confidence,
            expected_margin=(expected_prob - 0.5) * 15,
            actual_margin=margin,
            deviation=deficit,
            bet_opportunity=False,  # Ставить против фаворита рискованно
            quarter=game.score.quarter,
            time_remaining=game.score.time_remaining,
            home_team=game.home_team,
            away_team=game.away_team,
            current_score=f"{game.score.away_score}-{game.score.home_score}"
        )
    
    def _detect_momentum_shift(self, game: LiveGameState, q_weight: float) -> Optional[LiveAnomaly]:
        """Детектирует momentum shift из истории счёта"""
        
        history = game.score_history
        if len(history) < 3:
            return None
        
        # Смотрим последние 5 записей
        recent = history[-5:]
        
        # Ищем run (серию очков без ответа)
        home_run = 0
        away_run = 0
        
        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]
            
            home_diff = curr.get('home_score', 0) - prev.get('home_score', 0)
            away_diff = curr.get('away_score', 0) - prev.get('away_score', 0)
            
            if home_diff > 0 and away_diff == 0:
                home_run += home_diff
            elif away_diff > 0 and home_diff == 0:
                away_run += away_diff
            else:
                # Обе команды забили - сбрасываем
                home_run = max(0, home_diff)
                away_run = max(0, away_diff)
        
        max_run = max(home_run, away_run)
        
        if max_run >= self.MOMENTUM_SHIFT_POINTS:
            running_team = game.home_team if home_run > away_run else game.away_team
            running_side = "home" if home_run > away_run else "away"
            
            return LiveAnomaly(
                anomaly_type=AnomalyType.MOMENTUM_SHIFT,
                game_id=game.game_id,
                timestamp=datetime.now(),
                description=f"🏃 MOMENTUM SHIFT: {running_team} набрали {max_run} очков подряд!",
                severity="medium",
                confidence=0.7 * q_weight,
                expected_margin=0,
                actual_margin=game.score.margin,
                deviation=max_run,
                bet_opportunity=max_run >= 12,
                recommended_side=running_side,
                edge_estimate=0.05 if max_run >= 12 else 0,
                quarter=game.score.quarter,
                time_remaining=game.score.time_remaining,
                home_team=game.home_team,
                away_team=game.away_team,
                current_score=f"{game.score.away_score}-{game.score.home_score}"
            )
        
        return None
    
    def _create_unexpected_margin_anomaly(
        self, game: LiveGameState, expected_margin: float,
        actual_margin: int, deviation: float, q_weight: float
    ) -> LiveAnomaly:
        """Создаёт аномалию неожиданного счёта"""
        
        leading_team = game.home_team if actual_margin > 0 else game.away_team
        
        return LiveAnomaly(
            anomaly_type=AnomalyType.UNEXPECTED_MARGIN,
            game_id=game.game_id,
            timestamp=datetime.now(),
            description=f"📊 Неожиданный счёт: {leading_team} +{abs(actual_margin)} (ожидали ~{expected_margin:+.0f})",
            severity="medium" if deviation < 15 else "high",
            confidence=min(0.8, 0.5 + deviation / 30),
            expected_margin=expected_margin,
            actual_margin=actual_margin,
            deviation=deviation,
            quarter=game.score.quarter,
            time_remaining=game.score.time_remaining,
            home_team=game.home_team,
            away_team=game.away_team,
            current_score=f"{game.score.away_score}-{game.score.home_score}"
        )
    
    def _evaluate_live_bet_opportunity(
        self, game: LiveGameState, anomalies: List[LiveAnomaly],
        expected_home_prob: float, q_weight: float
    ) -> Optional[LiveAnomaly]:
        """
        Оценивает есть ли хорошая возможность для live-ставки
        
        Критерии:
        - Q2 или Q3 (не слишком рано, не слишком поздно)
        - Edge > 8%
        - Несколько аномалий указывают в одну сторону
        """
        
        if game.score.quarter not in [2, 3]:
            return None
        
        # Собираем сигналы
        home_signals = 0
        away_signals = 0
        total_edge = 0
        
        for a in anomalies:
            if a.bet_opportunity:
                if a.recommended_side == "home":
                    home_signals += 1
                    total_edge += a.edge_estimate
                elif a.recommended_side == "away":
                    away_signals += 1
                    total_edge += a.edge_estimate
        
        # Нужно минимум 1 сигнал и хороший edge
        best_side = "home" if home_signals > away_signals else "away"
        signals = max(home_signals, away_signals)
        
        if signals >= 1 and total_edge >= 0.08:
            return LiveAnomaly(
                anomaly_type=AnomalyType.LIVE_BET_OPPORTUNITY,
                game_id=game.game_id,
                timestamp=datetime.now(),
                description=f"🎯 LIVE BET OPPORTUNITY: Ставь на {game.home_team if best_side == 'home' else game.away_team}!",
                severity="critical",
                confidence=min(0.9, 0.6 + signals * 0.1 + total_edge),
                expected_margin=(expected_home_prob - 0.5) * 15,
                actual_margin=game.score.margin,
                deviation=abs(game.score.margin - (expected_home_prob - 0.5) * 15),
                bet_opportunity=True,
                recommended_side=best_side,
                edge_estimate=total_edge,
                quarter=game.score.quarter,
                time_remaining=game.score.time_remaining,
                home_team=game.home_team,
                away_team=game.away_team,
                current_score=f"{game.score.away_score}-{game.score.home_score}"
            )
        
        return None
    
    def get_game_anomalies(self, game_id: str) -> List[LiveAnomaly]:
        """Возвращает все аномалии для игры"""
        return self.detected_anomalies.get(game_id, [])
    
    def format_anomaly_alert(self, anomaly: LiveAnomaly) -> str:
        """Форматирует аномалию для уведомления"""
        
        emoji_map = {
            AnomalyType.UNDERDOG_LEADING: "🔥",
            AnomalyType.FAVORITE_STRUGGLING: "⚠️",
            AnomalyType.MOMENTUM_SHIFT: "🏃",
            AnomalyType.LIVE_BET_OPPORTUNITY: "🎯",
            AnomalyType.UNEXPECTED_MARGIN: "📊"
        }
        
        emoji = emoji_map.get(anomaly.anomaly_type, "❗")
        
        text = f"""
{emoji} LIVE ANOMALY DETECTED
━━━━━━━━━━━━━━━━━━━━━━━

{anomaly.away_team} @ {anomaly.home_team}
Q{anomaly.quarter} {anomaly.time_remaining}
Score: {anomaly.current_score}

{anomaly.description}

Confidence: {anomaly.confidence:.0%}
"""
        
        if anomaly.bet_opportunity:
            text += f"""
🎯 BET OPPORTUNITY:
   Side: {anomaly.recommended_side.upper()}
   Edge: {anomaly.edge_estimate:.1%}
"""
        
        return text


class LiveScoreProvider:
    """
    Провайдер live данных
    Использует ESPN API (бесплатно, неофициально)
    """
    
    ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    ESPN_GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
    
    def __init__(self):
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
    
    def get_live_games(self) -> List[LiveGameState]:
        """Получает все текущие игры"""
        if not REQUESTS_AVAILABLE:
            return []
        
        try:
            response = self.session.get(self.ESPN_SCOREBOARD_URL, timeout=10)
            data = response.json()
            
            games = []
            
            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                
                # Команды
                competitors = competition.get('competitors', [])
                if len(competitors) != 2:
                    continue
                
                home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                
                # Статус
                status_data = event.get('status', {})
                status_type = status_data.get('type', {}).get('state', 'pre')
                period = status_data.get('period', 0)
                
                if status_type == 'pre':
                    status = GameStatus.SCHEDULED
                elif status_type == 'in':
                    if period == 1:
                        status = GameStatus.LIVE_Q1
                    elif period == 2:
                        status = GameStatus.LIVE_Q2
                    elif period == 3:
                        status = GameStatus.LIVE_Q3
                    elif period == 4:
                        status = GameStatus.LIVE_Q4
                    else:
                        status = GameStatus.OVERTIME
                elif status_type == 'post':
                    status = GameStatus.FINAL
                else:
                    status = GameStatus.SCHEDULED
                
                # Score
                score = LiveScore(
                    home_score=int(home.get('score', 0) or 0),
                    away_score=int(away.get('score', 0) or 0),
                    quarter=period,
                    time_remaining=status_data.get('displayClock', '')
                )
                
                # Получаем счет по четвертям из linescores
                home_linescores = home.get('linescores', [])
                away_linescores = away.get('linescores', [])
                
                if len(home_linescores) >= 1:
                    score.home_q1 = int(home_linescores[0].get('value', 0))
                if len(home_linescores) >= 2:
                    score.home_q2 = int(home_linescores[1].get('value', 0))
                if len(home_linescores) >= 3:
                    score.home_q3 = int(home_linescores[2].get('value', 0))
                if len(home_linescores) >= 4:
                    score.home_q4 = int(home_linescores[3].get('value', 0))
                    
                if len(away_linescores) >= 1:
                    score.away_q1 = int(away_linescores[0].get('value', 0))
                if len(away_linescores) >= 2:
                    score.away_q2 = int(away_linescores[1].get('value', 0))
                if len(away_linescores) >= 3:
                    score.away_q3 = int(away_linescores[2].get('value', 0))
                if len(away_linescores) >= 4:
                    score.away_q4 = int(away_linescores[3].get('value', 0))
                
                game = LiveGameState(
                    game_id=event.get('id', ''),
                    home_team=home.get('team', {}).get('displayName', ''),
                    away_team=away.get('team', {}).get('displayName', ''),
                    status=status,
                    score=score,
                    last_update=datetime.now()
                )
                
                games.append(game)
            
            return games
            
        except Exception as e:
            logger.error(f"Ошибка получения live данных: {e}")
            return []
    
    def get_game_details(self, game_id: str) -> Optional[Dict]:
        """Получает детали конкретной игры"""
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            response = self.session.get(
                self.ESPN_GAME_URL,
                params={'event': game_id},
                timeout=10
            )
            return response.json()
        except Exception as e:
            logger.error(f"Ошибка получения деталей игры: {e}")
            return None


class LiveWinProbabilityCalculator:
    """
    Расчет вероятности победы в реальном времени
    """
    
    def calculate_live_prob(
        self,
        score: LiveScore,
        status: GameStatus,
        pre_game_home_prob: float = 0.5
    ) -> float:
        """
        Рассчитывает вероятность победы home team
        на основе текущего счета и времени
        """
        if status == GameStatus.FINAL:
            return 1.0 if score.margin > 0 else 0.0
        
        if status == GameStatus.SCHEDULED:
            return pre_game_home_prob
        
        margin = score.margin
        quarter = score.quarter
        
        # Оставшееся время (приблизительно)
        if quarter == 1:
            time_pct = 0.125  # 1/8 игры прошло
        elif quarter == 2:
            time_pct = 0.375  # 3/8
        elif quarter == 3:
            time_pct = 0.625  # 5/8
        elif quarter == 4:
            time_pct = 0.875  # 7/8
        else:
            time_pct = 0.95  # OT
        
        # Базовая модель: margin влияет на вероятность
        # Каждые 4 очка преимущества ≈ 10% вероятности
        # Но влияние растет к концу игры
        
        time_factor = 0.5 + time_pct * 0.5  # От 0.5 до 1.0
        margin_impact = margin / 40  # -1 to +1 примерно
        
        # Комбинируем pre-game prob с текущим состоянием
        adjusted_prob = pre_game_home_prob * (1 - time_pct) + \
                       (0.5 + margin_impact * time_factor) * time_pct
        
        # Clamp
        return max(0.01, min(0.99, adjusted_prob))
    
    def calculate_projected_total(
        self,
        score: LiveScore,
        status: GameStatus,
        pre_game_total: float = 220
    ) -> float:
        """Прогнозирует итоговый тотал"""
        if status == GameStatus.FINAL:
            return float(score.total)
        
        if status == GameStatus.SCHEDULED:
            return pre_game_total
        
        current_total = score.total
        quarter = score.quarter
        
        # Оставшееся время
        if quarter == 1:
            elapsed_pct = 0.25
        elif quarter == 2:
            elapsed_pct = 0.50
        elif quarter == 3:
            elapsed_pct = 0.75
        elif quarter == 4:
            elapsed_pct = 1.0
        else:
            elapsed_pct = 1.0
        
        if elapsed_pct < 0.1:
            return pre_game_total
        
        # Проецируем текущий темп
        projected = current_total / elapsed_pct
        
        # Смешиваем с pre-game
        weight = elapsed_pct
        return pre_game_total * (1 - weight) + projected * weight


class LiveGameMonitor:
    """
    Основной монитор живых игр с детекцией аномалий
    
    Особенности:
    - Отслеживание счёта в реальном времени
    - Расчёт live win probability
    - Детекция аномалий (underdog leading, momentum shift, etc.)
    - Оповещения о возможностях для live-ставок
    """
    
    def __init__(
        self,
        update_interval: int = 30,
        alert_callback: Callable[[LiveAlert], None] = None,
        anomaly_callback: Callable[[LiveAnomaly], None] = None
    ):
        self.provider = LiveScoreProvider()
        self.calculator = LiveWinProbabilityCalculator()
        self.anomaly_detector = AnomalyDetector()  # Детектор аномалий
        
        self.update_interval = update_interval
        self.alert_callback = alert_callback
        self.anomaly_callback = anomaly_callback  # Callback для аномалий
        
        # Отслеживаемые игры
        self.games: Dict[str, LiveGameState] = {}
        
        # Pre-game predictions для сравнения
        self.pre_game_predictions: Dict[str, Dict] = {}
        
        # Наши ставки для отслеживания
        self.tracked_bets: Dict[str, Dict] = {}
        
        # Обнаруженные аномалии
        self.anomalies: List[LiveAnomaly] = []
        
        # Фоновый поток
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def add_prediction(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        predicted_home_prob: float,
        predicted_margin: float = 0,
        predicted_total: float = 220,
        pre_game_odds: Dict = None
    ):
        """Добавляет наш прогноз для отслеживания"""
        self.games[game_id] = LiveGameState(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            status=GameStatus.SCHEDULED,
            predicted_home_prob=predicted_home_prob,
            predicted_margin=predicted_margin,
            predicted_total=predicted_total
        )
        
        # Сохраняем pre-game prediction для детекции аномалий
        self.pre_game_predictions[game_id] = {
            'home_prob': predicted_home_prob,
            'margin': predicted_margin,
            'total': predicted_total,
            'odds': pre_game_odds or {}
        }
    
    def add_bet(
        self,
        game_id: str,
        side: str,  # "home", "away", "over", "under"
        amount: float,
        line: float = 0  # spread или total line
    ):
        """Добавляет ставку для отслеживания"""
        if game_id in self.games:
            self.games[game_id].bet_placed = True
            self.games[game_id].bet_side = side
            self.games[game_id].bet_amount = amount
        
        self.tracked_bets[game_id] = {
            'side': side,
            'amount': amount,
            'line': line,
            'status': 'active'
        }
    
    def update(self) -> List[LiveGameState]:
        """Обновляет все игры и детектирует аномалии"""
        live_games = self.provider.get_live_games()
        
        for live in live_games:
            game_id = live.game_id
            
            # Если игра уже отслеживается
            if game_id in self.games:
                game = self.games[game_id]
                old_score = game.score
                
                # Обновляем данные
                game.score = live.score
                game.status = live.status
                game.last_update = datetime.now()
                
                # Рассчитываем live вероятность
                game.live_home_win_prob = self.calculator.calculate_live_prob(
                    live.score,
                    live.status,
                    game.predicted_home_prob
                )
                
                # Сравниваем с прогнозом
                self._check_prediction_accuracy(game)
                
                # Сохраняем историю
                game.score_history.append({
                    'time': datetime.now().isoformat(),
                    'home_score': live.score.home_score,
                    'away_score': live.score.away_score,
                    'quarter': live.score.quarter,
                    'live_prob': game.live_home_win_prob
                })
                
                # Проверяем алерты
                self._check_alerts(game, old_score)
                
                # === ДЕТЕКЦИЯ АНОМАЛИЙ ===
                pre_game = self.pre_game_predictions.get(game_id, {})
                anomalies = self.anomaly_detector.detect_anomalies(
                    game,
                    pre_game_odds=pre_game
                )
                
                for anomaly in anomalies:
                    self.anomalies.append(anomaly)
                    
                    # Логируем аномалию
                    logger.info(f"\n{'='*50}")
                    logger.info(f"🚨 ANOMALY DETECTED: {anomaly.anomaly_type.value}")
                    logger.info(f"   {anomaly.description}")
                    
                    if anomaly.bet_opportunity:
                        logger.info(f"   🎯 BET OPPORTUNITY: {anomaly.recommended_side.upper()}")
                        logger.info(f"   Edge estimate: {anomaly.edge_estimate:.1%}")
                    
                    logger.info(f"{'='*50}\n")
                    
                    # Callback если есть
                    if self.anomaly_callback:
                        self.anomaly_callback(anomaly)
                
            else:
                # Новая игра - добавляем с дефолтными прогнозами
                self.games[game_id] = live
        
        return list(self.games.values())
    
    def _check_prediction_accuracy(self, game: LiveGameState):
        """Проверяет насколько игра идет по прогнозу"""
        if game.status in [GameStatus.SCHEDULED, GameStatus.FINAL]:
            return
        
        # Отклонение по margin
        expected_margin_now = game.predicted_margin * (game.score.quarter / 4)
        actual_margin = game.score.margin
        
        margin_deviation = actual_margin - expected_margin_now
        
        # Отклонение по тоталу
        projected_total = self.calculator.calculate_projected_total(
            game.score, game.status, game.predicted_total
        )
        total_deviation = projected_total - game.predicted_total
        
        # Считаем общее отклонение
        game.deviation_from_prediction = abs(margin_deviation) + abs(total_deviation) / 10
        
        # Определяем "on track" если отклонение < 10 очков
        game.is_on_track = game.deviation_from_prediction < 10
    
    def _check_alerts(self, game: LiveGameState, old_score: LiveScore):
        """Проверяет и генерирует алерты"""
        alerts = []
        
        # Momentum shift (большой run)
        score_change_home = game.score.home_score - old_score.home_score
        score_change_away = game.score.away_score - old_score.away_score
        
        if abs(score_change_home - score_change_away) >= 10:
            direction = "Home" if score_change_home > score_change_away else "Away"
            alerts.append(LiveAlert(
                timestamp=datetime.now(),
                game_id=game.game_id,
                alert_type="momentum_shift",
                message=f"🔥 {direction} run! {game.home_team} {game.score.home_score}-{game.score.away_score} {game.away_team}",
                severity="warning",
                data={'run': abs(score_change_home - score_change_away)}
            ))
        
        # Blowout (разрыв > 20)
        if abs(game.score.margin) > 20 and game.score.quarter >= 3:
            leader = game.home_team if game.score.margin > 0 else game.away_team
            alerts.append(LiveAlert(
                timestamp=datetime.now(),
                game_id=game.game_id,
                alert_type="blowout",
                message=f"💨 Blowout: {leader} leading by {abs(game.score.margin)}",
                severity="info"
            ))
        
        # Close game в 4-й четверти
        if game.status == GameStatus.LIVE_Q4 and abs(game.score.margin) <= 5:
            alerts.append(LiveAlert(
                timestamp=datetime.now(),
                game_id=game.game_id,
                alert_type="close_game",
                message=f"🔥 Close game Q4! {game.home_team} {game.score.home_score}-{game.score.away_score} {game.away_team}",
                severity="warning"
            ))
        
        # Bet at risk
        if game.bet_placed:
            bet_info = self.tracked_bets.get(game.game_id, {})
            
            if bet_info.get('side') == 'home' and game.live_home_win_prob < 0.3:
                alerts.append(LiveAlert(
                    timestamp=datetime.now(),
                    game_id=game.game_id,
                    alert_type="bet_at_risk",
                    message=f"⚠️ Home bet at risk! Win prob: {game.live_home_win_prob:.0%}",
                    severity="critical",
                    data={'bet_amount': bet_info.get('amount', 0)}
                ))
            elif bet_info.get('side') == 'away' and game.live_home_win_prob > 0.7:
                alerts.append(LiveAlert(
                    timestamp=datetime.now(),
                    game_id=game.game_id,
                    alert_type="bet_at_risk",
                    message=f"⚠️ Away bet at risk! Win prob: {1-game.live_home_win_prob:.0%}",
                    severity="critical",
                    data={'bet_amount': bet_info.get('amount', 0)}
                ))
        
        # Отправляем алерты
        for alert in alerts:
            if self.alert_callback:
                self.alert_callback(alert)
            else:
                logger.info(f"ALERT: {alert.message}")
    
    def start_monitoring(self):
        """Запускает фоновый мониторинг"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Live monitoring started")
    
    def stop_monitoring(self):
        """Останавливает мониторинг"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Live monitoring stopped")
    
    def _monitoring_loop(self):
        """Цикл мониторинга"""
        while self._running:
            try:
                self.update()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(self.update_interval)
    
    def get_game_status(self, game_id: str) -> Optional[LiveGameState]:
        """Получает статус конкретной игры"""
        return self.games.get(game_id)
    
    def get_live_summary(self) -> str:
        """Возвращает текстовый summary всех игр"""
        lines = ["=" * 50, "🏀 LIVE GAMES", "=" * 50, ""]
        
        for game in self.games.values():
            status_emoji = {
                GameStatus.SCHEDULED: "⏰",
                GameStatus.LIVE_Q1: "1️⃣",
                GameStatus.LIVE_Q2: "2️⃣",
                GameStatus.HALFTIME: "🔄",
                GameStatus.LIVE_Q3: "3️⃣",
                GameStatus.LIVE_Q4: "4️⃣",
                GameStatus.OVERTIME: "⏱️",
                GameStatus.FINAL: "✅"
            }.get(game.status, "❓")
            
            line = f"{status_emoji} {game.away_team} @ {game.home_team}"
            
            if game.status not in [GameStatus.SCHEDULED]:
                line += f": {game.score.away_score}-{game.score.home_score}"
                
                if game.status != GameStatus.FINAL:
                    line += f" (Q{game.score.quarter} {game.score.time_remaining})"
                    line += f" | Live prob: {game.live_home_win_prob:.0%}"
                    
                    if game.bet_placed:
                        on_track = "✅" if game.is_on_track else "⚠️"
                        line += f" | Bet: {game.bet_side} {on_track}"
            
            lines.append(line)
        
        return "\n".join(lines)
    
    def get_bet_status(self, game_id: str) -> Dict:
        """Получает статус ставки"""
        game = self.games.get(game_id)
        bet = self.tracked_bets.get(game_id)
        
        if not game or not bet:
            return {}
        
        # Рассчитываем текущее состояние ставки
        side = bet['side']
        line = bet['line']
        
        if side == 'home':
            current_cover = game.score.margin > line if line else game.score.margin > 0
            win_prob = game.live_home_win_prob
        elif side == 'away':
            current_cover = -game.score.margin > line if line else game.score.margin < 0
            win_prob = 1 - game.live_home_win_prob
        elif side == 'over':
            projected = self.calculator.calculate_projected_total(game.score, game.status, game.predicted_total)
            current_cover = projected > line
            win_prob = 0.5 + (projected - line) / 40  # rough estimate
        else:  # under
            projected = self.calculator.calculate_projected_total(game.score, game.status, game.predicted_total)
            current_cover = projected < line
            win_prob = 0.5 - (projected - line) / 40
        
        win_prob = max(0.05, min(0.95, win_prob))
        
        return {
            'game_id': game_id,
            'side': side,
            'amount': bet['amount'],
            'line': line,
            'current_status': 'winning' if current_cover else 'losing',
            'win_probability': win_prob,
            'score': f"{game.score.away_score}-{game.score.home_score}",
            'quarter': game.score.quarter,
            'is_on_track': game.is_on_track,
            'deviation': game.deviation_from_prediction
        }
    
    def get_live_bet_opportunities(self) -> List[LiveAnomaly]:
        """Возвращает текущие возможности для live-ставок"""
        opportunities = []
        
        for anomaly in self.anomalies:
            if anomaly.bet_opportunity:
                # Проверяем что игра ещё идёт
                game = self.games.get(anomaly.game_id)
                if game and game.status in [GameStatus.LIVE_Q1, GameStatus.LIVE_Q2, 
                                            GameStatus.LIVE_Q3, GameStatus.LIVE_Q4]:
                    opportunities.append(anomaly)
        
        return opportunities
    
    def get_recent_anomalies(self, minutes: int = 30) -> List[LiveAnomaly]:
        """Возвращает аномалии за последние N минут"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [a for a in self.anomalies if a.timestamp > cutoff]
    
    def get_anomaly_summary(self) -> str:
        """Возвращает summary всех аномалий"""
        if not self.anomalies:
            return "No anomalies detected"
        
        lines = [
            "=" * 50,
            "🚨 ANOMALY SUMMARY",
            "=" * 50,
            ""
        ]
        
        # Группируем по типу
        by_type = {}
        for a in self.anomalies:
            t = a.anomaly_type.value
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(a)
        
        for anomaly_type, anomalies in by_type.items():
            lines.append(f"\n📌 {anomaly_type.upper()}: {len(anomalies)}")
            
            for a in anomalies[-3:]:  # Последние 3
                lines.append(f"   • {a.home_team} vs {a.away_team}: {a.description[:50]}...")
                if a.bet_opportunity:
                    lines.append(f"     🎯 Bet on {a.recommended_side} (edge: {a.edge_estimate:.1%})")
        
        # Live opportunities
        opportunities = self.get_live_bet_opportunities()
        if opportunities:
            lines.append(f"\n🎯 ACTIVE BET OPPORTUNITIES: {len(opportunities)}")
            for opp in opportunities:
                lines.append(f"   • {opp.away_team} @ {opp.home_team}")
                lines.append(f"     Side: {opp.recommended_side.upper()}, Edge: {opp.edge_estimate:.1%}")
        
        return "\n".join(lines)


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Live Game Monitor ===\n")
    
    # Alert callback
    def on_alert(alert: LiveAlert):
        print(f"[{alert.severity.upper()}] {alert.message}")
    
    # Создаем монитор
    monitor = LiveGameMonitor(update_interval=30, alert_callback=on_alert)
    
    # Добавляем тестовые прогнозы
    monitor.add_prediction(
        game_id="401584701",
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        predicted_home_prob=0.58,
        predicted_margin=4.5,
        predicted_total=228.5
    )
    
    monitor.add_prediction(
        game_id="401584702",
        home_team="Boston Celtics",
        away_team="Miami Heat",
        predicted_home_prob=0.72,
        predicted_margin=8.0,
        predicted_total=215.0
    )
    
    # Добавляем ставку
    monitor.add_bet(
        game_id="401584701",
        side="home",
        amount=15.00,
        line=-3.5
    )
    
    print("Получаем live данные...\n")
    
    # Обновляем (получаем реальные данные)
    games = monitor.update()
    
    print(f"Найдено игр: {len(games)}\n")
    
    # Выводим summary
    print(monitor.get_live_summary())
    
    # Тест калькулятора вероятностей
    print("\n\n=== Тест Live Probability Calculator ===\n")
    
    calc = LiveWinProbabilityCalculator()
    
    # Симулируем разные ситуации
    scenarios = [
        ("Q1, tied", LiveScore(home_score=28, away_score=28, quarter=1), GameStatus.LIVE_Q1),
        ("Q2, home +10", LiveScore(home_score=58, away_score=48, quarter=2), GameStatus.LIVE_Q2),
        ("Q3, away +5", LiveScore(home_score=75, away_score=80, quarter=3), GameStatus.LIVE_Q3),
        ("Q4, home +15", LiveScore(home_score=105, away_score=90, quarter=4), GameStatus.LIVE_Q4),
        ("Q4, tied", LiveScore(home_score=98, away_score=98, quarter=4), GameStatus.LIVE_Q4),
    ]
    
    pre_game_prob = 0.55
    
    for name, score, status in scenarios:
        live_prob = calc.calculate_live_prob(score, status, pre_game_prob)
        projected_total = calc.calculate_projected_total(score, status, 220)
        
        print(f"{name}:")
        print(f"  Score: {score.home_score}-{score.away_score}")
        print(f"  Pre-game prob: {pre_game_prob:.0%}")
        print(f"  Live prob: {live_prob:.0%}")
        print(f"  Projected total: {projected_total:.1f}")
        print()
    
    # Тест статуса ставки
    print("=== Тест Bet Status ===\n")
    
    # Симулируем игру в процессе
    test_game = monitor.games.get("401584701")
    if test_game:
        # Имитируем счет
        test_game.score = LiveScore(
            home_score=72,
            away_score=68,
            quarter=3,
            time_remaining="5:30"
        )
        test_game.status = GameStatus.LIVE_Q3
        test_game.live_home_win_prob = calc.calculate_live_prob(
            test_game.score, test_game.status, test_game.predicted_home_prob
        )
        
        bet_status = monitor.get_bet_status("401584701")
        
        print(f"Bet on: {bet_status.get('side')}")
        print(f"Amount: ${bet_status.get('amount')}")
        print(f"Current status: {bet_status.get('current_status')}")
        print(f"Win probability: {bet_status.get('win_probability'):.0%}")
        print(f"Score: {bet_status.get('score')}")
        print(f"On track: {bet_status.get('is_on_track')}")
    
    print("\n✅ Тест завершен")
    print("\nДля continuous мониторинга вызовите:")
    print("  monitor.start_monitoring()")
