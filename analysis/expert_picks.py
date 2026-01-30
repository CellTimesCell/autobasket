"""
AutoBasket - Expert Picks Tracker
=================================
Автоматический сбор и анализ прогнозов спортивных экспертов.

Источники:
- Covers.com - публичный консенсус ставок
- ESPN - expert picks
- Action Network - sharp money
- Odds API - line movements

Работает автоматически 24/7:
- Утром собирает прогнозы на сегодня
- Вечером обновляет результаты
- Отслеживает track record каждого эксперта
"""

import os
import re
import sqlite3
import logging
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urljoin

try:
    import requests
    from bs4 import BeautifulSoup
    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Типы экспертов"""
    FORMER_PLAYER = "former_player"      # Бывший игрок NBA
    ANALYST = "analyst"                  # Спортивный аналитик
    JOURNALIST = "journalist"            # Журналист
    HANDICAPPER = "handicapper"          # Профессиональный каппер
    BLOGGER = "blogger"                  # Спортивный блогер
    INSIDER = "insider"                  # Инсайдер (знает закулисье)
    AI_MODEL = "ai_model"                # Другая AI модель
    CONSENSUS = "consensus"              # Консенсус публики


class PickConfidence(Enum):
    """Уверенность в прогнозе"""
    LOW = 1        # "может быть", "возможно"
    MEDIUM = 2     # "думаю", "скорее всего"
    HIGH = 3       # "уверен", "точно"
    LOCK = 4       # "100%", "гарантия", "lock of the day"


@dataclass
class ExpertPick:
    """Прогноз эксперта"""
    expert_id: int
    expert_name: str
    game_id: str
    game_date: str
    
    # Прогноз
    picked_team: str           # Команда которую выбрал
    pick_type: str             # "moneyline", "spread", "total"
    pick_value: str            # Например "-3.5" для spread
    confidence: PickConfidence
    
    # Объяснение
    reasoning: str             # Почему выбрал эту команду
    key_factors: List[str] = field(default_factory=list)  # Ключевые факторы
    
    # Источник
    source_url: str = ""
    source_platform: str = ""  # "twitter", "youtube", "espn", etc
    
    # Результат (заполняется после игры)
    result: Optional[str] = None  # "won", "lost", "push"
    actual_score: str = ""
    
    # Мета
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class ExpertProfile:
    """Профиль эксперта с track record"""
    expert_id: int
    name: str
    expert_type: ExpertType
    
    # Биография
    description: str = ""
    credentials: str = ""      # "Former NBA player", "20 years analyst"
    platforms: List[str] = field(default_factory=list)  # twitter, youtube, etc
    
    # Track Record
    total_picks: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    
    @property
    def win_rate(self) -> float:
        """Процент попаданий"""
        decided = self.wins + self.losses
        return self.wins / decided if decided > 0 else 0.0
    
    @property
    def roi(self) -> float:
        """ROI при flat betting"""
        # Assuming -110 odds (1.91)
        if self.total_picks == 0:
            return 0.0
        profit = (self.wins * 0.91) - self.losses
        return profit / self.total_picks
    
    # Специализация (на чём хорош)
    best_teams: List[str] = field(default_factory=list)
    worst_teams: List[str] = field(default_factory=list)
    best_bet_types: List[str] = field(default_factory=list)
    
    # Статус
    is_sharp: bool = False     # 55%+ win rate с достаточным sample size
    is_trusted: bool = False   # Мы доверяем этому эксперту
    
    last_updated: str = ""


class ExpertPicksTracker:
    """
    Система отслеживания экспертных прогнозов
    
    Использование:
        tracker = ExpertPicksTracker()
        
        # Добавляем эксперта
        tracker.add_expert(
            name="Kenny Smith",
            expert_type=ExpertType.FORMER_PLAYER,
            credentials="2x NBA Champion, TNT Analyst"
        )
        
        # Записываем его прогноз
        tracker.record_pick(
            expert_name="Kenny Smith",
            game_id="0022400123",
            picked_team="Golden State Warriors",
            confidence=PickConfidence.HIGH,
            reasoning="Curry в отличной форме последние 5 игр"
        )
        
        # После игры обновляем результат
        tracker.update_pick_result(pick_id, "won")
        
        # Получаем консенсус экспертов на игру
        consensus = tracker.get_expert_consensus("0022400123")
    """
    
    def __init__(self, db_path: str = "expert_picks.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        self._seed_known_experts()
    
    def _create_tables(self):
        """Создает таблицы"""
        cursor = self.conn.cursor()
        
        # Эксперты
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experts (
                expert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                expert_type TEXT,
                description TEXT,
                credentials TEXT,
                platforms TEXT,
                total_picks INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                pushes INTEGER DEFAULT 0,
                best_teams TEXT,
                worst_teams TEXT,
                best_bet_types TEXT,
                is_sharp INTEGER DEFAULT 0,
                is_trusted INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)
        
        # Прогнозы
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS picks (
                pick_id INTEGER PRIMARY KEY AUTOINCREMENT,
                expert_id INTEGER,
                expert_name TEXT,
                game_id TEXT,
                game_date TEXT,
                picked_team TEXT,
                pick_type TEXT,
                pick_value TEXT,
                confidence INTEGER,
                reasoning TEXT,
                key_factors TEXT,
                source_url TEXT,
                source_platform TEXT,
                result TEXT,
                actual_score TEXT,
                timestamp TEXT,
                FOREIGN KEY (expert_id) REFERENCES experts(expert_id)
            )
        """)
        
        # Индексы
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_picks_game ON picks(game_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_picks_expert ON picks(expert_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_picks_date ON picks(game_date)")
        
        self.conn.commit()
    
    def _seed_known_experts(self):
        """Добавляет известных экспертов"""
        known_experts = [
            # Бывшие игроки / аналитики TNT/ESPN
            ("Kenny Smith", ExpertType.FORMER_PLAYER, "2x NBA Champion, Inside the NBA analyst"),
            ("Charles Barkley", ExpertType.FORMER_PLAYER, "Hall of Famer, Inside the NBA"),
            ("Shaquille O'Neal", ExpertType.FORMER_PLAYER, "4x NBA Champion, Inside the NBA"),
            ("Stephen A. Smith", ExpertType.ANALYST, "ESPN First Take host"),
            ("Kendrick Perkins", ExpertType.FORMER_PLAYER, "NBA Champion, ESPN analyst"),
            ("JJ Redick", ExpertType.FORMER_PLAYER, "15-year NBA veteran, podcaster"),
            
            # Профессиональные капперы
            ("Action Network Consensus", ExpertType.CONSENSUS, "Professional betting platform consensus"),
            ("Covers Consensus", ExpertType.CONSENSUS, "Public betting percentages"),
            
            # Журналисты / инсайдеры
            ("Adrian Wojnarowski", ExpertType.INSIDER, "ESPN Senior NBA Insider"),
            ("Shams Charania", ExpertType.INSIDER, "The Athletic NBA Insider"),
            ("Zach Lowe", ExpertType.JOURNALIST, "ESPN Senior Writer, analytics expert"),
            
            # AI модели (для сравнения)
            ("ESPN BPI", ExpertType.AI_MODEL, "ESPN Basketball Power Index model"),
            ("FiveThirtyEight", ExpertType.AI_MODEL, "Nate Silver's prediction model"),
        ]
        
        for name, expert_type, credentials in known_experts:
            self.add_expert(name, expert_type, credentials, update_if_exists=False)
    
    # =========================================================================
    # УПРАВЛЕНИЕ ЭКСПЕРТАМИ
    # =========================================================================
    
    def add_expert(
        self,
        name: str,
        expert_type: ExpertType,
        credentials: str = "",
        description: str = "",
        platforms: List[str] = None,
        update_if_exists: bool = True
    ) -> int:
        """Добавляет эксперта"""
        cursor = self.conn.cursor()
        
        # Проверяем существует ли
        cursor.execute("SELECT expert_id FROM experts WHERE name = ?", (name,))
        existing = cursor.fetchone()
        
        if existing:
            if update_if_exists:
                cursor.execute("""
                    UPDATE experts SET 
                        expert_type = ?, credentials = ?, description = ?,
                        platforms = ?, last_updated = ?
                    WHERE name = ?
                """, (
                    expert_type.value, credentials, description,
                    json.dumps(platforms or []), datetime.now().isoformat(),
                    name
                ))
                self.conn.commit()
            return existing['expert_id']
        
        cursor.execute("""
            INSERT INTO experts (name, expert_type, credentials, description, platforms, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name, expert_type.value, credentials, description,
            json.dumps(platforms or []), datetime.now().isoformat()
        ))
        self.conn.commit()
        
        return cursor.lastrowid
    
    def get_expert(self, name: str) -> Optional[ExpertProfile]:
        """Получает профиль эксперта"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experts WHERE name = ?", (name,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return ExpertProfile(
            expert_id=row['expert_id'],
            name=row['name'],
            expert_type=ExpertType(row['expert_type']),
            description=row['description'] or "",
            credentials=row['credentials'] or "",
            platforms=json.loads(row['platforms'] or "[]"),
            total_picks=row['total_picks'],
            wins=row['wins'],
            losses=row['losses'],
            pushes=row['pushes'],
            best_teams=json.loads(row['best_teams'] or "[]"),
            worst_teams=json.loads(row['worst_teams'] or "[]"),
            best_bet_types=json.loads(row['best_bet_types'] or "[]"),
            is_sharp=bool(row['is_sharp']),
            is_trusted=bool(row['is_trusted']),
            last_updated=row['last_updated'] or ""
        )
    
    def get_all_experts(self, only_sharp: bool = False) -> List[ExpertProfile]:
        """Получает всех экспертов"""
        cursor = self.conn.cursor()
        
        if only_sharp:
            cursor.execute("SELECT * FROM experts WHERE is_sharp = 1 ORDER BY wins DESC")
        else:
            cursor.execute("SELECT * FROM experts ORDER BY total_picks DESC")
        
        experts = []
        for row in cursor.fetchall():
            experts.append(ExpertProfile(
                expert_id=row['expert_id'],
                name=row['name'],
                expert_type=ExpertType(row['expert_type']),
                total_picks=row['total_picks'],
                wins=row['wins'],
                losses=row['losses'],
                pushes=row['pushes'],
                is_sharp=bool(row['is_sharp']),
                is_trusted=bool(row['is_trusted'])
            ))
        
        return experts
    
    # =========================================================================
    # ЗАПИСЬ ПРОГНОЗОВ
    # =========================================================================
    
    def record_pick(
        self,
        expert_name: str,
        game_id: str,
        picked_team: str,
        pick_type: str = "moneyline",
        pick_value: str = "",
        confidence: PickConfidence = PickConfidence.MEDIUM,
        reasoning: str = "",
        key_factors: List[str] = None,
        source_url: str = "",
        source_platform: str = ""
    ) -> int:
        """Записывает прогноз эксперта"""
        cursor = self.conn.cursor()
        
        # Получаем expert_id
        cursor.execute("SELECT expert_id FROM experts WHERE name = ?", (expert_name,))
        row = cursor.fetchone()
        
        if not row:
            # Создаем эксперта если не существует
            expert_id = self.add_expert(expert_name, ExpertType.BLOGGER)
        else:
            expert_id = row['expert_id']
        
        # Записываем прогноз
        cursor.execute("""
            INSERT INTO picks (
                expert_id, expert_name, game_id, game_date,
                picked_team, pick_type, pick_value, confidence,
                reasoning, key_factors, source_url, source_platform, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            expert_id, expert_name, game_id, date.today().isoformat(),
            picked_team, pick_type, pick_value, confidence.value,
            reasoning, json.dumps(key_factors or []),
            source_url, source_platform, datetime.now().isoformat()
        ))
        
        # Обновляем счетчик
        cursor.execute(
            "UPDATE experts SET total_picks = total_picks + 1 WHERE expert_id = ?",
            (expert_id,)
        )
        
        self.conn.commit()
        
        logger.info(f"📝 Recorded pick: {expert_name} → {picked_team}")
        
        return cursor.lastrowid
    
    def update_pick_result(self, pick_id: int, result: str, actual_score: str = ""):
        """Обновляет результат прогноза"""
        cursor = self.conn.cursor()
        
        # Получаем прогноз
        cursor.execute("SELECT expert_id, result FROM picks WHERE pick_id = ?", (pick_id,))
        row = cursor.fetchone()
        
        if not row:
            return
        
        if row['result']:  # Уже обновлен
            return
        
        expert_id = row['expert_id']
        
        # Обновляем прогноз
        cursor.execute("""
            UPDATE picks SET result = ?, actual_score = ? WHERE pick_id = ?
        """, (result, actual_score, pick_id))
        
        # Обновляем статистику эксперта
        if result == "won":
            cursor.execute("UPDATE experts SET wins = wins + 1 WHERE expert_id = ?", (expert_id,))
        elif result == "lost":
            cursor.execute("UPDATE experts SET losses = losses + 1 WHERE expert_id = ?", (expert_id,))
        elif result == "push":
            cursor.execute("UPDATE experts SET pushes = pushes + 1 WHERE expert_id = ?", (expert_id,))
        
        # Проверяем sharp статус
        self._update_sharp_status(expert_id)
        
        self.conn.commit()
    
    def _update_sharp_status(self, expert_id: int):
        """Обновляет sharp статус эксперта"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT wins, losses FROM experts WHERE expert_id = ?", (expert_id,))
        row = cursor.fetchone()
        
        if not row:
            return
        
        wins = row['wins']
        losses = row['losses']
        total = wins + losses
        
        # Sharp = 55%+ с минимум 50 прогнозами
        is_sharp = (total >= 50) and (wins / total >= 0.55) if total > 0 else False
        
        cursor.execute("UPDATE experts SET is_sharp = ? WHERE expert_id = ?", (int(is_sharp), expert_id))
    
    # =========================================================================
    # АНАЛИЗ И КОНСЕНСУС
    # =========================================================================
    
    def get_game_picks(self, game_id: str) -> List[ExpertPick]:
        """Получает все прогнозы на игру"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT p.*, e.is_sharp, e.wins, e.losses
            FROM picks p
            JOIN experts e ON p.expert_id = e.expert_id
            WHERE p.game_id = ?
            ORDER BY e.is_sharp DESC, e.wins DESC
        """, (game_id,))
        
        picks = []
        for row in cursor.fetchall():
            picks.append(ExpertPick(
                expert_id=row['expert_id'],
                expert_name=row['expert_name'],
                game_id=row['game_id'],
                game_date=row['game_date'],
                picked_team=row['picked_team'],
                pick_type=row['pick_type'],
                pick_value=row['pick_value'] or "",
                confidence=PickConfidence(row['confidence']),
                reasoning=row['reasoning'] or "",
                key_factors=json.loads(row['key_factors'] or "[]"),
                source_url=row['source_url'] or "",
                source_platform=row['source_platform'] or "",
                result=row['result']
            ))
        
        return picks
    
    def get_expert_consensus(self, game_id: str, home_team: str, away_team: str) -> Dict:
        """
        Получает консенсус экспертов на игру
        
        Returns:
            {
                'total_picks': 10,
                'home_picks': 6,
                'away_picks': 4,
                'home_pct': 0.6,
                'sharp_picks': {...},
                'confidence_weighted': {...},
                'key_reasons_home': [...],
                'key_reasons_away': [...],
                'recommendation': "home" or "away" or "no_consensus"
            }
        """
        picks = self.get_game_picks(game_id)
        
        if not picks:
            return {
                'total_picks': 0,
                'recommendation': 'no_data'
            }
        
        home_picks = []
        away_picks = []
        
        for pick in picks:
            if pick.picked_team == home_team:
                home_picks.append(pick)
            elif pick.picked_team == away_team:
                away_picks.append(pick)
        
        total = len(home_picks) + len(away_picks)
        
        if total == 0:
            return {'total_picks': 0, 'recommendation': 'no_data'}
        
        # Считаем weighted score (sharp эксперты весят больше)
        home_score = sum(
            (2 if self._is_sharp_expert(p.expert_id) else 1) * p.confidence.value
            for p in home_picks
        )
        away_score = sum(
            (2 if self._is_sharp_expert(p.expert_id) else 1) * p.confidence.value
            for p in away_picks
        )
        
        total_score = home_score + away_score
        
        # Sharp picks отдельно
        sharp_home = [p for p in home_picks if self._is_sharp_expert(p.expert_id)]
        sharp_away = [p for p in away_picks if self._is_sharp_expert(p.expert_id)]
        
        # Причины
        home_reasons = []
        for p in home_picks[:3]:
            if p.reasoning:
                home_reasons.append(f"{p.expert_name}: {p.reasoning[:100]}")
        
        away_reasons = []
        for p in away_picks[:3]:
            if p.reasoning:
                away_reasons.append(f"{p.expert_name}: {p.reasoning[:100]}")
        
        # Рекомендация
        if total < 3:
            recommendation = 'insufficient_data'
        elif home_score > away_score * 1.5:
            recommendation = 'home'
        elif away_score > home_score * 1.5:
            recommendation = 'away'
        else:
            recommendation = 'no_consensus'
        
        return {
            'total_picks': total,
            'home_team': home_team,
            'away_team': away_team,
            'home_picks': len(home_picks),
            'away_picks': len(away_picks),
            'home_pct': len(home_picks) / total,
            'away_pct': len(away_picks) / total,
            'sharp_picks': {
                'home': len(sharp_home),
                'away': len(sharp_away)
            },
            'weighted_score': {
                'home': home_score / total_score if total_score > 0 else 0.5,
                'away': away_score / total_score if total_score > 0 else 0.5
            },
            'key_reasons_home': home_reasons,
            'key_reasons_away': away_reasons,
            'recommendation': recommendation
        }
    
    def _is_sharp_expert(self, expert_id: int) -> bool:
        """Проверяет sharp ли эксперт"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT is_sharp FROM experts WHERE expert_id = ?", (expert_id,))
        row = cursor.fetchone()
        return bool(row['is_sharp']) if row else False
    
    # =========================================================================
    # LEADERBOARD
    # =========================================================================
    
    def get_leaderboard(self, min_picks: int = 20) -> List[Dict]:
        """Получает рейтинг экспертов"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT *,
                   CASE WHEN (wins + losses) > 0 
                        THEN CAST(wins AS REAL) / (wins + losses) 
                        ELSE 0 END as win_rate
            FROM experts
            WHERE total_picks >= ?
            ORDER BY win_rate DESC, wins DESC
        """, (min_picks,))
        
        leaderboard = []
        for rank, row in enumerate(cursor.fetchall(), 1):
            wins = row['wins']
            losses = row['losses']
            total = wins + losses
            
            # ROI calculation (assuming -110 odds)
            roi = ((wins * 0.91) - losses) / total if total > 0 else 0
            
            leaderboard.append({
                'rank': rank,
                'name': row['name'],
                'type': row['expert_type'],
                'picks': row['total_picks'],
                'record': f"{wins}-{losses}",
                'win_rate': f"{row['win_rate']:.1%}",
                'roi': f"{roi:+.1%}",
                'is_sharp': bool(row['is_sharp']),
                'is_trusted': bool(row['is_trusted'])
            })
        
        return leaderboard
    
    # =========================================================================
    # ИНТЕГРАЦИЯ С CLAUDE
    # =========================================================================
    
    def format_for_claude(self, game_id: str, home_team: str, away_team: str) -> str:
        """
        Форматирует данные экспертов для Claude анализа
        """
        consensus = self.get_expert_consensus(game_id, home_team, away_team)
        
        if consensus['total_picks'] == 0:
            return "Нет данных от экспертов по этой игре."
        
        text = f"""
## 📊 МНЕНИЯ ЭКСПЕРТОВ: {away_team} @ {home_team}

**Всего прогнозов:** {consensus['total_picks']}
- За {home_team} (дома): {consensus['home_picks']} ({consensus['home_pct']:.0%})
- За {away_team} (гости): {consensus['away_picks']} ({consensus['away_pct']:.0%})

**Sharp эксперты (55%+ win rate):**
- За {home_team}: {consensus['sharp_picks']['home']}
- За {away_team}: {consensus['sharp_picks']['away']}

**Взвешенный консенсус:** {consensus['weighted_score']['home']:.0%} за {home_team}

**Ключевые аргументы за {home_team}:**
"""
        for reason in consensus['key_reasons_home']:
            text += f"- {reason}\n"
        
        text += f"\n**Ключевые аргументы за {away_team}:**\n"
        for reason in consensus['key_reasons_away']:
            text += f"- {reason}\n"
        
        text += f"\n**Рекомендация экспертов:** {consensus['recommendation'].upper()}"
        
        return text


# =============================================================================
# АВТОМАТИЧЕСКИЕ WEB SCRAPERS
# =============================================================================

class AutoScraper:
    """
    Автоматический сборщик прогнозов из разных источников
    
    Работает без участия пользователя:
    - Утром собирает прогнозы на игры дня
    - Вечером/ночью обновляет результаты
    - Ведёт историю всех прогнозов
    """
    
    # User-Agent для запросов
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    def __init__(self, tracker: ExpertPicksTracker):
        self.tracker = tracker
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        
        # Rate limiting
        self._last_request = 0
        self.min_delay = 2.0  # Минимум 2 секунды между запросами
        
        # Лог сбора
        self.collection_log: List[Dict] = []
    
    def _rate_limit(self):
        """Соблюдаем паузу между запросами"""
        elapsed = time.time() - self._last_request
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)
        self._last_request = time.time()
    
    def _safe_request(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Безопасный запрос с retry"""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=15)
                response.raise_for_status()
                return response.text
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    # =========================================================================
    # COVERS.COM - Public Consensus
    # =========================================================================
    
    def scrape_covers_consensus(self) -> List[Dict]:
        """
        Собирает публичный консенсус с Covers.com
        
        Показывает % ставок на каждую сторону от обычных бетторов.
        """
        if not SCRAPING_AVAILABLE:
            logger.warning("BeautifulSoup not installed. Run: pip install beautifulsoup4")
            return []
        
        logger.info("📊 Scraping Covers.com consensus...")
        
        # Актуальный URL (может меняться)
        urls_to_try = [
            "https://www.covers.com/sports/nba/matchups",
            "https://www.covers.com/sport/basketball/nba/odds",
            "https://www.covers.com/sports/nba/odds"
        ]
        
        html = None
        for url in urls_to_try:
            html = self._safe_request(url)
            if html:
                logger.info(f"   Found working URL: {url}")
                break
        
        if not html:
            logger.warning("Could not fetch Covers.com - all URLs failed")
            return []
        
        picks = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Ищем карточки матчей (структура может меняться)
            game_cards = soup.find_all(['div', 'article'], class_=lambda x: x and ('game' in x.lower() or 'matchup' in x.lower() or 'event' in x.lower()))
            
            if not game_cards:
                # Альтернативный поиск по таблицам
                game_cards = soup.find_all('tr', {'data-game-id': True})
            
            if not game_cards:
                # Ещё одна попытка - ищем любые элементы с командами
                game_cards = soup.find_all(['div', 'li'], class_=lambda x: x and 'team' in str(x).lower())
            
            logger.info(f"   Found {len(game_cards)} potential game elements")
            
            for card in game_cards[:20]:  # Лимит на случай мусора
                try:
                    # Ищем названия команд
                    team_elements = card.find_all(['span', 'div', 'a'], class_=lambda x: x and 'team' in str(x).lower())
                    
                    if len(team_elements) >= 2:
                        away_team = team_elements[0].get_text(strip=True)
                        home_team = team_elements[1].get_text(strip=True)
                        
                        # Ищем проценты
                        pct_elements = card.find_all(['span', 'div'], class_=lambda x: x and ('pct' in str(x).lower() or 'percent' in str(x).lower()))
                        
                        if pct_elements:
                            pct_text = pct_elements[0].get_text()
                            pct = self._parse_percentage(pct_text)
                            
                            if pct > 50:
                                picked_team = home_team
                            else:
                                picked_team = away_team
                                pct = 100 - pct
                            
                            pick_data = {
                                'source': 'Covers Consensus',
                                'home_team': home_team,
                                'away_team': away_team,
                                'picked_team': picked_team,
                                'consensus_pct': pct,
                                'confidence': PickConfidence.HIGH if pct > 70 else PickConfidence.MEDIUM
                            }
                            
                            picks.append(pick_data)
                            
                            self.tracker.record_pick(
                                expert_name="Covers Consensus",
                                game_id=f"{date.today().isoformat()}_{away_team}_{home_team}",
                                picked_team=picked_team,
                                confidence=pick_data['confidence'],
                                reasoning=f"Public consensus: {pct:.0f}% on {picked_team}",
                                source_platform="covers.com"
                            )
                
                except Exception as e:
                    continue
            
            logger.info(f"   ✅ Collected {len(picks)} consensus picks from Covers")
            
        except Exception as e:
            logger.error(f"Error parsing Covers.com: {e}")
        
        return picks
    
    def _parse_percentage(self, text: str) -> float:
        """Парсит процент из текста"""
        try:
            numbers = re.findall(r'[\d.]+', text)
            if numbers:
                return float(numbers[0])
        except:
            pass
        return 50.0
    
    # =========================================================================
    # ESPN - Expert Picks (через scoreboard API)
    # =========================================================================
    
    def scrape_espn_picks(self) -> List[Dict]:
        """
        Собирает данные из ESPN API (более надёжно чем scraping)
        """
        logger.info("📺 Fetching ESPN data...")
        
        picks = []
        
        try:
            # ESPN имеет публичный JSON API
            url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
            
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"ESPN API returned {response.status_code}")
                return []
            
            data = response.json()
            events = data.get('events', [])
            
            for event in events:
                try:
                    competitions = event.get('competitions', [{}])
                    if not competitions:
                        continue
                    
                    comp = competitions[0]
                    competitors = comp.get('competitors', [])
                    
                    if len(competitors) < 2:
                        continue
                    
                    # ESPN: competitors[0] = home, competitors[1] = away
                    home_data = competitors[0] if competitors[0].get('homeAway') == 'home' else competitors[1]
                    away_data = competitors[1] if competitors[0].get('homeAway') == 'home' else competitors[0]
                    
                    home_team = home_data.get('team', {}).get('displayName', '')
                    away_team = away_data.get('team', {}).get('displayName', '')
                    
                    # Ищем odds если есть
                    odds = comp.get('odds', [{}])
                    if odds:
                        spread = odds[0].get('details', '')
                        over_under = odds[0].get('overUnder', 0)
                        
                        # Если spread указывает на фаворита
                        if spread and '-' in spread:
                            # Негативный spread = фаворит
                            if home_team.split()[-1] in spread or home_team.split()[0] in spread:
                                picked_team = home_team
                                reasoning = f"ESPN odds favor {home_team} (spread: {spread})"
                            else:
                                picked_team = away_team
                                reasoning = f"ESPN odds favor {away_team} (spread: {spread})"
                            
                            pick_data = {
                                'source': 'ESPN Odds',
                                'home_team': home_team,
                                'away_team': away_team,
                                'picked_team': picked_team,
                                'spread': spread,
                                'over_under': over_under
                            }
                            
                            picks.append(pick_data)
                            
                            self.tracker.record_pick(
                                expert_name="ESPN Line",
                                game_id=f"{date.today().isoformat()}_{away_team}_{home_team}",
                                picked_team=picked_team,
                                confidence=PickConfidence.MEDIUM,
                                reasoning=reasoning,
                                source_platform="espn.com"
                            )
                
                except Exception as e:
                    continue
            
            logger.info(f"   ✅ Collected {len(picks)} ESPN picks")
            
        except Exception as e:
            logger.error(f"Error fetching ESPN: {e}")
        
        return picks
    
    # =========================================================================
    # ODDS API - Sharp Money / Line Movements
    # =========================================================================
    
    def get_sharp_indicators(self, odds_api_key: str = None) -> List[Dict]:
        """
        Получает индикаторы sharp money через движение линий.
        
        Логика:
        - Если линия двигается ПРОТИВ публичных денег = sharp money
        - Большая разница между букмекерами = sharps загрузились
        """
        api_key = odds_api_key or os.getenv('ODDS_API_KEY')
        
        if not api_key:
            logger.warning("ODDS_API_KEY not set for sharp indicators")
            return []
        
        logger.info("💰 Checking sharp money indicators...")
        
        url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        params = {
            'apiKey': api_key,
            'regions': 'us',
            'markets': 'spreads,h2h',
            'oddsFormat': 'american'
        }
        
        try:
            self._rate_limit()
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Odds API returned {response.status_code}")
                return []
            
            data = response.json()
            indicators = []
            
            for game in data:
                home_team = game.get('home_team', '')
                away_team = game.get('away_team', '')
                
                bookmakers = game.get('bookmakers', [])
                if len(bookmakers) < 2:
                    continue
                
                # Собираем spreads от всех букмекеров
                home_spreads = []
                home_odds_list = []
                
                for bookmaker in bookmakers:
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'spreads':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home_team:
                                    spread = outcome.get('point', 0)
                                    price = outcome.get('price', -110)
                                    home_spreads.append(spread)
                                    home_odds_list.append(price)
                        
                        elif market.get('key') == 'h2h':
                            for outcome in market.get('outcomes', []):
                                if outcome.get('name') == home_team:
                                    ml_price = outcome.get('price', 0)
                
                if len(home_spreads) >= 2:
                    spread_variance = max(home_spreads) - min(home_spreads)
                    avg_spread = sum(home_spreads) / len(home_spreads)
                    
                    # Большая разница между букмекерами = sharp action
                    if spread_variance >= 1.0:
                        # Определяем сторону sharp money
                        # Если spread движется в сторону home (становится больше) = sharps на away
                        # Если spread движется в сторону away (становится меньше) = sharps на home
                        
                        if avg_spread < 0:  # Home is favorite
                            sharp_side = home_team
                            reasoning = f"Home favored by {abs(avg_spread):.1f}, variance {spread_variance:.1f} across books"
                        else:  # Away is favorite
                            sharp_side = away_team
                            reasoning = f"Away favored, home +{avg_spread:.1f}, variance {spread_variance:.1f}"
                        
                        signal_strength = 'strong' if spread_variance >= 2.0 else 'moderate'
                        
                        indicator = {
                            'game': f"{away_team} @ {home_team}",
                            'home_team': home_team,
                            'away_team': away_team,
                            'sharp_side': sharp_side,
                            'avg_spread': avg_spread,
                            'spread_variance': spread_variance,
                            'signal_strength': signal_strength,
                            'books_count': len(home_spreads)
                        }
                        
                        indicators.append(indicator)
                        
                        # Записываем pick
                        confidence = PickConfidence.HIGH if signal_strength == 'strong' else PickConfidence.MEDIUM
                        
                        self.tracker.record_pick(
                            expert_name="Sharp Money Indicator",
                            game_id=f"{date.today().isoformat()}_{away_team}_{home_team}",
                            picked_team=sharp_side,
                            confidence=confidence,
                            reasoning=reasoning,
                            key_factors=[f"Spread variance: {spread_variance:.1f}", f"Books analyzed: {len(home_spreads)}"],
                            source_platform="odds_api"
                        )
            
            logger.info(f"   ✅ Found {len(indicators)} sharp money indicators")
            return indicators
            
        except Exception as e:
            logger.error(f"Error fetching sharp indicators: {e}")
            return []
    
    # =========================================================================
    # MASTER COLLECTION
    # =========================================================================
    
    def collect_all_picks(self, odds_api_key: str = None) -> Dict:
        """
        Собирает прогнозы из всех источников.
        
        Вызывать утром перед играми.
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔄 COLLECTING EXPERT PICKS FROM ALL SOURCES")
        logger.info("=" * 60)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'date': date.today().isoformat(),
            'sources': {},
            'total_picks': 0
        }
        
        # 1. Covers Consensus (публичные ставки)
        try:
            covers = self.scrape_covers_consensus()
            results['sources']['covers'] = len(covers)
            results['total_picks'] += len(covers)
        except Exception as e:
            logger.error(f"Covers scraping failed: {e}")
            results['sources']['covers'] = 0
        
        time.sleep(3)  # Пауза между источниками
        
        # 2. ESPN Expert Picks
        try:
            espn = self.scrape_espn_picks()
            results['sources']['espn'] = len(espn)
            results['total_picks'] += len(espn)
        except Exception as e:
            logger.error(f"ESPN scraping failed: {e}")
            results['sources']['espn'] = 0
        
        time.sleep(3)
        
        # 3. Sharp Money Indicators (через Odds API)
        if odds_api_key or os.getenv('ODDS_API_KEY'):
            try:
                sharp = self.get_sharp_indicators(odds_api_key)
                results['sources']['sharp_money'] = len(sharp)
                results['total_picks'] += len(sharp)
            except Exception as e:
                logger.error(f"Sharp money check failed: {e}")
                results['sources']['sharp_money'] = 0
        
        # Логируем результаты
        self.collection_log.append(results)
        
        logger.info("\n📊 Collection Summary:")
        for source, count in results['sources'].items():
            logger.info(f"   {source}: {count} picks")
        logger.info(f"   TOTAL: {results['total_picks']} picks")
        
        return results
    
    def update_results(self, games_results: List[Dict]):
        """
        Обновляет результаты прогнозов после завершения игр.
        
        Args:
            games_results: [{
                'home_team': 'Lakers',
                'away_team': 'Warriors', 
                'home_score': 115,
                'away_score': 108,
                'winner': 'Lakers'
            }, ...]
        """
        logger.info("📝 Updating pick results...")
        
        cursor = self.tracker.conn.cursor()
        
        for game in games_results:
            winner = game.get('winner')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            if not winner:
                continue
            
            # Находим все прогнозы на эту игру
            cursor.execute("""
                SELECT pick_id, picked_team FROM picks
                WHERE game_date = ? AND result IS NULL
                AND (picked_team LIKE ? OR picked_team LIKE ?)
            """, (
                date.today().isoformat(),
                f"%{home_team}%",
                f"%{away_team}%"
            ))
            
            for row in cursor.fetchall():
                pick_id = row['pick_id']
                picked_team = row['picked_team']
                
                # Определяем результат
                if winner.lower() in picked_team.lower() or picked_team.lower() in winner.lower():
                    result = 'won'
                else:
                    result = 'lost'
                
                # Обновляем
                score_str = f"{game.get('home_score', 0)}-{game.get('away_score', 0)}"
                self.tracker.update_pick_result(pick_id, result, score_str)
        
        self.tracker.conn.commit()
        logger.info("   ✅ Results updated")


class ExpertPicksScheduler:
    """
    Планировщик автоматического сбора прогнозов.
    
    Расписание:
    - 10:00 - Первый сбор (утренние прогнозы)
    - 16:00 - Второй сбор (обновленные прогнозы перед играми)
    - 02:00 - Обновление результатов (после ночных игр)
    """
    
    def __init__(self, tracker: ExpertPicksTracker, odds_api_key: str = None):
        self.tracker = tracker
        self.scraper = AutoScraper(tracker)
        self.odds_api_key = odds_api_key or os.getenv('ODDS_API_KEY')
        
        self.last_collection: Optional[datetime] = None
        self.last_results_update: Optional[datetime] = None
    
    def should_collect(self) -> bool:
        """Проверяет нужно ли собирать прогнозы"""
        now = datetime.now()
        
        # Не собирали сегодня
        if not self.last_collection or self.last_collection.date() < now.date():
            return True
        
        # Прошло больше 6 часов с последнего сбора
        if (now - self.last_collection).total_seconds() > 6 * 3600:
            return True
        
        return False
    
    def should_update_results(self) -> bool:
        """Проверяет нужно ли обновлять результаты"""
        now = datetime.now()
        
        # Не обновляли сегодня и уже после полуночи
        if not self.last_results_update or self.last_results_update.date() < now.date():
            if now.hour >= 1:  # После 1:00 ночи
                return True
        
        return False
    
    def run_collection_cycle(self) -> Dict:
        """Запускает цикл сбора"""
        if not self.should_collect():
            logger.info("⏭️ Skipping collection (already collected recently)")
            return {}
        
        results = self.scraper.collect_all_picks(self.odds_api_key)
        self.last_collection = datetime.now()
        
        return results
    
    def run_results_update(self, games_results: List[Dict]):
        """Запускает обновление результатов"""
        self.scraper.update_results(games_results)
        self.last_results_update = datetime.now()


# =========================================================================
# ТЕСТИРОВАНИЕ
# =========================================================================

if __name__ == "__main__":
    print("=== Expert Picks Tracker Test ===\n")
    
    tracker = ExpertPicksTracker()
    
    # Показываем экспертов
    experts = tracker.get_all_experts()
    print(f"📊 Total experts in database: {len(experts)}")
    
    for exp in experts[:5]:
        print(f"   {exp.name} ({exp.expert_type.value})")
        if exp.total_picks > 0:
            print(f"      Record: {exp.wins}-{exp.losses} ({exp.win_rate:.1%})")
    
    # Тестируем автоматический сбор
    if SCRAPING_AVAILABLE:
        print("\n🔄 Testing auto-scraping...")
        
        scraper = AutoScraper(tracker)
        
        # Собираем из всех источников
        results = scraper.collect_all_picks()
        
        print(f"\n📊 Collection Results:")
        print(f"   Date: {results.get('date')}")
        print(f"   Total picks: {results.get('total_picks')}")
        for source, count in results.get('sources', {}).items():
            print(f"   - {source}: {count}")
    else:
        print("\n⚠️ BeautifulSoup not installed. Run:")
        print("   pip install beautifulsoup4")
    
    # Показываем leaderboard
    print("\n🏆 Expert Leaderboard (min 5 picks):")
    leaderboard = tracker.get_leaderboard(min_picks=5)
    
    for entry in leaderboard[:10]:
        sharp_marker = "🔥" if entry['is_sharp'] else ""
        print(f"   {entry['rank']}. {entry['name']} {sharp_marker}")
        print(f"      {entry['record']} ({entry['win_rate']}) | ROI: {entry['roi']}")
    
    print("\n✅ Test complete")
