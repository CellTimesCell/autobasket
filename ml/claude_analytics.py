"""
AutoBasket - Claude AI Analytics
================================
Интеграция Claude API для глубокого анализа и принятия решений
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """Типы анализа"""
    PRE_GAME = "pre_game"           # Перед игрой
    INJURY_ASSESSMENT = "injury"     # Оценка травм
    LINE_ANOMALY = "line_anomaly"    # Аномалия линии
    BET_VALIDATION = "bet_validation" # Проверка ставки
    POST_GAME = "post_game"          # После игры
    NEWS_SENTIMENT = "news"          # Анализ новостей
    MATCHUP_DEEP = "matchup"         # Глубокий матчап


@dataclass
class ClaudeAnalysis:
    """Результат анализа Claude"""
    analysis_type: AnalysisType
    timestamp: datetime
    
    # Основной вывод
    recommendation: str  # "bet", "skip", "reduce", "increase"
    confidence: float    # 0-1
    reasoning: str
    
    # Детали
    key_factors: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    
    # Корректировки
    suggested_adjustments: Dict = field(default_factory=dict)
    
    # Мета
    tokens_used: int = 0
    model: str = ""


@dataclass
class GameContext:
    """Контекст игры для анализа"""
    game_id: str
    home_team: str
    away_team: str
    game_date: date
    
    # Наш прогноз
    predicted_home_prob: float
    predicted_margin: float
    predicted_total: float
    model_confidence: float
    
    # Рыночные данные
    home_odds: float = 0.0
    away_odds: float = 0.0
    spread_line: float = 0.0
    total_line: float = 0.0
    
    # Контекст
    injuries: List[Dict] = field(default_factory=list)
    recent_news: List[str] = field(default_factory=list)
    h2h_summary: str = ""
    
    # Дополнительно
    home_record: str = ""
    away_record: str = ""
    home_streak: str = ""
    away_streak: str = ""
    rest_days_home: int = 1
    rest_days_away: int = 1


class ClaudeAnalyzer:
    """
    Основной анализатор на базе Claude API
    """
    
    SYSTEM_PROMPT = """You are an expert NBA betting analyst with deep knowledge of:
- Team dynamics, player impacts, and coaching strategies
- Statistical analysis and probability assessment
- Market inefficiencies and line value
- Injury impact assessment
- Situational factors (rest, travel, motivation)

Your role is to analyze betting opportunities and provide actionable insights.
Be direct, concise, and focus on factors that actually move the needle.
Always consider contrarian angles and market psychology.

Respond in JSON format with the following structure:
{
    "recommendation": "bet|skip|reduce|increase",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "key_factors": ["factor1", "factor2"],
    "risks": ["risk1", "risk2"],
    "adjustments": {"probability": 0.0, "edge": 0.0}
}"""

    def __init__(
        self,
        api_key: str = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        min_bet_for_analysis: float = 10.0,
        confidence_threshold: Tuple[float, float] = (0.55, 0.65)
    ):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        self.model = model
        self.max_tokens = max_tokens
        self.min_bet_for_analysis = min_bet_for_analysis
        self.confidence_threshold = confidence_threshold
        
        # Статистика использования
        self.total_tokens_used = 0
        self.analyses_count = 0
        
        # Кэш для экономии токенов
        self._cache: Dict[str, ClaudeAnalysis] = {}
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
            if not ANTHROPIC_AVAILABLE:
                logger.warning("anthropic package not installed. pip install anthropic")
            if not self.api_key:
                logger.warning("ANTHROPIC_API_KEY not set")
    
    def should_analyze(
        self,
        bet_amount: float,
        model_confidence: float,
        is_anomaly: bool = False
    ) -> bool:
        """
        Решает, нужен ли анализ Claude
        Экономит API вызовы
        """
        # Всегда анализируем аномалии
        if is_anomaly:
            return True
        
        # Маленькие ставки - не тратим токены
        if bet_amount < self.min_bet_for_analysis:
            return False
        
        # Серая зона confidence - нужна помощь
        min_conf, max_conf = self.confidence_threshold
        if min_conf <= model_confidence <= max_conf:
            return True
        
        # Большие ставки всегда проверяем
        if bet_amount >= 25.0:
            return True
        
        return False
    
    def analyze_game(
        self,
        context: GameContext,
        analysis_type: AnalysisType = AnalysisType.PRE_GAME
    ) -> ClaudeAnalysis:
        """
        Основной метод анализа игры
        """
        # Проверяем кэш
        cache_key = f"{context.game_id}_{analysis_type.value}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Кэш валиден 2 часа
            if (datetime.now() - cached.timestamp).seconds < 7200:
                return cached
        
        # Формируем промпт
        prompt = self._build_prompt(context, analysis_type)
        
        # Вызываем Claude
        response = self._call_claude(prompt, analysis_type)
        
        # Кэшируем
        self._cache[cache_key] = response
        
        return response
    
    def validate_bet(
        self,
        context: GameContext,
        bet_side: str,
        bet_amount: float,
        our_edge: float
    ) -> ClaudeAnalysis:
        """
        Валидация конкретной ставки перед размещением
        """
        prompt = f"""Validate this NBA bet:

GAME: {context.away_team} @ {context.home_team}
DATE: {context.game_date}

BET DETAILS:
- Side: {bet_side}
- Amount: ${bet_amount:.2f}
- Our calculated edge: {our_edge:.1%}
- Model confidence: {context.model_confidence:.1%}

MARKET:
- Spread: {context.home_team} {context.spread_line}
- Total: {context.total_line}
- Moneyline: Home {context.home_odds}, Away {context.away_odds}

OUR PREDICTION:
- Home win prob: {context.predicted_home_prob:.1%}
- Predicted margin: {context.predicted_margin:+.1f}
- Predicted total: {context.predicted_total:.1f}

CONTEXT:
- Home record: {context.home_record}, streak: {context.home_streak}
- Away record: {context.away_record}, streak: {context.away_streak}
- Rest days: Home {context.rest_days_home}, Away {context.rest_days_away}
- H2H: {context.h2h_summary}

INJURIES:
{self._format_injuries(context.injuries)}

RECENT NEWS:
{chr(10).join(context.recent_news[:5]) if context.recent_news else 'No significant news'}

Analyze if this bet is worth taking. Consider:
1. Is our edge real or are we missing something?
2. Any red flags in the injury/news situation?
3. Is the line value actually there?
4. What could go wrong?

Provide your recommendation in JSON format."""

        return self._call_claude(prompt, AnalysisType.BET_VALIDATION)
    
    def analyze_injury_impact(
        self,
        team: str,
        injuries: List[Dict],
        opponent: str
    ) -> ClaudeAnalysis:
        """
        Глубокий анализ влияния травм
        """
        injuries_text = self._format_injuries(injuries)
        
        prompt = f"""Analyze injury impact for {team} vs {opponent}:

INJURY REPORT:
{injuries_text}

For each injured player, assess:
1. How critical is this player to the team's system?
2. Who replaces them and quality drop-off?
3. Historical team performance without this player
4. Matchup-specific impact against {opponent}

Provide:
- Overall impact score (1-10)
- Point spread adjustment
- Total adjustment
- Specific vulnerabilities created

Respond in JSON format with:
{{
    "recommendation": "significant_impact|moderate_impact|minimal_impact",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "adjustments": {{"spread": 0.0, "total": 0.0, "win_prob": 0.0}}
}}"""

        return self._call_claude(prompt, AnalysisType.INJURY_ASSESSMENT)
    
    def detect_line_anomaly(
        self,
        context: GameContext,
        opening_line: float,
        current_line: float,
        public_betting_pct: float
    ) -> ClaudeAnalysis:
        """
        Анализ аномальных движений линии
        """
        line_move = current_line - opening_line
        
        prompt = f"""Analyze this line movement anomaly:

GAME: {context.away_team} @ {context.home_team}

LINE MOVEMENT:
- Opening: {context.home_team} {opening_line}
- Current: {context.home_team} {current_line}
- Movement: {line_move:+.1f} points

PUBLIC BETTING:
- {public_betting_pct:.0f}% on {context.home_team if public_betting_pct > 50 else context.away_team}

RED FLAGS TO CHECK:
1. Reverse line movement (public on one side, line moving other way)?
2. Steam move (sharp action)?
3. Injury news not yet public?
4. Insider information possibility?
5. Market overreaction?

Our model prediction: {context.home_team} {-context.predicted_margin:+.1f}

Is there value in fading the public or following the sharp money?

Respond in JSON format."""

        return self._call_claude(prompt, AnalysisType.LINE_ANOMALY)
    
    def post_game_analysis(
        self,
        context: GameContext,
        actual_home_score: int,
        actual_away_score: int,
        bet_result: str  # "won", "lost", "push"
    ) -> ClaudeAnalysis:
        """
        Анализ после игры - почему прогноз сработал/не сработал
        """
        actual_margin = actual_home_score - actual_away_score
        actual_total = actual_home_score + actual_away_score
        
        margin_error = actual_margin - context.predicted_margin
        total_error = actual_total - context.predicted_total
        
        prompt = f"""Post-game analysis:

GAME: {context.away_team} @ {context.home_team}

RESULT:
- Final: {context.away_team} {actual_away_score} - {context.home_team} {actual_home_score}
- Margin: {actual_margin:+d} (predicted {context.predicted_margin:+.1f}, error: {margin_error:+.1f})
- Total: {actual_total} (predicted {context.predicted_total:.1f}, error: {total_error:+.1f})

OUR BET RESULT: {bet_result.upper()}

PRE-GAME PREDICTION:
- Home win prob: {context.predicted_home_prob:.1%}
- Confidence: {context.model_confidence:.1%}

Analyze:
1. Why did the prediction miss (if it did)?
2. What factors weren't properly weighted?
3. Was this predictable or variance?
4. What should we learn for future similar matchups?

Focus on actionable insights for model improvement.

Respond in JSON format with:
{{
    "recommendation": "model_correct|model_wrong_fixable|model_wrong_variance",
    "reasoning": "explanation",
    "key_factors": ["what we missed"],
    "adjustments": {{"factor_name": weight_change}}
}}"""

        return self._call_claude(prompt, AnalysisType.POST_GAME)
    
    def analyze_news_sentiment(
        self,
        team: str,
        news_items: List[str]
    ) -> ClaudeAnalysis:
        """
        Анализ новостей и настроений
        """
        prompt = f"""Analyze recent news sentiment for {team}:

NEWS ITEMS:
{chr(10).join(f'- {item}' for item in news_items)}

Assess:
1. Overall sentiment (positive/negative/neutral)
2. Impact on team performance
3. Locker room dynamics
4. Motivation factors
5. Any hidden concerns?

Rate impact on upcoming games.

Respond in JSON format with:
{{
    "recommendation": "positive|negative|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "adjustments": {{"morale_factor": -1.0 to 1.0}}
}}"""

        return self._call_claude(prompt, AnalysisType.NEWS_SENTIMENT)
    
    def get_matchup_insights(
        self,
        context: GameContext,
        home_stats: Dict,
        away_stats: Dict,
        coach_info: Dict
    ) -> ClaudeAnalysis:
        """
        Глубокий анализ матчапа
        """
        prompt = f"""Deep matchup analysis:

GAME: {context.away_team} @ {context.home_team}

HOME TEAM ({context.home_team}):
- Record: {context.home_record}
- Off Rating: {home_stats.get('off_rating', 'N/A')}
- Def Rating: {home_stats.get('def_rating', 'N/A')}
- Pace: {home_stats.get('pace', 'N/A')}
- Coach: {coach_info.get('home_coach', 'N/A')} ({coach_info.get('home_style', 'N/A')})

AWAY TEAM ({context.away_team}):
- Record: {context.away_record}
- Off Rating: {away_stats.get('off_rating', 'N/A')}
- Def Rating: {away_stats.get('def_rating', 'N/A')}
- Pace: {away_stats.get('pace', 'N/A')}
- Coach: {coach_info.get('away_coach', 'N/A')} ({coach_info.get('away_style', 'N/A')})

H2H HISTORY: {context.h2h_summary}

MARKET:
- Spread: {context.home_team} {context.spread_line}
- Total: {context.total_line}

Analyze:
1. Style matchup (pace, offensive/defensive strengths)
2. Coaching chess match
3. Key player matchups
4. Historical patterns
5. Situational edges

Where is the market potentially wrong?

Respond in JSON format."""

        return self._call_claude(prompt, AnalysisType.MATCHUP_DEEP)
    
    def _build_prompt(self, context: GameContext, analysis_type: AnalysisType) -> str:
        """Строит промпт для анализа"""
        base = f"""Analyze this NBA game:

{context.away_team} @ {context.home_team}
Date: {context.game_date}

Our Model Prediction:
- Home win probability: {context.predicted_home_prob:.1%}
- Predicted margin: {context.predicted_margin:+.1f}
- Predicted total: {context.predicted_total:.1f}
- Model confidence: {context.model_confidence:.1%}

Market Lines:
- Spread: {context.home_team} {context.spread_line}
- Total: {context.total_line}
- Moneyline: Home {context.home_odds}, Away {context.away_odds}

Team Context:
- {context.home_team}: {context.home_record}, streak: {context.home_streak}, rest: {context.rest_days_home} days
- {context.away_team}: {context.away_record}, streak: {context.away_streak}, rest: {context.rest_days_away} days

H2H: {context.h2h_summary}

Injuries:
{self._format_injuries(context.injuries)}

Recent News:
{chr(10).join(context.recent_news[:3]) if context.recent_news else 'None'}

Provide analysis focusing on:
1. Is our model probability accurate?
2. Where is the value (if any)?
3. Key factors that could swing the game
4. Risks we should consider

Respond in JSON format."""
        
        return base
    
    def _format_injuries(self, injuries: List[Dict]) -> str:
        """Форматирует список травм"""
        if not injuries:
            return "No significant injuries reported"
        
        lines = []
        for inj in injuries:
            player = inj.get('player', 'Unknown')
            status = inj.get('status', 'Unknown')
            injury_type = inj.get('injury', '')
            team = inj.get('team', '')
            
            lines.append(f"- {player} ({team}): {status} - {injury_type}")
        
        return "\n".join(lines)
    
    def _call_claude(self, prompt: str, analysis_type: AnalysisType) -> ClaudeAnalysis:
        """Вызывает Claude API"""
        
        if not self.client:
            logger.warning("Claude client not available, returning default analysis")
            return self._default_analysis(analysis_type)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Парсим ответ
            content = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            
            self.total_tokens_used += tokens
            self.analyses_count += 1
            
            # Пытаемся распарсить JSON
            try:
                # Извлекаем JSON из ответа
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    data = json.loads(json_str)
                else:
                    data = {}
            except json.JSONDecodeError:
                data = {}
            
            return ClaudeAnalysis(
                analysis_type=analysis_type,
                timestamp=datetime.now(),
                recommendation=data.get('recommendation', 'skip'),
                confidence=float(data.get('confidence', 0.5)),
                reasoning=data.get('reasoning', content[:500]),
                key_factors=data.get('key_factors', []),
                risks=data.get('risks', []),
                opportunities=data.get('opportunities', []),
                suggested_adjustments=data.get('adjustments', {}),
                tokens_used=tokens,
                model=self.model
            )
            
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._default_analysis(analysis_type)
    
    def _default_analysis(self, analysis_type: AnalysisType) -> ClaudeAnalysis:
        """Возвращает дефолтный анализ при ошибке"""
        return ClaudeAnalysis(
            analysis_type=analysis_type,
            timestamp=datetime.now(),
            recommendation="skip",
            confidence=0.5,
            reasoning="Analysis unavailable - using model prediction only",
            key_factors=["Claude API unavailable"],
            risks=["No external validation"],
            tokens_used=0,
            model="none"
        )
    
    def get_usage_stats(self) -> Dict:
        """Возвращает статистику использования"""
        return {
            'total_tokens': self.total_tokens_used,
            'total_analyses': self.analyses_count,
            'avg_tokens_per_analysis': self.total_tokens_used / max(self.analyses_count, 1),
            'estimated_cost': self.total_tokens_used * 0.000003  # Примерная стоимость
        }


class SmartBettingAdvisor:
    """
    Умный советник по ставкам
    Комбинирует ML модель + Claude анализ
    """
    
    def __init__(self, claude_analyzer: ClaudeAnalyzer = None):
        self.claude = claude_analyzer or ClaudeAnalyzer()
    
    def get_recommendation(
        self,
        context: GameContext,
        ml_recommendation: str,  # "bet", "skip"
        ml_edge: float,
        bet_amount: float
    ) -> Dict:
        """
        Получает финальную рекомендацию
        """
        result = {
            'ml_recommendation': ml_recommendation,
            'ml_edge': ml_edge,
            'ml_confidence': context.model_confidence,
            'claude_used': False,
            'final_recommendation': ml_recommendation,
            'final_confidence': context.model_confidence,
            'reasoning': []
        }
        
        # Проверяем, нужен ли Claude
        is_anomaly = abs(context.predicted_margin - context.spread_line) > 5
        
        if not self.claude.should_analyze(bet_amount, context.model_confidence, is_anomaly):
            result['reasoning'].append("Small bet / high confidence - using ML only")
            return result
        
        # Запускаем Claude анализ
        result['claude_used'] = True
        
        analysis = self.claude.validate_bet(
            context=context,
            bet_side="home" if ml_recommendation == "bet" and context.predicted_home_prob > 0.5 else "away",
            bet_amount=bet_amount,
            our_edge=ml_edge
        )
        
        result['claude_analysis'] = {
            'recommendation': analysis.recommendation,
            'confidence': analysis.confidence,
            'reasoning': analysis.reasoning,
            'key_factors': analysis.key_factors,
            'risks': analysis.risks
        }
        
        # Комбинируем решения
        if analysis.recommendation == "skip" and analysis.confidence > 0.7:
            result['final_recommendation'] = "skip"
            result['reasoning'].append(f"Claude override: {analysis.reasoning}")
        
        elif analysis.recommendation == "reduce":
            result['final_recommendation'] = "reduce"
            result['suggested_amount'] = bet_amount * 0.5
            result['reasoning'].append(f"Claude suggests reducing: {analysis.reasoning}")
        
        elif analysis.recommendation == "bet" and ml_recommendation == "bet":
            # Оба согласны - увеличиваем уверенность
            result['final_confidence'] = min(0.9, context.model_confidence + 0.1)
            result['reasoning'].append("ML + Claude agree")
        
        elif analysis.recommendation == "increase" and analysis.confidence > 0.7:
            result['final_recommendation'] = "bet"
            result['suggested_amount'] = bet_amount * 1.5
            result['reasoning'].append(f"Claude sees extra value: {analysis.reasoning}")
        
        # Добавляем риски
        if analysis.risks:
            result['risks'] = analysis.risks
        
        # Корректировки
        if analysis.suggested_adjustments:
            result['adjustments'] = analysis.suggested_adjustments
        
        return result
    
    def daily_briefing(self, games: List[GameContext]) -> str:
        """
        Генерирует дневной брифинг по всем играм
        """
        if not self.claude.client:
            return "Claude unavailable for daily briefing"
        
        games_summary = []
        for g in games:
            games_summary.append(
                f"- {g.away_team} @ {g.home_team}: Model {g.predicted_home_prob:.0%} home, "
                f"line {g.spread_line}, conf {g.model_confidence:.0%}"
            )
        
        prompt = f"""Daily NBA betting briefing for {date.today()}:

GAMES:
{chr(10).join(games_summary)}

Provide a brief executive summary:
1. Best value plays today
2. Games to avoid
3. Key injuries/news to watch
4. Overall market assessment

Keep it concise and actionable."""

        try:
            response = self.claude.client.messages.create(
                model=self.claude.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            return f"Briefing error: {e}"


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Claude AI Analytics ===\n")
    
    # Создаем анализатор (без реального API ключа для теста)
    analyzer = ClaudeAnalyzer(api_key=None)  # Будет использовать default analysis
    
    # Тестовый контекст
    context = GameContext(
        game_id="test123",
        home_team="Los Angeles Lakers",
        away_team="Golden State Warriors",
        game_date=date.today(),
        predicted_home_prob=0.58,
        predicted_margin=4.5,
        predicted_total=228.5,
        model_confidence=0.62,
        home_odds=1.75,
        away_odds=2.10,
        spread_line=-3.5,
        total_line=226.5,
        home_record="28-18",
        away_record="25-21",
        home_streak="W3",
        away_streak="L1",
        rest_days_home=2,
        rest_days_away=1,
        h2h_summary="Lakers 3-1 this season",
        injuries=[
            {"player": "Anthony Davis", "team": "LAL", "status": "Questionable", "injury": "Knee"},
            {"player": "Draymond Green", "team": "GSW", "status": "Out", "injury": "Back"}
        ],
        recent_news=[
            "LeBron James says team chemistry at all-time high",
            "Warriors struggling on road trip"
        ]
    )
    
    # Тест should_analyze
    print("Тест 1: Should Analyze Logic")
    print("-" * 50)
    
    test_cases = [
        (5.0, 0.70, False, "Small bet, high conf"),
        (15.0, 0.60, False, "Medium bet, gray zone conf"),
        (30.0, 0.55, False, "Large bet"),
        (10.0, 0.80, True, "Anomaly detected"),
    ]
    
    for amount, conf, anomaly, desc in test_cases:
        should = analyzer.should_analyze(amount, conf, anomaly)
        print(f"  ${amount}, {conf:.0%} conf, anomaly={anomaly}: {should} ({desc})")
    
    # Тест анализа (без API - вернет default)
    print("\n\nТест 2: Game Analysis (without API)")
    print("-" * 50)
    
    analysis = analyzer.analyze_game(context)
    
    print(f"Type: {analysis.analysis_type.value}")
    print(f"Recommendation: {analysis.recommendation}")
    print(f"Confidence: {analysis.confidence:.0%}")
    print(f"Reasoning: {analysis.reasoning}")
    print(f"Tokens used: {analysis.tokens_used}")
    
    # Тест Smart Advisor
    print("\n\nТест 3: Smart Betting Advisor")
    print("-" * 50)
    
    advisor = SmartBettingAdvisor(analyzer)
    
    recommendation = advisor.get_recommendation(
        context=context,
        ml_recommendation="bet",
        ml_edge=0.08,
        bet_amount=15.0
    )
    
    print(f"ML recommendation: {recommendation['ml_recommendation']}")
    print(f"Claude used: {recommendation['claude_used']}")
    print(f"Final recommendation: {recommendation['final_recommendation']}")
    print(f"Final confidence: {recommendation['final_confidence']:.0%}")
    print(f"Reasoning: {recommendation['reasoning']}")
    
    # Статистика
    print("\n\nТест 4: Usage Stats")
    print("-" * 50)
    
    stats = analyzer.get_usage_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n" + "=" * 50)
    print("Для работы с реальным API:")
    print("  1. pip install anthropic")
    print("  2. export ANTHROPIC_API_KEY=your_key")
    print("  3. analyzer = ClaudeAnalyzer()")
    print("=" * 50)
