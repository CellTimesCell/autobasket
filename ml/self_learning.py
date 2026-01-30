"""
AutoBasket - Self-Learning Module
=================================
Автоматическое обучение и улучшение модели на основе результатов
"""

import logging
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
import statistics
from collections import defaultdict

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss, log_loss
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Запись предсказания для обучения"""
    prediction_id: int
    game_id: int
    timestamp: datetime
    
    # Предсказание
    home_team: str
    away_team: str
    predicted_home_prob: float
    predicted_winner: str
    confidence: float
    
    # Фичи использованные
    features: Dict = field(default_factory=dict)
    model_weights: Dict = field(default_factory=dict)
    
    # Результат (заполняется после игры)
    actual_home_won: Optional[bool] = None
    actual_margin: Optional[int] = None
    settled: bool = False
    settled_at: Optional[datetime] = None
    
    # Анализ ошибки
    was_correct: Optional[bool] = None
    probability_error: Optional[float] = None  # |predicted - actual|
    calibration_bucket: Optional[int] = None   # 0-9 (по 10% bins)


@dataclass
class ModelPerformance:
    """Метрики производительности модели"""
    period_start: date
    period_end: date
    
    # Базовые метрики
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    
    # Калибровка
    brier_score: float = 0.0          # Чем меньше, тем лучше (0 = идеал)
    log_loss: float = 0.0             # Чем меньше, тем лучше
    calibration_error: float = 0.0    # ECE (Expected Calibration Error)
    
    # По confidence buckets
    accuracy_by_confidence: Dict[str, float] = field(default_factory=dict)
    
    # По категориям
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)
    
    # Тренды
    accuracy_trend: str = ""  # "improving", "stable", "declining"
    recent_accuracy: float = 0.0  # Последние 50 предсказаний


@dataclass
class LearningAdjustment:
    """Корректировка модели"""
    timestamp: datetime
    adjustment_type: str  # "weight", "calibration", "feature", "threshold"
    parameter: str
    old_value: float
    new_value: float
    reason: str
    impact_estimate: float  # Ожидаемое улучшение


class PredictionTracker:
    """
    Отслеживает все предсказания и их результаты
    """
    
    def __init__(self, storage_path: str = "predictions_history.json"):
        self.storage_path = Path(storage_path)
        self.predictions: List[PredictionRecord] = []
        self._next_id = 1
        self._load()
    
    def record_prediction(
        self,
        game_id: int,
        home_team: str,
        away_team: str,
        predicted_home_prob: float,
        confidence: float,
        features: Dict = None,
        model_weights: Dict = None
    ) -> int:
        """Записывает новое предсказание"""
        
        record = PredictionRecord(
            prediction_id=self._next_id,
            game_id=game_id,
            timestamp=datetime.now(),
            home_team=home_team,
            away_team=away_team,
            predicted_home_prob=predicted_home_prob,
            predicted_winner=home_team if predicted_home_prob > 0.5 else away_team,
            confidence=confidence,
            features=self._serialize_features(features) if features else {},
            model_weights=model_weights or {}
        )
        
        # Определяем calibration bucket (0-9)
        record.calibration_bucket = min(int(predicted_home_prob * 10), 9)
        
        self.predictions.append(record)
        self._next_id += 1
        self._save()
        
        return record.prediction_id
    
    def _serialize_features(self, features: Dict) -> Dict:
        """Преобразует features в JSON-сериализуемый формат"""
        result = {}
        for key, value in features.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif hasattr(value, '__dict__'):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def record_result(
        self,
        game_id: int,
        home_won: bool,
        margin: int = None
    ) -> Optional[PredictionRecord]:
        """Записывает результат игры"""
        
        for pred in reversed(self.predictions):
            if pred.game_id == game_id and not pred.settled:
                pred.actual_home_won = home_won
                pred.actual_margin = margin
                pred.settled = True
                pred.settled_at = datetime.now()
                
                # Анализ
                pred.was_correct = (pred.predicted_home_prob > 0.5) == home_won
                pred.probability_error = abs(
                    pred.predicted_home_prob - (1.0 if home_won else 0.0)
                )
                
                self._save()
                return pred
        
        return None
    
    def get_unsettled(self) -> List[PredictionRecord]:
        """Возвращает неразрешенные предсказания"""
        return [p for p in self.predictions if not p.settled]
    
    def get_settled(self, days: int = None) -> List[PredictionRecord]:
        """Возвращает разрешенные предсказания"""
        settled = [p for p in self.predictions if p.settled]
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            settled = [p for p in settled if p.settled_at and p.settled_at > cutoff]
        
        return settled
    
    def _save(self):
        """Сохраняет в файл"""
        data = []
        for p in self.predictions:
            d = {
                'prediction_id': p.prediction_id,
                'game_id': p.game_id,
                'timestamp': p.timestamp.isoformat(),
                'home_team': p.home_team,
                'away_team': p.away_team,
                'predicted_home_prob': p.predicted_home_prob,
                'predicted_winner': p.predicted_winner,
                'confidence': p.confidence,
                'features': p.features,
                'actual_home_won': p.actual_home_won,
                'actual_margin': p.actual_margin,
                'settled': p.settled,
                'settled_at': p.settled_at.isoformat() if p.settled_at else None,
                'was_correct': p.was_correct,
                'probability_error': p.probability_error,
                'calibration_bucket': p.calibration_bucket
            }
            data.append(d)
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Загружает из файла"""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for d in data:
                p = PredictionRecord(
                    prediction_id=d['prediction_id'],
                    game_id=d['game_id'],
                    timestamp=datetime.fromisoformat(d['timestamp']),
                    home_team=d['home_team'],
                    away_team=d['away_team'],
                    predicted_home_prob=d['predicted_home_prob'],
                    predicted_winner=d['predicted_winner'],
                    confidence=d['confidence'],
                    features=d.get('features', {}),
                    actual_home_won=d.get('actual_home_won'),
                    actual_margin=d.get('actual_margin'),
                    settled=d.get('settled', False),
                    settled_at=datetime.fromisoformat(d['settled_at']) if d.get('settled_at') else None,
                    was_correct=d.get('was_correct'),
                    probability_error=d.get('probability_error'),
                    calibration_bucket=d.get('calibration_bucket')
                )
                self.predictions.append(p)
            
            if self.predictions:
                self._next_id = max(p.prediction_id for p in self.predictions) + 1
                
        except Exception as e:
            logger.error(f"Ошибка загрузки predictions: {e}")


class ModelAnalyzer:
    """
    Анализирует производительность модели
    """
    
    def __init__(self, tracker: PredictionTracker):
        self.tracker = tracker
    
    def calculate_performance(
        self,
        days: int = 30
    ) -> ModelPerformance:
        """Рассчитывает метрики за период"""
        
        predictions = self.tracker.get_settled(days=days)
        
        if not predictions:
            return ModelPerformance(
                period_start=date.today() - timedelta(days=days),
                period_end=date.today()
            )
        
        perf = ModelPerformance(
            period_start=min(p.timestamp.date() for p in predictions),
            period_end=max(p.timestamp.date() for p in predictions),
            total_predictions=len(predictions)
        )
        
        # Accuracy
        perf.correct_predictions = sum(1 for p in predictions if p.was_correct)
        perf.accuracy = perf.correct_predictions / perf.total_predictions
        
        # Brier Score & Log Loss
        if NUMPY_AVAILABLE:
            y_true = [1 if p.actual_home_won else 0 for p in predictions]
            y_prob = [p.predicted_home_prob for p in predictions]
            
            perf.brier_score = self._brier_score(y_true, y_prob)
            perf.log_loss = self._log_loss(y_true, y_prob)
            perf.calibration_error = self._calibration_error(y_true, y_prob)
        
        # Accuracy by confidence
        perf.accuracy_by_confidence = self._accuracy_by_confidence(predictions)
        
        # Recent accuracy (last 50)
        recent = predictions[-50:] if len(predictions) >= 50 else predictions
        perf.recent_accuracy = sum(1 for p in recent if p.was_correct) / len(recent)
        
        # Trend
        if len(predictions) >= 100:
            first_half = predictions[:len(predictions)//2]
            second_half = predictions[len(predictions)//2:]
            
            first_acc = sum(1 for p in first_half if p.was_correct) / len(first_half)
            second_acc = sum(1 for p in second_half if p.was_correct) / len(second_half)
            
            if second_acc > first_acc + 0.02:
                perf.accuracy_trend = "improving"
            elif second_acc < first_acc - 0.02:
                perf.accuracy_trend = "declining"
            else:
                perf.accuracy_trend = "stable"
        
        return perf
    
    def _brier_score(self, y_true: List[int], y_prob: List[float]) -> float:
        """Brier Score - среднеквадратичная ошибка вероятностей"""
        return sum((p - y)**2 for p, y in zip(y_prob, y_true)) / len(y_true)
    
    def _log_loss(self, y_true: List[int], y_prob: List[float]) -> float:
        """Log Loss"""
        import math
        eps = 1e-15
        total = 0
        for y, p in zip(y_true, y_prob):
            p = max(min(p, 1 - eps), eps)
            total += y * math.log(p) + (1 - y) * math.log(1 - p)
        return -total / len(y_true)
    
    def _calibration_error(self, y_true: List[int], y_prob: List[float]) -> float:
        """Expected Calibration Error"""
        bins = defaultdict(list)
        
        for y, p in zip(y_true, y_prob):
            bucket = min(int(p * 10), 9)
            bins[bucket].append((y, p))
        
        ece = 0
        total = len(y_true)
        
        for bucket, values in bins.items():
            if not values:
                continue
            
            n = len(values)
            actual_freq = sum(v[0] for v in values) / n
            avg_prob = sum(v[1] for v in values) / n
            
            ece += (n / total) * abs(actual_freq - avg_prob)
        
        return ece
    
    def _accuracy_by_confidence(self, predictions: List[PredictionRecord]) -> Dict[str, float]:
        """Accuracy по диапазонам уверенности"""
        buckets = {
            '50-55%': [],
            '55-60%': [],
            '60-65%': [],
            '65-70%': [],
            '70-75%': [],
            '75%+': []
        }
        
        for p in predictions:
            conf = p.confidence
            if conf < 0.55:
                buckets['50-55%'].append(p.was_correct)
            elif conf < 0.60:
                buckets['55-60%'].append(p.was_correct)
            elif conf < 0.65:
                buckets['60-65%'].append(p.was_correct)
            elif conf < 0.70:
                buckets['65-70%'].append(p.was_correct)
            elif conf < 0.75:
                buckets['70-75%'].append(p.was_correct)
            else:
                buckets['75%+'].append(p.was_correct)
        
        return {
            k: sum(v) / len(v) if v else 0
            for k, v in buckets.items()
        }
    
    def find_weak_spots(self, predictions: List[PredictionRecord]) -> List[Dict]:
        """Находит слабые места модели"""
        weak_spots = []
        
        # По командам
        team_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for p in predictions:
            for team in [p.home_team, p.away_team]:
                team_performance[team]['total'] += 1
                if p.was_correct:
                    team_performance[team]['correct'] += 1
        
        for team, stats in team_performance.items():
            if stats['total'] >= 5:
                acc = stats['correct'] / stats['total']
                if acc < 0.45:
                    weak_spots.append({
                        'type': 'team',
                        'subject': team,
                        'accuracy': acc,
                        'sample_size': stats['total'],
                        'recommendation': f"Проверить данные/модель для {team}"
                    })
        
        # По confidence - overconfidence detection
        acc_by_conf = self._accuracy_by_confidence(predictions)
        
        for bucket, acc in acc_by_conf.items():
            expected_min = float(bucket.split('-')[0].replace('%', '').replace('+', '')) / 100
            if acc < expected_min - 0.05 and acc > 0:
                weak_spots.append({
                    'type': 'overconfidence',
                    'subject': bucket,
                    'accuracy': acc,
                    'expected': expected_min,
                    'recommendation': f"Калибровать вероятности в диапазоне {bucket}"
                })
        
        return weak_spots


class SelfLearner:
    """
    Главный модуль самообучения
    
    Функции:
    1. Отслеживание результатов
    2. Анализ ошибок
    3. Автоматическая корректировка
    4. Переобучение моделей
    """
    
    def __init__(
        self,
        tracker: PredictionTracker = None,
        model_path: str = "model_state.pkl",
        min_samples_for_adjustment: int = 50
    ):
        self.tracker = tracker or PredictionTracker()
        self.analyzer = ModelAnalyzer(self.tracker)
        self.model_path = Path(model_path)
        self.min_samples = min_samples_for_adjustment
        
        # Текущее состояние модели
        self.model_state = {
            'calibration_factor': 0.85,
            'home_advantage': 3.5,
            'model_weights': {
                'xgboost': 0.45,
                'gradient_boost': 0.35,
                'random_forest': 0.20
            },
            'feature_weights': {
                'elo_diff': 1.0,
                'net_rating_diff': 1.0,
                'rest_advantage': 1.0,
                'injury_impact': 1.0,
                'recent_form': 1.0,
                'h2h': 1.0
            },
            'confidence_thresholds': {
                'min_bet': 0.52,
                'safe_bet': 0.70,
                'high_value': 0.60
            }
        }
        
        # История корректировок
        self.adjustments: List[LearningAdjustment] = []
        
        self._load_state()
    
    def learn_from_results(self) -> Dict:
        """
        Главный метод обучения
        Анализирует результаты и корректирует модель
        """
        logger.info("Starting learning cycle...")
        
        # 1. Получаем метрики
        perf = self.analyzer.calculate_performance(days=30)
        
        if perf.total_predictions < self.min_samples:
            logger.info(f"Not enough data: {perf.total_predictions}/{self.min_samples}")
            return {'status': 'insufficient_data', 'samples': perf.total_predictions}
        
        adjustments_made = []
        
        # 2. Проверяем калибровку
        cal_adjustment = self._adjust_calibration(perf)
        if cal_adjustment:
            adjustments_made.append(cal_adjustment)
        
        # 3. Проверяем home advantage
        ha_adjustment = self._adjust_home_advantage()
        if ha_adjustment:
            adjustments_made.append(ha_adjustment)
        
        # 4. Корректируем веса фичей на основе ошибок
        feature_adjustments = self._adjust_feature_weights()
        adjustments_made.extend(feature_adjustments)
        
        # 5. Корректируем пороги уверенности
        threshold_adjustment = self._adjust_confidence_thresholds(perf)
        if threshold_adjustment:
            adjustments_made.append(threshold_adjustment)
        
        # 6. Сохраняем состояние
        self._save_state()
        
        return {
            'status': 'success',
            'performance': {
                'accuracy': perf.accuracy,
                'brier_score': perf.brier_score,
                'calibration_error': perf.calibration_error,
                'trend': perf.accuracy_trend
            },
            'adjustments': len(adjustments_made),
            'details': [
                {'type': a.adjustment_type, 'param': a.parameter, 'change': f"{a.old_value:.3f} → {a.new_value:.3f}"}
                for a in adjustments_made
            ]
        }
    
    def _adjust_calibration(self, perf: ModelPerformance) -> Optional[LearningAdjustment]:
        """Корректирует фактор калибровки"""
        
        # Если calibration error высокий, нужно сжать вероятности к 0.5
        if perf.calibration_error > 0.05:
            old_factor = self.model_state['calibration_factor']
            
            # Увеличиваем сжатие
            new_factor = old_factor * 0.95
            new_factor = max(new_factor, 0.70)  # Не сжимать слишком сильно
            
            if abs(new_factor - old_factor) > 0.01:
                self.model_state['calibration_factor'] = new_factor
                
                adj = LearningAdjustment(
                    timestamp=datetime.now(),
                    adjustment_type='calibration',
                    parameter='calibration_factor',
                    old_value=old_factor,
                    new_value=new_factor,
                    reason=f"High calibration error: {perf.calibration_error:.3f}",
                    impact_estimate=perf.calibration_error * 0.2
                )
                self.adjustments.append(adj)
                return adj
        
        # Если модель underpredicting (accuracy выше confidence)
        elif perf.calibration_error < 0.02 and perf.accuracy > 0.55:
            old_factor = self.model_state['calibration_factor']
            
            # Уменьшаем сжатие (больше доверяем модели)
            new_factor = min(old_factor * 1.02, 0.95)
            
            if abs(new_factor - old_factor) > 0.01:
                self.model_state['calibration_factor'] = new_factor
                
                adj = LearningAdjustment(
                    timestamp=datetime.now(),
                    adjustment_type='calibration',
                    parameter='calibration_factor',
                    old_value=old_factor,
                    new_value=new_factor,
                    reason="Model underconfident, increasing trust",
                    impact_estimate=0.01
                )
                self.adjustments.append(adj)
                return adj
        
        return None
    
    def _adjust_home_advantage(self) -> Optional[LearningAdjustment]:
        """Корректирует home advantage на основе реальных данных"""
        
        predictions = self.tracker.get_settled(days=60)
        
        if len(predictions) < 100:
            return None
        
        # Считаем реальный home win %
        home_wins = sum(1 for p in predictions if p.actual_home_won)
        actual_home_pct = home_wins / len(predictions)
        
        # Текущий ожидаемый home advantage в вероятности
        # ~3.5 очка ≈ ~55% win probability
        expected_home_pct = 0.55
        
        old_ha = self.model_state['home_advantage']
        
        # Корректируем если отклонение > 2%
        if abs(actual_home_pct - expected_home_pct) > 0.02:
            # Каждый 1% разницы ≈ 0.3 очка
            adjustment = (actual_home_pct - expected_home_pct) * 30
            new_ha = old_ha + adjustment * 0.5  # Плавная корректировка
            new_ha = max(2.0, min(5.0, new_ha))  # Разумные границы
            
            if abs(new_ha - old_ha) > 0.1:
                self.model_state['home_advantage'] = new_ha
                
                adj = LearningAdjustment(
                    timestamp=datetime.now(),
                    adjustment_type='feature',
                    parameter='home_advantage',
                    old_value=old_ha,
                    new_value=new_ha,
                    reason=f"Actual home win%: {actual_home_pct:.1%} vs expected {expected_home_pct:.1%}",
                    impact_estimate=abs(adjustment) * 0.01
                )
                self.adjustments.append(adj)
                return adj
        
        return None
    
    def _adjust_feature_weights(self) -> List[LearningAdjustment]:
        """Корректирует веса фичей на основе их предсказательной силы"""
        
        predictions = self.tracker.get_settled(days=30)
        adjustments = []
        
        if len(predictions) < self.min_samples:
            return adjustments
        
        # Анализируем каждую фичу
        for feature in self.model_state['feature_weights']:
            # Группируем по значению фичи
            high_feature = [p for p in predictions if p.features.get(feature, 0) > 0.5]
            low_feature = [p for p in predictions if p.features.get(feature, 0) <= 0.5]
            
            if len(high_feature) < 10 or len(low_feature) < 10:
                continue
            
            # Accuracy когда фича "за" предсказание
            high_acc = sum(1 for p in high_feature if p.was_correct) / len(high_feature)
            low_acc = sum(1 for p in low_feature if p.was_correct) / len(low_feature)
            
            old_weight = self.model_state['feature_weights'][feature]
            
            # Если фича помогает - увеличиваем вес
            if high_acc > low_acc + 0.05:
                new_weight = min(old_weight * 1.1, 1.5)
            elif low_acc > high_acc + 0.05:
                new_weight = max(old_weight * 0.9, 0.5)
            else:
                continue
            
            if abs(new_weight - old_weight) > 0.05:
                self.model_state['feature_weights'][feature] = new_weight
                
                adj = LearningAdjustment(
                    timestamp=datetime.now(),
                    adjustment_type='feature',
                    parameter=f'feature_weight_{feature}',
                    old_value=old_weight,
                    new_value=new_weight,
                    reason=f"High feature acc: {high_acc:.1%}, Low: {low_acc:.1%}",
                    impact_estimate=abs(high_acc - low_acc) * 0.1
                )
                self.adjustments.append(adj)
                adjustments.append(adj)
        
        return adjustments
    
    def _adjust_confidence_thresholds(self, perf: ModelPerformance) -> Optional[LearningAdjustment]:
        """Корректирует пороги уверенности"""
        
        acc_by_conf = perf.accuracy_by_confidence
        
        # Находим минимальный порог с accuracy > 52%
        thresholds = [
            (0.525, '50-55%'),
            (0.575, '55-60%'),
            (0.625, '60-65%'),
            (0.675, '65-70%'),
            (0.725, '70-75%'),
            (0.775, '75%+')
        ]
        
        optimal_min = 0.52
        for threshold, bucket in thresholds:
            if bucket in acc_by_conf and acc_by_conf[bucket] >= 0.52:
                optimal_min = threshold - 0.025
                break
        
        old_min = self.model_state['confidence_thresholds']['min_bet']
        
        if abs(optimal_min - old_min) > 0.02:
            self.model_state['confidence_thresholds']['min_bet'] = optimal_min
            
            adj = LearningAdjustment(
                timestamp=datetime.now(),
                adjustment_type='threshold',
                parameter='min_bet_confidence',
                old_value=old_min,
                new_value=optimal_min,
                reason=f"Accuracy analysis suggests new threshold",
                impact_estimate=0.02
            )
            self.adjustments.append(adj)
            return adj
        
        return None
    
    def get_adjusted_prediction(
        self,
        raw_home_prob: float,
        features: Dict = None
    ) -> Tuple[float, float]:
        """
        Применяет learned корректировки к предсказанию
        
        Returns:
            (adjusted_probability, confidence)
        """
        # 1. Применяем калибровку
        cal_factor = self.model_state['calibration_factor']
        calibrated = 0.5 + (raw_home_prob - 0.5) * cal_factor
        
        # 2. Применяем веса фичей (если есть)
        if features:
            feature_adjustment = 0
            total_weight = 0
            
            for feature, weight in self.model_state['feature_weights'].items():
                if feature in features:
                    feature_val = features[feature]
                    # Positive feature value = supports home
                    feature_adjustment += (feature_val - 0.5) * weight * 0.1
                    total_weight += weight
            
            if total_weight > 0:
                calibrated += feature_adjustment / total_weight
        
        # Clamp
        final_prob = max(0.01, min(0.99, calibrated))
        
        # Confidence = distance from 0.5
        confidence = abs(final_prob - 0.5) * 2 + 0.5
        
        return final_prob, confidence
    
    def should_bet(self, confidence: float) -> bool:
        """Проверяет, достаточна ли уверенность для ставки"""
        return confidence >= self.model_state['confidence_thresholds']['min_bet']
    
    def get_learning_report(self) -> str:
        """Генерирует отчет об обучении"""
        perf = self.analyzer.calculate_performance(days=30)
        predictions = self.tracker.get_settled(days=30)
        weak_spots = self.analyzer.find_weak_spots(predictions)
        
        lines = [
            "=" * 50,
            "📊 SELF-LEARNING REPORT",
            "=" * 50,
            "",
            f"📅 Period: {perf.period_start} to {perf.period_end}",
            f"📈 Total predictions: {perf.total_predictions}",
            f"✅ Accuracy: {perf.accuracy:.1%}",
            f"📉 Brier Score: {perf.brier_score:.4f}",
            f"🎯 Calibration Error: {perf.calibration_error:.4f}",
            f"📊 Trend: {perf.accuracy_trend}",
            "",
            "Accuracy by Confidence:",
        ]
        
        for bucket, acc in perf.accuracy_by_confidence.items():
            lines.append(f"  {bucket}: {acc:.1%}")
        
        lines.extend([
            "",
            "Current Model State:",
            f"  Calibration factor: {self.model_state['calibration_factor']:.3f}",
            f"  Home advantage: {self.model_state['home_advantage']:.1f}",
            f"  Min bet confidence: {self.model_state['confidence_thresholds']['min_bet']:.1%}",
        ])
        
        if weak_spots:
            lines.extend(["", "⚠️ Weak Spots Found:"])
            for ws in weak_spots[:5]:
                lines.append(f"  - {ws['type']}: {ws['subject']} ({ws['accuracy']:.1%})")
        
        if self.adjustments:
            recent_adj = [a for a in self.adjustments 
                        if (datetime.now() - a.timestamp).days < 7]
            if recent_adj:
                lines.extend(["", "🔧 Recent Adjustments:"])
                for a in recent_adj[-5:]:
                    lines.append(f"  - {a.parameter}: {a.old_value:.3f} → {a.new_value:.3f}")
        
        return "\n".join(lines)
    
    def _save_state(self):
        """Сохраняет состояние модели"""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model_state': self.model_state,
                'adjustments': self.adjustments[-100:]  # Last 100
            }, f)
    
    def _load_state(self):
        """Загружает состояние модели"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model_state.update(data.get('model_state', {}))
                    self.adjustments = data.get('adjustments', [])
            except Exception as e:
                logger.error(f"Error loading model state: {e}")


# === АВТОМАТИЧЕСКИЙ SCHEDULER ===

class LearningScheduler:
    """
    Планировщик автоматического обучения
    """
    
    def __init__(self, learner: SelfLearner):
        self.learner = learner
        self.last_learning = None
        self.learning_interval_hours = 24
    
    def should_learn(self) -> bool:
        """Проверяет, пора ли запускать обучение"""
        if self.last_learning is None:
            return True
        
        elapsed = datetime.now() - self.last_learning
        return elapsed.total_seconds() > self.learning_interval_hours * 3600
    
    def run_if_needed(self) -> Optional[Dict]:
        """Запускает обучение если нужно"""
        if self.should_learn():
            result = self.learner.learn_from_results()
            self.last_learning = datetime.now()
            return result
        return None


# === ТЕСТИРОВАНИЕ ===

if __name__ == "__main__":
    print("=== Тест Self-Learning Module ===\n")
    
    # Инициализация
    tracker = PredictionTracker(storage_path="test_predictions.json")
    learner = SelfLearner(tracker=tracker, model_path="test_model.pkl")
    
    # Симулируем предсказания и результаты
    print("Симуляция 100 предсказаний...")
    
    import random
    random.seed(42)
    
    teams = ["Lakers", "Warriors", "Celtics", "Heat", "Nuggets", "Suns"]
    
    for i in range(100):
        home = random.choice(teams)
        away = random.choice([t for t in teams if t != home])
        
        # Симулируем предсказание
        true_home_prob = random.uniform(0.4, 0.7)
        predicted_prob = true_home_prob + random.uniform(-0.1, 0.1)
        predicted_prob = max(0.3, min(0.7, predicted_prob))
        
        pred_id = tracker.record_prediction(
            game_id=1000 + i,
            home_team=home,
            away_team=away,
            predicted_home_prob=predicted_prob,
            confidence=abs(predicted_prob - 0.5) * 2 + 0.5,
            features={
                'elo_diff': random.uniform(0.3, 0.7),
                'net_rating_diff': random.uniform(0.3, 0.7),
                'rest_advantage': random.uniform(0.4, 0.6)
            }
        )
        
        # Симулируем результат
        home_won = random.random() < true_home_prob
        tracker.record_result(1000 + i, home_won, margin=random.randint(-20, 20))
    
    print(f"Записано предсказаний: {len(tracker.predictions)}")
    print(f"Разрешено: {len(tracker.get_settled())}")
    
    # Запускаем обучение
    print("\n" + "=" * 50)
    print("Запуск цикла обучения...")
    print("=" * 50)
    
    result = learner.learn_from_results()
    
    print(f"\nСтатус: {result['status']}")
    print(f"Accuracy: {result['performance']['accuracy']:.1%}")
    print(f"Brier Score: {result['performance']['brier_score']:.4f}")
    print(f"Calibration Error: {result['performance']['calibration_error']:.4f}")
    print(f"Trend: {result['performance']['trend']}")
    print(f"Adjustments made: {result['adjustments']}")
    
    if result['details']:
        print("\nКорректировки:")
        for d in result['details']:
            print(f"  {d['param']}: {d['change']}")
    
    # Полный отчет
    print("\n" + learner.get_learning_report())
    
    # Тест применения корректировок
    print("\n" + "=" * 50)
    print("Тест применения learned корректировок")
    print("=" * 50)
    
    raw_prob = 0.65
    adjusted_prob, conf = learner.get_adjusted_prediction(
        raw_prob,
        features={'elo_diff': 0.7, 'net_rating_diff': 0.6}
    )
    
    print(f"Raw probability: {raw_prob:.1%}")
    print(f"Adjusted probability: {adjusted_prob:.1%}")
    print(f"Confidence: {conf:.1%}")
    print(f"Should bet: {learner.should_bet(conf)}")
    
    # Cleanup
    import os
    os.remove("test_predictions.json") if os.path.exists("test_predictions.json") else None
    os.remove("test_model.pkl") if os.path.exists("test_model.pkl") else None
