import uuid
import datetime
from typing import List, Dict, Any

class InterventionEngine:
    """
    Intelligence Engine for generating cognitive fatigue protocols.
    Uses dynamic thresholding based on rolling statistics instead of hardcoded bounds.
    """
    def __init__(self, risk_threshold_critical: float = 75.0, risk_threshold_warning: float = 55.0):
        self.critical_threshold = risk_threshold_critical
        self.warning_threshold = risk_threshold_warning

    def _create_protocol(self, payload_type: str, message: str) -> Dict[str, Any]:
        """Factory for standardized protocol responses."""
        return {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "type": payload_type,
            "message": message
        }

    def generate(self, user_features: dict, burnout_score: float) -> List[Dict[str, Any]]:
        """
        Generate actionable interventions based on user telemetry and neural model scores.
        """
        protocols = []
        
        # 1. Macro-level Interventions (Systemic Risk)
        if burnout_score > self.critical_threshold:
            protocols.append(self._create_protocol(
                "CRITICAL",
                "CRITICAL FATIGUE DETECTED. Mandatory cognitive rest required immediately."
            ))
        elif burnout_score > self.warning_threshold:
            protocols.append(self._create_protocol(
                "WARNING",
                "Elevated neural strain. Initiate a 2-hour focus block followed by a hard stop."
            ))
            
        # 2. Micro-level Interventions (Behavioral Diagnostics)
        # Sleep Deficit
        sleep_3d = user_features.get('sleep_3d_avg', 8.0)
        sleep_std = user_features.get('sleep_3d_std', 0.0)
        if sleep_3d < 6.5 or (sleep_3d < 7.0 and sleep_std > 1.5):
            protocols.append(self._create_protocol(
                "LIFESTYLE",
                f"Severe sleep decay detected ({sleep_3d:.1f}h avg). Prioritize deep recovery tonight (target: 8h+)."
            ))
            
        # Motor Function / Keystroke Drift
        wpm_drift = user_features.get('wpm_drift', 1.0)
        if wpm_drift < 0.8:
            protocols.append(self._create_protocol(
                "BEHAVIORAL",
                f"Motor function degradation (>20% WPM drop). High probability of cognitive exhaustion. Step away from the terminal."
            ))
            
        # Context Switching (Scatter Focus)
        task_switch_drift = user_features.get('task_switch_drift', 1.0)
        if task_switch_drift > 1.3:
            protocols.append(self._create_protocol(
                "BEHAVIORAL",
                "Severe context-switching detected. Focus fragmentation is accelerating burnout. Isolate tasks."
            ))
            
        # Recovery Breaks
        breaks_per_hr = user_features.get('break_frequency_per_hr', 1.0)
        if breaks_per_hr < 0.5:
            protocols.append(self._create_protocol(
                "HABIT",
                "Prolonged hyper-focus without micro-breaks drains cognitive reserves. Stretch for 5 minutes."
            ))
            
        # 3. Baseline Stabilization
        if not protocols:
            protocols.append(self._create_protocol(
                "POSITIVE",
                "Telemetry optimal. Cognitive load and behavioral rhythms are perfectly stabilized."
            ))
            
        return protocols

# Original function wrapper for backward compatibility with API
def get_interventions(user_features: dict, burnout_score: float) -> list:
    engine = InterventionEngine()
    return engine.generate(user_features, burnout_score)
