import os
import time
import json
from typing import Dict


class StatefulAlertManager:
    """Управляемый менеджер алертов со встроенным подавлением повторов (Deduplication)"""

    def __init__(self, log_path: str = "reports/LAB8/alerts_sample.jsonl"):
        self.log_path = log_path
        self.last_alerts: Dict[str, str] = {}
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def trigger_alert(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        condition: str,
        level: str,
    ) -> None:
        """Проверяет порог и записывает алерт, если состояние изменилось или повтор отсутствует"""
        is_triggered = False
        if condition == "greater" and value > threshold:
            is_triggered = True
        elif condition == "less" and value < threshold:
            is_triggered = True

        if is_triggered:
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

            if self.last_alerts.get(metric_name) == level:
                return

            self.last_alerts[metric_name] = level

            alert_entry = {
                "timestamp": current_time,
                "metric": metric_name,
                "value": round(value, 4),
                "threshold": threshold,
                "level": level,
                "action": (
                    "SUPPRESS_REPEATS_ACTIVE"
                    if level == "WARNING"
                    else "FORCE_REDEPLOY"
                ),
            }

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_entry, ensure_ascii=False) + "\n")
            print(
                f"ALERT [{level}]: Метрика {metric_name} = {value} нарушила порог {threshold}!"
            )
        else:
            if metric_name in self.last_alerts:
                del self.last_alerts[metric_name]
