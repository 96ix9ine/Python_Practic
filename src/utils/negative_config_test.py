import os
import sys
from pydantic import ValidationError
from configs.config_schema import Lab2Config


def run_negative_test():
    log_path = "reports/LAB2/config_negative_test.log"
    os.makedirs("reports/LAB2", exist_ok=True)

    bad_scenarios = [
        {
            "name": "Случай 1: Неподдерживаемый тип модели (invalid_boost)",
            "patch": {"pipeline": {"model": {"model_type": "invalid_boost"}}},
        },
        {
            "name": "Случай 2: Избыточное количество компонент PCA (>64)",
            "patch": {"pipeline": {"model": {"pca_components": 128}}},
        },
        {
            "name": "Случай 3: Невалидный формат файла данных (требуется .csv)",
            "patch": {"pipeline": {"data_path": "data/raw/digits.txt"}},
        },
    ]

    print("STATUS: Запуск тестирования негативных конфигураций...")

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("=== ЛОГ НЕГАТИВНЫХ ТЕСТОВ КОНФИГУРАЦИИ ===\n\n")

        for scenario in bad_scenarios:
            log_file.write(f"--- {scenario['name']} ---\n")
            try:
                Lab2Config(**scenario["patch"])
                log_file.write(
                    "СТАТУС: ТЕСТ ПРОВАЛЕН. Конфигурация ошибочно признана валидной!\n\n"
                )
            except ValidationError as e:
                log_file.write("СТАТУС: FAILED\n")
                log_file.write("ОБЪЯСНЕНИЕ ОШИБКИ ВАЛИДАЦИИ:\n")
                log_file.write(str(e) + "\n\n")
                print(f"SUCCESS: Сценарий '{scenario['name']}' успешно заблокирован.")

    print(f"STATUS: Тестирование завершено. Лог сохранен в: {log_path}")
    sys.exit(0)


if __name__ == "__main__":
    run_negative_test()
