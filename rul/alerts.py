from rul.config import RED_BELOW, AMBER_BELOW


def get_alert_level(rul: float) -> tuple[str, str]:
    # returns (level, message) for a predicted RUL
    if rul < RED_BELOW:
        return 'RED', 'Immediate maintenance required'
    if rul < AMBER_BELOW:
        return 'AMBER', 'Schedule maintenance soon'
    return 'GREEN', 'Engine healthy'
