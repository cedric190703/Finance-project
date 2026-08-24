"""Market conventions: day counts, business-day calendars, tenors and schedules.

Nothing here is intellectually hard, and everything here is easy to get subtly
wrong. Isolating it in one package means the rest of the engine never has to
guess whether a year has 360 or 365 days in it.
"""

from aegis.conventions.calendars import (
    NYSE,
    TARGET,
    WEEKENDS_ONLY,
    BusinessCalendar,
    JointCalendar,
    easter_sunday,
)
from aegis.conventions.daycount import DayCount
from aegis.conventions.schedule import (
    BusinessDayConvention,
    CalendarLike,
    Frequency,
    StubConvention,
    adjust,
    generate_schedule,
)
from aegis.conventions.tenor import Tenor, TenorUnit

__all__ = [
    "NYSE",
    "TARGET",
    "WEEKENDS_ONLY",
    "BusinessCalendar",
    "BusinessDayConvention",
    "CalendarLike",
    "DayCount",
    "Frequency",
    "JointCalendar",
    "StubConvention",
    "Tenor",
    "TenorUnit",
    "adjust",
    "easter_sunday",
    "generate_schedule",
]
