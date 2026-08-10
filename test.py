import nflreadpy as nfl

schedules = nfl.load_schedules(seasons = [2025])
print(schedules.columns)
print(schedules.head())

pbp = nfl.load_pbp(seasons = [2025])
print(pbp.columns)
print(pbp.head)