"""
Тест импорта nba_api
"""
print("Testing nba_api import...")

try:
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    print("✅ live_scoreboard imported")
except Exception as e:
    print(f"❌ live_scoreboard: {e}")

try:
    from nba_api.stats.endpoints import leaguegamefinder
    print("✅ leaguegamefinder imported")
except Exception as e:
    print(f"❌ leaguegamefinder: {e}")

try:
    from nba_api.stats.static import teams as nba_teams
    print("✅ teams imported")
    print(f"   Found {len(nba_teams.get_teams())} teams")
except Exception as e:
    print(f"❌ teams: {e}")

print("\nTrying to fetch today's games (LIVE API)...")
try:
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    
    sb = live_scoreboard.ScoreBoard()
    data = sb.get_dict()
    
    games = data.get('scoreboard', {}).get('games', [])
    print(f"✅ Found {len(games)} games today")
    
    for game in games[:5]:
        home = game.get('homeTeam', {})
        away = game.get('awayTeam', {})
        status = game.get('gameStatusText', '')
        print(f"   {away.get('teamName')} @ {home.get('teamName')} - {status}")
        
except Exception as e:
    print(f"❌ Error fetching games: {e}")

print("\n✅ Test complete")
