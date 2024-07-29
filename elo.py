import math

def calculate_elo(player_rating, opponent_rating, result, k=32):
    """
    player_rating: 현재 플레이어의 레이팅
    opponent_rating: 상대 플레이어의 레이팅
    result: 경기 결과 (1: 승리, 0.5: 무승부, 0: 패배)
    k: K-팩터 (변동 상수)
    """
    if result == 1:
        result1 = 1
        result2 = 0
    elif result == 0:
        result1 = 0
        result2 = 1
    else:
        result1 = 0.5
        result2 = 0.5
    expected_score1 = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
    new_rating1 = player_rating + k * (result1 - expected_score1)

    expected_score2 = 1 / (1 + 10 ** ((player_rating - opponent_rating) / 400))
    new_rating2 = opponent_rating + k * (result2 - expected_score2)
    
    return new_rating1, new_rating2


