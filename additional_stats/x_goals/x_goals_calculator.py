from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import pandas as pd
from tensorflow.python.keras.models import load_model

from carball.generated.api.game_pb2 import Game
from ..utils.utils import normalise_df, flip_teams
from ..x_goals.x_goals import get_input_and_output_from_game_datas


class PlayerIdAndName(NamedTuple):
    id_: str
    name: str


@dataclass
class CalculatedXG:
    is_goal: bool
    predicted_xG: float
    frame: int
    seconds_remaining: float
    defenders: List[PlayerIdAndName]
    shooter: PlayerIdAndName
    save: Optional[PlayerIdAndName]
    shooter_is_orange: bool
    data: pd.Series


def calculate_x_goals_prediction(df: pd.DataFrame, proto: Game, model_path: str = None, model=None) -> List[CalculatedXG]:
    if model is None:
        MODEL = load_model(model_path)
    else:
        MODEL = model

    normalised_df = normalise_df(df)
    input_, output, hits_order = get_input_and_output_from_game_datas(normalised_df, proto)

    predicted_x_goals = MODEL.predict(input_).flatten()
    output = output.flatten()

    hit_players = find_shooters_and_defenders(df, proto)

    hits = proto.game_stats.hits
    hit_frame_numbers = {hit.frame_number: hit for hit in hits}
    player_id_to_player = {
        player.id.id: player
        for player in proto.players
    }

    calculated_xGs = []
    for i, _hit_players in enumerate(hit_players):
        frame = _hit_players.frame
        seconds_remaining = df.loc[frame, ('game', 'seconds_remaining')]
        if ('game', 'is_overtime') in df:
            is_overtime = df.loc[frame, ('game', 'is_overtime')]
            if is_overtime:
                seconds_remaining *= -1
        is_goal = output[i]
        predicted_xG = predicted_x_goals[i]
        defenders = _hit_players.defenders
        shooter = _hit_players.shooter
        save = None
        data = df.loc[frame]
        shooter_is_orange = _hit_players.shooter_is_orange

        hit = hit_frame_numbers[frame]
        try:
            next_hit = hit_frame_numbers[hit.next_hit_frame_number]
            if next_hit.save:
                save = PlayerIdAndName(next_hit.player_id.id, player_id_to_player[next_hit.player_id.id].name)
        except KeyError:
            pass

        calculated_xG = CalculatedXG(
            is_goal=is_goal, predicted_xG=predicted_xG, frame=frame, seconds_remaining=seconds_remaining,
            defenders=defenders, shooter=shooter, save=save, shooter_is_orange=shooter_is_orange, data=data
        )
        calculated_xGs.append(calculated_xG)

    return calculated_xGs


@dataclass
class HitPlayers:
    frame: int
    shooter: PlayerIdAndName
    shooter_is_orange: bool
    defenders: List[PlayerIdAndName]


def find_shooters_and_defenders(df: pd.DataFrame, proto: Game) -> List[HitPlayers]:
    # name_team_map = {player.name: player.is_orange for player in proto.players}
    player_id_to_player = {
        player.id.id: player
        for player in proto.players
    }
    teams = {
        0: [player for player in proto.players if not player.is_orange],
        1: [player for player in proto.players if player.is_orange]
    }

    df_orange = flip_teams(df)

    hits = proto.game_stats.hits

    # hit_frame_numbers = np.array([hit.frame_number for hit in hits if hit.shot])
    # hit_frames = df.loc[hit_frame_numbers, (slice(None), ['pos_x', 'pos_y', 'pos_z'])]

    hit_players = []
    for hit in hits:
        if not hit.shot:
            continue
        shooter = player_id_to_player[hit.player_id.id]
        # SET DEFENDER TO ORANGE TEAM
        if shooter.is_orange:
            _df = df_orange
        else:
            _df = df

        # Find defenders behind ball
        defending_team = teams[0 if shooter.is_orange else 1]

        # hit_players[hit.frame_number] = [shooter.name, _df.loc[hit.frame_number, ('game', 'seconds_remaining')]]
        shooter_id_and_name = PlayerIdAndName(shooter.id.id, shooter.name)
        shooter_is_orange = shooter.is_orange
        defenders = []
        for defender in defending_team:
            ball_y = _df.loc[hit.frame_number, ('ball', 'pos_y')]
            defender_y = _df.loc[hit.frame_number, (defender.name, 'pos_y')]
            ball_x = _df.loc[hit.frame_number, ('ball', 'pos_x')]
            defender_x = _df.loc[hit.frame_number, (defender.name, 'pos_x')]
            if defender_y > ball_y and (
                    abs(defender_x) < 900 or (-800 < defender_x < ball_x) or (ball_x < defender_x < 800)):
                defenders.append(PlayerIdAndName(defender.id.id, defender.name))
        _hit_players = HitPlayers(hit.frame_number, shooter_id_and_name, shooter_is_orange, defenders)
        hit_players.append(_hit_players)
    return hit_players
