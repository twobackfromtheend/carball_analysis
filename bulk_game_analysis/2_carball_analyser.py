import datetime
import pickle
from collections import OrderedDict, Counter, defaultdict
from pathlib import Path

from typing import Callable, Dict

import pandas as pd
import numpy as np

from additional_stats.x_goals.x_goals_calculator import calculate_x_goals_prediction
from carball.analysis.analysis_manager import AnalysisManager
from carball.analysis.utils.proto_manager import ProtobufManager
from carball.generated.api.game_pb2 import Game
from carball.json_parser.game import Game as JsonParserGame
from bulk_game_analysis.folders import *


def save_replay_data(save_fn: Callable[[pd.DataFrame, Game, JsonParserGame], None],
                     decompiled_output_folder: Path, analysis_output_folder: Path):
    replay_jsons = [file for file in decompiled_output_folder.glob("**/*.json")]

    for replay_json_filepath in replay_jsons:
        try:
            print(replay_json_filepath)
            base_filename = replay_json_filepath.name[:-5]
            proto_filepath: Path = analysis_output_folder / (base_filename + '.proto')
            df_filepath: Path = analysis_output_folder / (base_filename + '.df.xz')
            jsongame_filepath: Path = analysis_output_folder / (base_filename + '.json_game.pkl')

            if proto_filepath.is_file() and df_filepath.is_file() and jsongame_filepath.is_file():
                with proto_filepath.open('rb') as f:
                    proto_game = ProtobufManager.read_proto_out_from_file(f)
                df = pd.read_pickle(df_filepath, compression='xz')
                with jsongame_filepath.open('rb') as f:
                    game = pickle.load(f)
            else:
                game = JsonParserGame()
                game.initialize(file_path=replay_json_filepath, clean_player_names=True)
                analysis = AnalysisManager(game)
                analysis.create_analysis()

                df = analysis.data_frame
                proto_game = analysis.protobuf_game

                with proto_filepath.open('wb') as f:
                    analysis.write_proto_out_to_file(f)

                df.to_pickle(df_filepath, compression='xz')

                with jsongame_filepath.open('wb') as f:
                    pickle.dump(game, f, protocol=pickle.HIGHEST_PROTOCOL)

            save_fn(df, proto_game, game)

        except:
            import traceback
            with open('errors.txt', 'a') as f:
                f.write(str(replay_json_filepath) + '\nTraceback:\n' + traceback.format_exc() + '\n')
        finally:
            try:
                del df
            except NameError:
                pass
            try:
                del proto_game
            except NameError:
                pass
            try:
                del game
            except NameError:
                pass

    finish(analysis_output_folder)


player_order = []
stats_dicts = {}
shots = []

from tensorflow.python.keras.models import load_model
MODEL = load_model(str(XG_MODEL_PATH))


def save_fn(df, proto_game: Game, game: JsonParserGame):
    calculated_xGs = calculate_x_goals_prediction(df, proto_game, model=MODEL)
    # xGs = defaultdict(list)
    # xGas = defaultdict(list)
    # for calculated_xG in calculated_xGs:
    #     if calculated_xG.predicted_xG < 0.01:
    #         continue
    #     xGs[calculated_xG.shooter.id_].append(calculated_xG)
    #     for defender in calculated_xG.defenders:
    #         xGas[defender.id_].append(calculated_xG)
    #
    #     saver = calculated_xG.save
    #     if saver is not None and saver not in calculated_xG.defenders:
    #         xGas[saver.id_].append(calculated_xG)

    shots.append(calculated_xGs)

    demos = Counter()
    for demo_dict in game.demos:
        frame_number = demo_dict['frame_number']
        attacker = demo_dict['attacker'].name
        if not np.isnan(df.loc[frame_number, ('game', 'goal_number')]):
            demos[attacker] += 1

    # KICKOFF GOALS: TODO: MOVE
    kickoff_goals = Counter()
    goal_times = []
    for goal_number in range(len(game.goals)):
        goal_frame_duration = (df[('game', 'delta')][df[('game', 'goal_number')] == goal_number]).sum()
        if goal_frame_duration <= 10.5:
            kickoff_goals[game.goals[goal_number].player_name] += 1
        goal_times.append(f"{goal_frame_duration:.1f}")
    # print(goal_times)

    goal_frame_mask = ~np.isnan(df[('game', 'goal_number')])
    total_game_duration = (df[('game', 'delta')][goal_frame_mask]).sum()
    # total_player_playing_times = {
    #     player.name: (df[('game', 'delta')][~np.isnan(df[(player.name, 'pos_x')]) & goal_frame_mask]).sum()
    #     for player in proto_game.players
    # }

    total_possession = 0
    data_dict = {}
    players = [player for player in proto_game.players if "RLCS" not in player.name]
    for player in players:
        if "RLCS" in player.name:
            continue

        time_in_game = player.time_in_game
        per_five = 300 / time_in_game
        time_in_air = player.stats.positional_tendencies.time_low_in_air \
                      + player.stats.positional_tendencies.time_high_in_air
        data_dict[player.name] = OrderedDict([
            ('goals', player.goals),
            ('assists', player.assists),
            ('shots', player.shots),
            ('saves', player.saves),
            ('demos', demos[player.name]),
            ('kickoff_goals', kickoff_goals[player.name]),
            ('possession_duration', player.stats.possession.possession_time / per_five
             ),
            # ('possession_percentage', None),  # Placeholder
            ('boost_used', player.stats.boost.boost_usage),
            ('boost_per_minute', player.stats.boost.boost_usage / time_in_game * 60),
            ('wasted_usage_percentage', player.stats.boost.wasted_usage / max(1, player.stats.boost.boost_usage)),
            # ('wasted_collection', player.stats.boost.wasted_collection),
            ('num_large_boosts', player.stats.boost.num_large_boosts * per_five),
            ('num_small_boosts', player.stats.boost.num_small_boosts * per_five),
            ('boost_ratio', player.stats.boost.num_small_boosts / max(1, player.stats.boost.num_large_boosts)),
            ('stolen_boosts', player.stats.boost.num_stolen_boosts * per_five),
            ('time_full_boost', player.stats.boost.time_full_boost * per_five),
            ('time_low_boost', player.stats.boost.time_low_boost * per_five),
            ('time_no_boost', player.stats.boost.time_no_boost * per_five),
            ('average_boost_level', player.stats.boost.average_boost_level * 100 / 255.),
            ('average_speed', player.stats.averages.average_speed / 10),
            ('time_high_in_air', player.stats.positional_tendencies.time_high_in_air * per_five),
            ('time_in_air', time_in_air * per_five),
            ('time_on_ground', player.stats.positional_tendencies.time_on_ground * per_five),
            ('time_at_slow_speed', player.stats.speed.time_at_slow_speed * per_five),
            ('time_at_boost_speed', player.stats.speed.time_at_boost_speed * per_five),
            ('time_at_super_sonic', player.stats.speed.time_at_super_sonic * per_five),
            ('time_in_attacking_half', player.stats.positional_tendencies.time_in_attacking_half * per_five),
            ('time_in_defending_half', player.stats.positional_tendencies.time_in_defending_half * per_five),
            ('time_in_attacking_third', player.stats.positional_tendencies.time_in_attacking_third * per_five),
            ('time_in_neutral_third', player.stats.positional_tendencies.time_in_neutral_third * per_five),
            ('time_in_defending_third', player.stats.positional_tendencies.time_in_defending_third * per_five),
            ('time_on_wall', player.stats.positional_tendencies.time_on_wall * per_five),

            ('average_hit_distance', player.stats.averages.average_hit_distance),
            ('ball_hit_forward', player.stats.distance.ball_hit_forward * per_five),
            ('ball_hit_forward_per_hit',
             player.stats.distance.ball_hit_forward / max(1, player.stats.hit_counts.total_hits)),
            ('time_close_to_ball', player.stats.distance.time_close_to_ball * per_five),
            ('time_closest_to_ball', player.stats.distance.time_closest_to_ball * per_five),
            ('time_behind_center_of_mass', player.stats.relative_positioning.time_behind_center_of_mass * per_five),
            ('time_in_front_of_center_of_mass',
             player.stats.relative_positioning.time_in_front_of_center_of_mass * per_five),
            ('time_most_back_player', player.stats.relative_positioning.time_most_back_player * per_five),
            ('time_between_players', player.stats.relative_positioning.time_between_players * per_five),
            ('time_most_forward_player', player.stats.relative_positioning.time_most_forward_player * per_five),

            ('count_of_possessions', player.stats.per_possession_stats.count * per_five),
            ('average_duration_of_possessions', player.stats.per_possession_stats.average_duration),
            ('hits_per_possession', player.stats.per_possession_stats.average_hits),
            ('shots_per_possession', player.stats.per_possession_stats.average_counts.shot),
            ('goals_per_possession', player.stats.per_possession_stats.average_counts.goal),
            ('saves_per_possession', player.stats.per_possession_stats.average_counts.save),
            ('passes_per_possession', player.stats.per_possession_stats.average_counts.pass_),
            ('aerials_per_possession', player.stats.per_possession_stats.average_counts.aerial),
            ('hits', player.stats.hit_counts.total_hits * per_five),
            ('aerials', player.stats.hit_counts.total_aerials * per_five),
            ('aerial_efficiency', player.stats.hit_counts.total_aerials / max(1, time_in_air)),
            ('passes', player.stats.hit_counts.total_passes * per_five),
            ('dribbles', player.stats.hit_counts.total_dribbles * per_five),
            ('hit_goals', player.stats.hit_counts.total_goals * per_five),
            ('hit_shots', player.stats.hit_counts.total_shots * per_five),
            ('hit_saves', player.stats.hit_counts.total_saves * per_five),
            ('turnovers', player.stats.possession.turnovers * per_five),
            ('turnovers_attacking_half', player.stats.possession.turnovers_on_their_half * per_five),
            ('turnovers_defending_half', player.stats.possession.turnovers_on_my_half * per_five),
            ('takeaways', player.stats.possession.won_turnovers * per_five),
            ('is_keyboard', player.stats.controller.is_keyboard),
            # ('xG', total_xGs[player.id.id]),
            # ('average_xG_per_shot', average_xG_per_shots[player.id.id]),
            # ('goals_per_xG', goals_per_xGs[player.id.id]),
            # ('xGa', total_xGas[player.id.id]),
            # ('average_xGa_per_defence', average_xGa_per_defences[player.id.id]),
            # ('goals_against_per_xGa', goals_against_per_xGas[player.id.id]),
        ])
        total_possession += player.stats.possession.possession_time

    for player in players:
        data_dict[player.name][
            'possession_percentage'] = player.stats.possession.possession_time / total_possession * 100

    stats_dicts[proto_game.game_metadata.id] = data_dict, proto_game
    global player_order
    player_order = [player.name for player in sorted(players, key=lambda player: player.is_orange)]


def finish(analysis_output_folder: Path):
    for game_id, (stats_dict, proto_game) in stats_dicts.items():
        game_time: int = proto_game.game_metadata.time
        df = pd.DataFrame.from_dict(stats_dict, orient='index').T
        df.to_csv(str(analysis_output_folder / f'{game_time}.csv'))

    with (analysis_output_folder / "shots.pkl").open('wb') as f:
        pickle.dump(shots, f, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':
    save_replay_data(
        save_fn=save_fn,
        decompiled_output_folder=decompiled_output_folder,
        analysis_output_folder=analysis_output_folder
    )
