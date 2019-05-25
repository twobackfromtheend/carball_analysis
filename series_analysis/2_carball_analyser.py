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
from series_analysis.folders import *


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
summary_name = "summary"


def save_fn(df, proto_game: Game, game: JsonParserGame):
    # calculated_xGs = calculate_x_goals_prediction(df, proto_game, model_path=str(XG_MODEL_PATH))
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
    #
    # total_xGs = defaultdict(
    #     lambda: 0,
    #     **{player_id: sum(_xG.predicted_xG for _xG in _xGs)
    #        for player_id, _xGs in xGs.items()}
    # )
    # goals_per_xGs = defaultdict(
    #     lambda: 0,
    #     **{player_id: sum(_xG.is_goal for _xG in _xGs) / total_xGs[player_id]
    #        for player_id, _xGs in xGs.items()}
    # )
    # average_xG_per_shots = defaultdict(
    #     lambda: 0,
    #     **{player_id: total_xGs[player_id] / len(_xGs)
    #        for player_id, _xGs in xGs.items()}
    # )
    # total_xGas = defaultdict(
    #     lambda: 0,
    #     **{player_id: sum(_xGa.predicted_xG for _xGa in _xGas) for player_id, _xGas in xGas.items()}
    # )
    # goals_against_per_xGas = defaultdict(
    #     lambda: 0,
    #     **{player_id: sum(_xGa.is_goal for _xGa in _xGas) / total_xGas[player_id]
    #        for player_id, _xGas in xGas.items()}
    # )
    # average_xGa_per_defences = defaultdict(
    #     lambda: 0,
    #     **{player_id: total_xGas[player_id] / len(_xGas)
    #        for player_id, _xGas in xGas.items()}
    # )

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
    for player in proto_game.players:
        data_dict[player.name] = OrderedDict([
            ('goals', player.goals),
            ('assists', player.assists),
            ('shots', player.shots),
            ('saves', player.saves),
            ('demos', demos[player.name]),
            ('kickoff_goals', kickoff_goals[player.name]),
            ('possession_duration', player.stats.possession.possession_time),
            ('possession_percentage', None),  # Placeholder
            ('boost_used', player.stats.boost.boost_usage),
            ('boost_per_minute', player.stats.boost.boost_usage / total_game_duration * 60),
            ('wasted_usage', player.stats.boost.wasted_usage),
            ('wasted_collection', player.stats.boost.wasted_collection),
            ('stolen_boosts', player.stats.boost.num_stolen_boosts),
            ('time_full_boost', player.stats.boost.time_full_boost),
            ('time_low_boost', player.stats.boost.time_low_boost),
            ('time_no_boost', player.stats.boost.time_no_boost),
            ('average_boost_level', player.stats.boost.average_boost_level * 100 / 255.),
            ('average_speed', player.stats.averages.average_speed / 10),
            ('time_high_in_air', player.stats.positional_tendencies.time_high_in_air),
            ('time_in_air', player.stats.positional_tendencies.time_low_in_air + player.stats.positional_tendencies.time_high_in_air),
            ('time_on_ground', player.stats.positional_tendencies.time_on_ground),
            ('time_at_slow_speed', player.stats.speed.time_at_slow_speed),
            ('time_at_boost_speed', player.stats.speed.time_at_boost_speed),
            ('time_at_super_sonic', player.stats.speed.time_at_super_sonic),
            ('time_in_attacking_half', player.stats.positional_tendencies.time_in_attacking_half),
            ('time_in_defending_half', player.stats.positional_tendencies.time_in_defending_half),
            ('time_in_attacking_third', player.stats.positional_tendencies.time_in_attacking_third),
            ('time_in_neutral_third', player.stats.positional_tendencies.time_in_neutral_third),
            ('time_in_defending_third', player.stats.positional_tendencies.time_in_defending_third),
            ('time_on_wall', player.stats.positional_tendencies.time_on_wall),
            ('count_of_possessions', player.stats.per_possession_stats.count),
            ('average_duration_of_possessions', player.stats.per_possession_stats.average_duration),
            ('hits_per_possession', player.stats.per_possession_stats.average_hits),
            ('shots_per_possession', player.stats.per_possession_stats.average_counts.shot),
            ('goals_per_possession', player.stats.per_possession_stats.average_counts.goal),
            ('saves_per_possession', player.stats.per_possession_stats.average_counts.save),
            ('passes_per_possession', player.stats.per_possession_stats.average_counts.pass_),
            ('aerials_per_possession', player.stats.per_possession_stats.average_counts.aerial),
            ('hits', player.stats.hit_counts.total_hits),
            ('aerials', player.stats.hit_counts.total_aerials),
            ('passes', player.stats.hit_counts.total_passes),
            ('dribbles', player.stats.hit_counts.total_dribbles),
            ('hit_goals', player.stats.hit_counts.total_goals),
            ('hit_shots', player.stats.hit_counts.total_shots),
            ('hit_saves', player.stats.hit_counts.total_saves),
            ('turnovers', player.stats.possession.turnovers),
            ('turnovers_attacking_half', player.stats.possession.turnovers_on_their_half),
            ('turnovers_defending_half', player.stats.possession.turnovers_on_my_half),
            ('takeaways', player.stats.possession.won_turnovers),
            ('is_keyboard', player.stats.controller.is_keyboard),
            # ('xG', total_xGs[player.id.id]),
            # ('average_xG_per_shot', average_xG_per_shots[player.id.id]),
            # ('goals_per_xG', goals_per_xGs[player.id.id]),
            # ('xGa', total_xGas[player.id.id]),
            # ('average_xGa_per_defence', average_xGa_per_defences[player.id.id]),
            # ('goals_against_per_xGa', goals_against_per_xGas[player.id.id]),
        ])
        total_possession += player.stats.possession.possession_time

    for player in proto_game.players:
        data_dict[player.name][
            'possession_percentage'] = player.stats.possession.possession_time / total_possession * 100

    stats_dicts[proto_game.game_metadata.id] = data_dict, proto_game
    global player_order
    player_order = [player.name for player in sorted(proto_game.players, key=lambda player: player.is_orange)]


def finish(analysis_output_folder: Path):
    # import xlsxwriter  # Writer used. pip install xlsxwriter if necessary.
    dfs = {game_id: pd.DataFrame.from_dict(stats_dict, orient='index').T for game_id, (stats_dict, proto_game) in
           stats_dicts.items()}

    # Arrange by datetime
    game_times = {game_id: proto_game.game_metadata.time for game_id, (stats_dict, proto_game) in stats_dicts.items()}

    game_names = {game_id: proto_game.game_metadata.name for game_id, (stats_dict, proto_game) in stats_dicts.items()}

    def _key(dfs_item):
        game_id, df = dfs_item
        return game_times.get(game_id, 500)

    dfs = add_summary_series(dfs)

    i = 1
    with pd.ExcelWriter(str(analysis_output_folder / f'{FOLDER_TO_ANALYSE}_stats.xlsx'),
                        engine='xlsxwriter') as writer:
        workbook = writer.book
        stat_name_format = workbook.add_format({'bold': True, 'align': 'right'})
        stats_format = workbook.add_format({'num_format': '#,##0.00'})
        player_name_format = workbook.add_format({'bold': True, 'align': 'center'})
        for game_id, df in sorted(dfs.items(), key=_key):
            if game_id != summary_name:
                game_name = f"Game {i} ({game_names[game_id]})"
                print(game_name, game_id, datetime.datetime.fromtimestamp(game_times[game_id]))
                i += 1
            else:
                game_name = summary_name

            df = df.reindex(player_order, axis=1)

            import pandas.io.formats.excel
            pandas.io.formats.excel.header_style = None

            df.to_excel(writer, sheet_name=game_name)
            worksheet = writer.sheets[game_name]

            worksheet.set_column(0, 0, 30, stat_name_format)
            worksheet.set_column(1, len(df.columns), 12, stats_format)
            worksheet.set_row(0, cell_format=player_name_format)
            if game_id != summary_name:
                for row in range(32, 40):
                    worksheet.write_comment(f"A{row}",
                                            "a possession is a sequence of consecutive (2 or more) hits, whose total duration is above 1s")

            for row in range(2, len(df) + 2):
                worksheet.conditional_format(f'A{row}:XFD{row}', {'type': '3_color_scale'})


def add_summary_series(dfs: Dict[str, pd.DataFrame]):
    player_names = list(dfs.values())[0].columns.values
    summary_dict = {}
    number_of_games = len(dfs)
    for player_name in player_names:
        summary_dict[player_name] = OrderedDict([
            ('total_goals', sum(df[player_name].goals for df in dfs.values())),
            ('total_assists', sum(df[player_name].assists for df in dfs.values())),
            ('total_shots', sum(df[player_name].shots for df in dfs.values())),
            ('total_saves', sum(df[player_name].saves for df in dfs.values())),
            ('total_demos', sum(df[player_name].demos for df in dfs.values())),
            ('total_kickoff_goals', sum(df[player_name].kickoff_goals for df in dfs.values())),
            ('average_possession_duration',
             sum(df[player_name].possession_duration for df in dfs.values()) / number_of_games),
            ('average_possession_percentage',
             sum(df[player_name].possession_percentage for df in dfs.values()) / number_of_games),
            ('total_boost_used', sum(df[player_name].boost_used for df in dfs.values())),
            ('average_boost_per_minute',
             sum(df[player_name].boost_per_minute for df in dfs.values()) / number_of_games),
            ('total_wasted_usage', sum(df[player_name].wasted_usage for df in dfs.values())),
            ('total_wasted_collection', sum(df[player_name].wasted_collection for df in dfs.values())),

            ('total_stolen_boost', sum(df[player_name].stolen_boosts for df in dfs.values())),

            ('average_average_boost_level',
             sum(df[player_name].average_boost_level for df in dfs.values()) / number_of_games),
            ('average_average_speed',
             sum(df[player_name].average_speed for df in dfs.values()) / number_of_games),
            ('total_time_high_in_air',
             sum(df[player_name].time_high_in_air for df in dfs.values())),
            ('total_time_in_air',
             sum(df[player_name].time_in_air for df in dfs.values())),
            ('total_time_on_ground',
             sum(df[player_name].time_on_ground for df in dfs.values())),
            ('average_time_at_slow_speed',
             sum(df[player_name].time_at_slow_speed for df in dfs.values()) / number_of_games),
            ('average_time_at_boost_speed',
             sum(df[player_name].time_at_boost_speed for df in dfs.values()) / number_of_games),
            ('average_time_at_super_sonic',
             sum(df[player_name].time_at_super_sonic for df in dfs.values()) / number_of_games),
            ('average_time_in_attacking_half',
             sum(df[player_name].time_in_attacking_half for df in dfs.values()) / number_of_games),
            ('average_time_in_defending_half',
             sum(df[player_name].time_in_defending_half for df in dfs.values()) / number_of_games),
            ('average_time_in_attacking_third',
             sum(df[player_name].time_in_attacking_third for df in dfs.values()) / number_of_games),
            ('average_time_in_neutral_third',
             sum(df[player_name].time_in_neutral_third for df in dfs.values()) / number_of_games),
            ('average_time_in_defending_third',
             sum(df[player_name].time_in_defending_third for df in dfs.values()) / number_of_games),
            ('average_time_on_wall',
             sum(df[player_name].time_on_wall for df in dfs.values()) / number_of_games),

            ('total_count_of_possessions', sum(df[player_name].count_of_possessions for df in dfs.values())),
            ('average_average_duration_of_possessions',
             sum(df[player_name].average_duration_of_possessions for df in dfs.values()) / number_of_games),
            ('average_shots_per_possession',
             sum(df[player_name].shots_per_possession for df in dfs.values()) / number_of_games),
            ('average_goals_per_possession',
             sum(df[player_name].goals_per_possession for df in dfs.values()) / number_of_games),
            ('average_saves_per_possession',
             sum(df[player_name].saves_per_possession for df in dfs.values()) / number_of_games),
            ('average_passes_per_possession',
             sum(df[player_name].passes_per_possession for df in dfs.values()) / number_of_games),

            # ('total_xG',
            #  sum(df[player_name].xG for df in dfs.values())),
            # ('average_average_xG_per_shot',
            #  sum(df[player_name].average_xG_per_shot for df in dfs.values()) / number_of_games),
            # ('average_goals_per_xGs',
            #  sum(df[player_name].goals_per_xG for df in dfs.values()) / number_of_games),
            # ('total_xGa',
            #  sum(df[player_name].xGa for df in dfs.values())),
            # ('average_average_xGa_per_defence',
            #  sum(df[player_name].average_xGa_per_defence for df in dfs.values()) / number_of_games),
            # ('average_goals_against_per_xGa',
            #  sum(df[player_name].goals_against_per_xGa for df in dfs.values()) / number_of_games),
        ])

    dfs[summary_name] = pd.DataFrame.from_dict(summary_dict, orient='index').T
    return dfs


if __name__ == '__main__':
    save_replay_data(
        save_fn=save_fn,
        decompiled_output_folder=decompiled_output_folder,
        analysis_output_folder=analysis_output_folder
    )
