import pickle
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.plotly as py

from additional_stats.x_goals.x_goals_calculator import CalculatedXG, PlayerIdAndName
from bulk_game_analysis.folders import *
from bulk_game_analysis.tournament_data import *
from additional_stats.utils import plotly_helpers
from additional_stats.utils.file_handling import get_game_name
from additional_stats.utils.utils import flip_teams


def rename_scheme(id_and_name: PlayerIdAndName) -> PlayerIdAndName:
    return PlayerIdAndName(id_and_name.id_, PLAYER_NAME_ALTERNATES.get(id_and_name.name, id_and_name.name))


def read_shots_data(analysis_folder: Path):
    # description_dict_paths = []
    # for root, dirs, files in os.walk(analysis_folder):
    #     for file in files:
    #         if file.endswith('.pkl'):
    #             filepath = os.path.join(root, file)
    #             description_dict_paths.append(filepath)
    # description_dicts = {}
    # for description_dict_path in description_dict_paths:
    #     with open(description_dict_path, 'rb') as f:
    #         description_dict = pickle.load(f)
    #     description_dicts[os.path.basename(description_dict_path)] = description_dict

    shots_data_path = analysis_folder / "shots.pkl"
    with shots_data_path.open('rb') as f:
        shots_data: List[List[CalculatedXG]] = pickle.load(f)

    # Rename players from alternate names
    for c_xgs in shots_data:
        for c_xg in c_xgs:
            c_xg.defenders = [rename_scheme(id_and_name) for id_and_name in c_xg.defenders]
            c_xg.shooter = rename_scheme(c_xg.shooter)
            c_xg.save = rename_scheme(c_xg.save) if c_xg.save is not None else None

            # Rename columns in data
            new_columns = []
            for index in c_xg.data.index:
                player_name = index[0]
                new_player_name = PLAYER_NAME_ALTERNATES.get(player_name, player_name)
                new_columns.append((new_player_name, index[1]))
            c_xg.data.index = pd.MultiIndex.from_tuples(new_columns)

    # Count matches
    matches = defaultdict(lambda: 0)
    for c_xgs in shots_data:
        players = set()

        for c_xg in c_xgs:
            players.update([defender.name for defender in c_xg.defenders])
            players.add(c_xg.shooter.name)
            if c_xg.save is not None:
                players.add(c_xg.save.name)

        if len(players) != 6:
            print(f"Found {len(players)} players: {players}")
        for player in players:
            matches[player] += 1

    return shots_data, matches



def analyse(analysis_folder: Path):
    shots_data, matches = read_shots_data(analysis_folder)

    xGs = defaultdict(lambda: [[], 0])
    xGas = defaultdict(lambda: [[], 0])

    for c_xgs in shots_data:
        for c_xg in c_xgs:
            if not c_xg.is_goal and c_xg.predicted_xG < 0.01:
                continue
            xGs[c_xg.shooter.name][0].append(c_xg.predicted_xG)
            if c_xg.is_goal:
                xGs[c_xg.shooter.name][1] += 1

            __defenders = [defender.name for defender in c_xg.defenders]
            if c_xg.save is not None:
                __defenders.append(c_xg.save)
            _defenders = set(__defenders)

            for defender in _defenders:
                xGas[defender][0].append(c_xg.predicted_xG)
                if c_xg.is_goal:
                    xGas[defender][1] += 1

    goal_per_xG = {player: _xGs[1] / sum(_xGs[0]) for player, _xGs in xGs.items()}
    average_xG_per_shot = {player: sum(_xGs[0]) / len(_xGs[0]) for player, _xGs in xGs.items()}
    xG_per_match = {player: sum(_xGs[0]) / matches[player] for player, _xGs in xGs.items()}

    goal_against_per_xGa = {player: _xGas[1] / sum(_xGas[0]) for player, _xGas in xGas.items()}
    average_xGa_per_defence = {player: sum(_xGas[0]) / len(_xGas[0]) for player, _xGas in xGas.items()}
    xGa_per_match = {player: sum(_xGas[0]) / matches[player] for player, _xGas in xGas.items()}

    goals = {player: _xGs[1] for player, _xGs in xGs.items()}
    goals_against = {player: _xGas[1] for player, _xGas in xGas.items()}
    data_dict = {
        "matches": matches,
        "goals": goals,
        "goal_per_xG": goal_per_xG,
        "average_xG_per_shot": average_xG_per_shot,
        "xG_per_match": xG_per_match,
        "goals_against": goals_against,
        "goal_against_per_xGa": goal_against_per_xGa,
        "average_xGa_per_defence": average_xGa_per_defence,
        "xGa_per_match": xGa_per_match,
        # "team": {player: player_to_team[player] for player in xGs.keys()}
    }

    xG_df = pd.DataFrame.from_dict(data_dict)
    xG_df.dropna().to_csv('xG_df.csv')
    pass


def visualise_shooter_positions(analysis_folder: str, save_fig_folder: str):
    shotmap_folder = "shot_charts"
    full_df, datas, matches = read_description_dicts(analysis_folder)

    shooting_positions = defaultdict(list)

    for filename, _df in datas.items():
        datas[filename] = (_df, flip_teams(_df))

    for row in full_df.itertuples():
        filename = row.Index[0]
        df_blue, df_orange = datas[filename]
        df = df_orange if row.shooter_is_orange else df_blue

        shooter_x = df.loc[row.frame, (row.shooter, 'pos_x')]
        shooter_y = df.loc[row.frame, (row.shooter, 'pos_y')]
        shooter_z = df.loc[row.frame, (row.shooter, 'pos_z')]
        shooting_positions[row.shooter].append(((shooter_x, shooter_y, shooter_z), row))

    # fig = plt.figure(figsize=(8, 7))

    for player, _data in shooting_positions.items():
        print(player)

        _shooting_positions, rows = zip(*_data)
        _xs, _ys, _zs = zip(*_shooting_positions)
        _xs = np.array(_xs)
        _ys = np.array(_ys)
        _zs = np.array(_zs)

        is_goals = np.array([row.is_goal for row in rows], dtype=bool)

        # cmap = 'viridis'
        cmap = 'inferno'
        xGs = np.array([row.predicted_xG for row in rows])

        fig = plt.figure(figsize=(8, 7))

        draw_map_z()

        # plt.scatter(_xs[is_goals], _ys[is_goals], c=colours[is_goals], marker='o', label='Goals')
        # plt.scatter(_xs[~is_goals], _ys[~is_goals], c=colours[~is_goals], marker='x', label='Not goals')
        alpha = 1
        plt.scatter(_xs[~is_goals], _ys[~is_goals], c=xGs[~is_goals], cmap=cmap, vmin=0, vmax=1, marker='x',
                    label='Not goals', alpha=alpha)
        plt.scatter(_xs[is_goals], _ys[is_goals], c=xGs[is_goals], cmap=('%s' % cmap), vmin=0, vmax=1, marker='o',
                    label='Goals', alpha=alpha)

        plt.title(player)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(save_fig_folder, shotmap_folder, player + '_shots.png'), dpi=300)
        plt.close(fig)
        # plt.show()

    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_fig_folder, shotmap_folder, '_everyone' + '_shots.png'), dpi=300)


def visualise_shot_time(analysis_folder: str, save_fig_folder: str):
    full_df, datas, matches = read_description_dicts(analysis_folder)

    shots = defaultdict(list)
    for row in full_df.itertuples():
        if not row.is_goal and row.predicted_xG < 0.01:
            continue
        if not np.isnan(row.seconds_remaining) and row.seconds_remaining > 0:
            shots[row.shooter].append((300 - row.seconds_remaining, row.predicted_xG, row.is_goal))

    # fig = plt.figure(figsize=(10, 6))
    # for player, _shots in shots.items():
    #     _xs, _ys, _is_goals = zip(*_shots)
    #     _xs = np.array(_xs)
    #     _ys = np.array(_ys)
    #     _is_goals = np.array(_is_goals, dtype=bool)
    #     alpha = 0.6
    #     plt.scatter(_xs[~_is_goals], _ys[~_is_goals], marker='x', label=player, alpha=alpha)
    #     plt.scatter(_xs[_is_goals], _ys[_is_goals], marker='o', alpha=alpha)
    # plt.show()

    fig = plt.figure(figsize=(10, 6))
    xs = []
    ys = []
    is_goals = []
    for player, _shots in shots.items():
        _xs, _ys, _is_goals = zip(*_shots)
        xs.append(_xs)
        ys.append(_ys)
        is_goals.append(_is_goals)
    xs = np.concatenate(xs).flatten()
    ys = np.concatenate(ys).flatten()
    is_goals = np.concatenate(is_goals).flatten().astype(np.bool)

    sorting_order = np.argsort(xs)
    xs = xs[sorting_order]
    ys = ys[sorting_order]
    is_goals = is_goals[sorting_order]
    # plt.hist([xs[is_goals], xs[~is_goals]], bins=np.arange(300), stacked=True, label=['Goals', 'Not goals'])
    # plt.xlim((0, 300))
    # plt.ylim((0, 20))
    # plt.legend()
    # plt.xlabel('Time elapsed (s)')
    # plt.ylabel('Count')
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_fig_folder, 'goals_histogram.png'), dpi=300)
    # plt.show()
    # return

    polynomial_coefficients, covariance = np.polyfit(xs, ys, 1, cov=True)
    polynomial = np.poly1d(polynomial_coefficients)
    plt.plot(np.unique(xs), polynomial(np.unique(xs)), '-',
             label=f'xG (slope: {polynomial_coefficients[0] * 60 * 100:.2f}% per minute)')
    print(polynomial_coefficients, np.sqrt(np.diag(covariance)))

    polynomial_coefficients = np.polyfit(xs, is_goals, 1)
    polynomial = np.poly1d(polynomial_coefficients)
    plt.plot(np.unique(xs), polynomial(np.unique(xs)), '-',
             label=f'is_goals (slope: {polynomial_coefficients[0] * 60 * 100:.2f}% per minute)')
    print(polynomial_coefficients, np.sqrt(np.diag(covariance)))

    def get_moving_average(xs: np.ndarray, ys: np.ndarray, x: float, range_: float = 10):
        left_x = max(0, x - range_)
        right_x = min(300, x + range_)

        left_lim = np.argmax(xs > left_x)

        if right_x >= max(xs):
            right_lim = len(xs)
        else:
            right_lim = np.argmax(xs > right_x)

        _ys = ys[left_lim:right_lim]

        duration = right_x - left_x

        return np.mean(_ys), np.std(_ys) / np.sqrt(_ys.size), np.sum(_ys), _ys.size / duration

    moving_averages, moving_averages_std, moving_sum, moving_size = zip(
        *[get_moving_average(xs, ys, x=x) for x in np.unique(xs)])
    moving_averages, moving_averages_std, moving_sum, moving_size = np.array(moving_averages), np.array(
        moving_averages_std), np.array(moving_sum), np.array(moving_size)
    plt.plot(np.unique(xs), moving_averages, color="C3", label=f'xG moving average')
    plt.fill_between(np.unique(xs), moving_averages - moving_averages_std, moving_averages + moving_averages_std,
                     color="C3", alpha=0.3)

    moving_averages, moving_averages_std, moving_sum, moving_size = zip(
        *[get_moving_average(xs, is_goals, x=x) for x in np.unique(xs)])
    moving_averages, moving_averages_std, moving_sum, moving_size = np.array(moving_averages), np.array(
        moving_averages_std), np.array(moving_sum), np.array(moving_size)
    plt.plot(np.unique(xs), moving_averages, color="C4", label=f'is_goal moving average')
    plt.fill_between(np.unique(xs), moving_averages - moving_averages_std, moving_averages + moving_averages_std,
                     color="C4", alpha=0.3)

    ax1 = plt.gca()
    ax2 = ax1.twinx()
    plt.plot(np.unique(xs), moving_size, color="C5", label=f'shots moving count')
    ax2.set_ylim((0, moving_size.max() * 2))
    ax2.set_xlim((0, 300))
    ax2.legend(loc=1)
    plt.sca(ax1)

    alpha = 0.1
    plt.scatter(xs[~is_goals], ys[~is_goals], marker='x', c='k', alpha=alpha, label='Not goals')
    plt.scatter(xs[is_goals], ys[is_goals], marker='o', c='k', alpha=alpha, label='Goals')
    plt.legend()
    plt.xlim((0, 300))
    plt.ylim((0, 1))
    plt.xticks(np.arange(0, 301, 60))
    plt.grid()
    plt.xlabel('Time elapsed (s)')
    plt.ylabel('xG')

    plt.tight_layout()
    plt.savefig(os.path.join(save_fig_folder, "goals_against_time.png"), dpi=300)
    plt.show()


def save_plotly_csv(replay_folder: str, analysis_folder: str, save_fig_folder: str):
    full_df, datas, matches = read_description_dicts(analysis_folder)

    shooting_positions = defaultdict(list)

    for filename, _df in datas.items():
        datas[filename] = (_df, flip_teams(_df))

    # Get all shots, with data needed for labels etc
    for row in full_df.itertuples():
        filename = row.Index[0]
        df_blue, df_orange = datas[filename]
        df = df_orange if row.shooter_is_orange else df_blue

        shooter_x = df.loc[row.frame, (row.shooter, 'pos_x')]
        shooter_y = df.loc[row.frame, (row.shooter, 'pos_y')]
        shooter_z = df.loc[row.frame, (row.shooter, 'pos_z')]

        game_names = get_game_name(row.Index[0][:-4], replay_folder)
        if len(game_names) == 1:
            game_name = os.path.basename(game_names[0])[2:].strip()
        else:
            print(f"Found {len(game_names)} game names for {row.Index[0][:-4]}: {game_names}")
            print(f"\tActual players: {row.shooter}, {row.defenders}")
            game_name = "unknown"

        shooting_positions[row.shooter].append(((shooter_x, shooter_y, shooter_z), row, game_name))
        pass
    pass

    data = []
    for player, _data in shooting_positions.items():
        _shooting_positions, rows, game_names = zip(*_data)
        _xs, _ys, _zs = zip(*_shooting_positions)

        for i in range(len(rows)):
            row = rows[i]
            data.append(
                {'player': player,
                 'xG': row.predicted_xG,
                 'x': _xs[i],
                 'y': _ys[i],
                 'z': _zs[i],
                 'is_goal': row.is_goal,
                 'game_name': game_name,
                 'team': player_teams[player]
                 }
            )

    x = pd.DataFrame.from_records(data)

    x.to_csv('x.csv')


def visualise_shooter_positions_plotly(replay_folder: str, analysis_folder: str, save_fig_folder: str):
    full_df, datas, matches = read_description_dicts(analysis_folder)

    shooting_positions = defaultdict(list)

    for filename, _df in datas.items():
        datas[filename] = (_df, flip_teams(_df))

    # Get all shots, with data needed for labels etc
    for row in full_df.itertuples():
        filename = row.Index[0]
        df_blue, df_orange = datas[filename]
        df = df_orange if row.shooter_is_orange else df_blue

        shooter_x = df.loc[row.frame, (row.shooter, 'pos_x')]
        shooter_y = df.loc[row.frame, (row.shooter, 'pos_y')]
        shooter_z = df.loc[row.frame, (row.shooter, 'pos_z')]

        game_names = get_game_name(row.Index[0][:-4], replay_folder)
        if len(game_names) == 1:
            game_name = os.path.basename(game_names[0])[2:].strip()
        else:
            print(f"Found {len(game_names)} game names for {row.Index[0][:-4]}: {game_names}")
            print(f"\tActual players: {row.shooter}, {row.defenders}")
            game_name = "unknown"

        shooting_positions[row.shooter].append(((shooter_x, shooter_y, shooter_z), row, game_name))
        pass
    pass

    # Get plotting datas

    # data = []
    # traces_count = []
    # thresholds = []
    # for i in range(10):
    #     threshold = i / 10
    #     new_data = plotly_helpers.get_data(shooting_positions, player_teams, threshold)
    #     if new_data:
    #         thresholds.append(threshold)
    #         current_length = len(data)
    #         data += new_data
    #         traces_count.append((current_length, len(data)))
    #
    # for i in range(traces_count[0][1]):  # Set all visible (threshold xG of 0)
    #     # data[i].visible = True
    #     data[i]['visible'] = True
    #
    # print(traces_count)
    # steps = []
    # for i in range(len(traces_count)):
    #     step = {
    #         'method': 'restyle',
    #         'args': ['visible', [False] * len(data)],
    #         'label': f"{thresholds[i]:.2f}"
    #     }
    #     start_index, end_index = traces_count[i]
    #     # print(start_index, end_index + 1)
    #     for _i in range(start_index, end_index):
    #         step['args'][1][_i] = True
    #     steps.append(step)

    data = plotly_helpers.get_data(shooting_positions, player_teams, 0)
    print(f"Found {len(data)} traces")
    steps = []
    for threshold in [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        step = {
            'method': 'restyle',
            'args': ['transforms[1]',
                     {'type': 'filter', 'target': 'marker.color', 'operation': '>', 'value': threshold}],
            'label': f"{threshold:.2f}"
        }
        steps.append(step)
    # print([sum(step['args'][1]) for step in steps])
    sliders = [
        {
            'active': 0,
            'currentvalue': {'prefix': "xG threshold: "},
            'pad': {'t': 30, 'b': 30, 'l': 30, 'r': 30},
            'steps': steps
        }
    ]

    updatemenus = [
        {
            'buttons': [
                {
                    'method': 'restyle',
                    'label': 'All shots',
                    'args': ['transforms[0]', {'type': 'filter', 'target': 'marker.size', 'operation': '>', 'value': 0}]
                },
                {
                    'method': 'restyle',
                    'label': 'Goals only',
                    'args': ['transforms[0]', {'type': 'filter', 'target': 'marker.size', 'operation': '>', 'value': 8}]
                },
                {
                    'method': 'restyle',
                    'label': 'Non-goals only',
                    'args': ['transforms[0]',
                             {'type': 'filter', 'target': 'marker.size', 'operation': '<=', 'value': 8}]
                }
            ],
            'pad': {'t': 30, 'l': 30},
            'type': 'buttons',
            'showactive': True
        },
    ]

    camera_norm = 7000
    layout = dict(
        scene=({
            'xaxis': {'range': [-4096, 4096], 'zeroline': False, "showticklabels": False, 'title': {"text": ""}},
            'yaxis': {'range': [-5120, 5120], 'zeroline': True, "showticklabels": False, 'title': {"text": ""}},
            'zaxis': {'range': [0, 2044], "showticklabels": False, 'title': {"text": ""}},
            # 'aspectmode': 'data'
            'aspectratio': {'x': 8192 / camera_norm, 'y': 10240 / camera_norm, 'z': 2044 / camera_norm},
            'camera': {'center': {'x': 0, 'y': 0.2, 'z': 0}}
        }),
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        sliders=sliders,
        updatemenus=updatemenus
    )
    plotly.offline.plot({'data': data, 'layout': layout}, auto_open=True, validate=False)
    # py.plot({'data': data, 'layout': layout}, filename="Expected goals (xG) across RLCS S6 League Play", auto_open=True,
    #         validate=False, sharing='public')

    # fig = go.Figure(data=data, layout=layout)
    # plotly.offline.plot(fig, auto_open=True)

    # plotly.offline.plot({"data": data, "layout": layout}, auto_open=True)


if __name__ == '__main__':
    analyse(analysis_folder=analysis_output_folder)
    # visualise_shooter_positions(analysis_folder=ANALYSIS_FOLDER, save_fig_folder=SAVE_FIG_FOLDER)
    visualise_shooter_positions_plotly(replay_folder=REPLAY_FOLDER, analysis_folder=ANALYSIS_FOLDER,
                                       save_fig_folder=SAVE_FIG_FOLDER)
    # save_plotly_csv(replay_folder=REPLAY_FOLDER, analysis_folder=ANALYSIS_FOLDER,
    #                                    save_fig_folder=SAVE_FIG_FOLDER)
    # visualise_shot_time(analysis_folder=ANALYSIS_FOLDER, save_fig_folder=SAVE_FIG_FOLDER)
