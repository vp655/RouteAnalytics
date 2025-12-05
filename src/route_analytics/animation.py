# src/ravens_route/animation.py

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

from .inference import predict_route_prob

# ------------------------------------------------------------
# Constants & helpers
# ------------------------------------------------------------

TEAM_COLORS = {
    "ARI": ("#97233F", "#000000"), "ATL": ("#A71930", "#000000"),
    "BAL": ("#241773", "#9E7C0C"), "BUF": ("#00338D", "#C60C30"),
    "CAR": ("#0085CA", "#101820"), "CHI": ("#0B162A", "#C83803"),
    "CIN": ("#FB4F14", "#000000"), "CLE": ("#311D00", "#FF3C00"),
    "DAL": ("#041E42", "#869397"), "DEN": ("#FB4F14", "#002244"),
    "DET": ("#0076B6", "#B0B7BC"), "GB":  ("#203731", "#FFB612"),
    "HOU": ("#03202F", "#A71930"), "IND": ("#002C5F", "#A2AAAD"),
    "JAX": ("#006778", "#9F792C"), "KC":  ("#E31837", "#FFB81C"),
    "LA":  ("#0080C6", "#FFC20E"), "LAR": ("#003594", "#FFA300"),
    "LV":  ("#000000", "#A5ACAF"), "MIA": ("#008E97", "#FC4C02"),
    "MIN": ("#4F2683", "#FFC62F"), "NE":  ("#002244", "#C60C30"),
    "NO":  ("#D3BC8D", "#101820"), "NYG": ("#0B2265", "#A71930"),
    "NYJ": ("#125740", "#FFFFFF"), "PHI": ("#004C54", "#A5ACAF"),
    "PIT": ("#FFB612", "#101820"), "SEA": ("#002244", "#69BE28"),
    "SF":  ("#AA0000", "#B3995D"), "TB":  ("#D50A0A", "#FF7900"),
    "TEN": ("#0C2340", "#4B92DB"), "WAS": ("#5A1414", "#FFB612"),
}

FIELD_W = 120.0
FIELD_H = 160.0 / 3.0
ARROW_SCALE = 1.0

ANGLE_THRESHOLD_DEG = 7.5  # minimum change in heading (deg) to count as a "move"
MIN_SPEED_YDPS = 1.5       # minimum speed (yd/s) to count as a "move"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        return get_ipython() is not None
    except Exception:
        return False


# ------------------------------------------------------------
# Geometry / drawing helpers
# ------------------------------------------------------------

def draw_field(ax,
               field_color: str = "#ffffff",
               line_color: str = "#212529",
               number_color: str = "#adb5bd") -> None:
    ax.set_xlim(0, FIELD_W)
    ax.set_ylim(0, FIELD_H)
    ax.set_aspect("equal")
    ax.set_facecolor(field_color)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Outer border
    ax.plot([0, FIELD_W], [0, 0], color=line_color, lw=1)
    ax.plot([0, FIELD_W], [FIELD_H, FIELD_H], color=line_color, lw=1)
    ax.plot([0, 0], [0, FIELD_H], color=line_color, lw=1)
    ax.plot([FIELD_W, FIELD_W], [0, FIELD_H], color=line_color, lw=1)

    # Yard lines
    for x in np.arange(10, 111, 5):
        ax.plot([x, x], [0, FIELD_H], color=line_color, lw=0.5)

    # Hash marks
    xs = np.arange(10, 111, 1)
    hash_y1 = (160 / 6 + 18.5 / 6)
    hash_y2 = (160 / 6 - 18.5 / 6)
    for x in xs:
        ax.plot([x, x], [0, 1], color=line_color, lw=0.5)
        ax.plot([x, x], [FIELD_H - 1, FIELD_H], color=line_color, lw=0.5)
        ax.plot([x, x], [hash_y1, hash_y1 + 1], color=line_color, lw=0.5)
        ax.plot([x, x], [hash_y2 - 1, hash_y2], color=line_color, lw=0.5)

    # Yard numbers
    nums = list(range(10, 60, 10)) + list(range(40, 0, -10))
    for i, x in enumerate(range(20, 101, 10)):
        ax.text(x, 12, str(nums[i]), color=number_color,
                ha="center", va="center", fontsize=12)
        ax.text(x, FIELD_H - 12, str(nums[i]), color=number_color,
                ha="center", va="center", fontsize=12, rotation=180)


def find_first_move_frame(track_df, by, val, angle_deg_thresh=2.0, min_speed=0.0):
    """
    Find the first frame where the highlighted WR changes heading by >= angle_deg_thresh
    while moving at >= min_speed. Returns (frameId, x, y) or (None, None, None).
    Heading from successive positions; angle diff is unwrapped for robustness.
    """

    # Restrict tracking data to frames after the ball snap (plus 3)
    snap_rows = track_df.loc[track_df["event"].astype(str).str.lower() == "ball_snap"]
    snap_frame = int(pd.to_numeric(snap_rows["frameId"], errors="coerce").min())
    track_df = track_df[pd.to_numeric(track_df["frameId"], errors="coerce") >= (snap_frame + 3)].copy()
    #print(f"Tracking data filtered: kept {len(track_df)} rows after ball_snap (starting at frame {snap_frame + 3})")

    # Restrict tracking data to frames before the pass_forward
    before_pass_rows = track_df.loc[track_df["event"].astype(str).str.lower() == "pass_forward"]
    before_pass_frame = int(pd.to_numeric(before_pass_rows["frameId"], errors="coerce").min())
    track_df = track_df[pd.to_numeric(track_df["frameId"], errors="coerce") < before_pass_frame].copy()
    #print(f"Tracking data filtered: kept {len(track_df)} rows before pass_forward (up to frame {before_pass_frame + 3})")

    wr = track_df[track_df[by] == val].dropna(subset=["frameId", "x", "y"]).copy()
    wr = wr.sort_values("frameId")
    #print(len(track_df), "asdfdsfsfdsf")

    dir_eff = wr["dir"].values
    spd_eff = wr["s"].values

    n = len(dir_eff)

    frames = wr["frameId"].values
    
    xs = wr["x"].values
    ys = wr["y"].values

    #print(dir_eff)
    #print(spd_eff)

    idx = None
    for i in range(1,n):
        if (np.abs(dir_eff[i] - dir_eff[i-1]) >= angle_deg_thresh) and (spd_eff[i] >= min_speed):
            idx = i   
            break

    if idx is None:
        return (None, None, None)

    return (int(frames[idx]), float(xs[idx]), float(ys[idx]))

# ------------------------------------------------------------
# Main public API
# ------------------------------------------------------------

def animate_play_from_row(
    row: pd.Series,
    data_dir: Union[str, Path] = "data",
    out_gif: Optional[Union[str, Path]] = None,
    fps: int = 10,
    show: bool = True,
):
    """
    Create an animation for a single play (WR vs CB matchup) using a row from
    the final matchup dataset, and the route model to compute the WR win score.

    Parameters
    ----------
    row : pd.Series
        A single-row Series from your final dataset. Must include at least:
        - 'gameId', 'playId'
        - 'nflIdOff' (WR id), 'nflIdDef' (CB id)
        as well as the features needed by `predict_route_prob`.
    data_dir : str or Path
        Directory containing games.csv, plays.csv, and week{week}.csv files.
    out_gif : str or Path, optional
        If provided, save the animation as a GIF to this path.
    fps : int
        Frames per second for the GIF.
    show : bool
        If True and running in a notebook, display the animation inline.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The constructed animation object.
    """
    data_dir = Path(data_dir)

    # IDs from the row
    GAME_ID = int(row["gameId"])
    PLAY_ID = int(row["playId"])
    WR_ID = int(row["nflIdOff"])
    CB_ID = int(row["nflIdDef"])

    # Where to save by default
    if out_gif is None:
        anim_dir = Path("animations")
        anim_dir.mkdir(parents=True, exist_ok=True)
        str_GAME_ID = f"{GAME_ID}_{PLAY_ID}"
        out_gif = anim_dir / f"{str_GAME_ID}_animation.gif"
    else:
        out_gif = Path(out_gif)

    HIGHLIGHT_WR_BY = "nflId"
    HIGHLIGHT_WR_VAL = WR_ID

    HIGHLIGHT_CB_BY = "nflId"
    HIGHLIGHT_CB_VAL = CB_ID

    # Compute probability from the packaged model
    prob = float(predict_route_prob(row))

    # Load games & plays
    games_csv = data_dir / "games.csv"
    plays_csv = data_dir / "plays.csv"

    games = pd.read_csv(games_csv)
    plays = pd.read_csv(plays_csv)

    # Get game row, infer week
    game_row = games.loc[games["gameId"] == GAME_ID]
    if game_row.empty:
        raise ValueError(f"gameId {GAME_ID} not found in {games_csv}")
    game_row = game_row.iloc[0]
    actual_week = int(game_row["week"])

    week_file_needed = data_dir / f"week{actual_week}.csv"
    tracking = pd.read_csv(week_file_needed)
    selected_week = week_file_needed

    # Filter tracking to this game & play
    track = tracking[
        (pd.to_numeric(tracking["gameId"], errors="coerce") == GAME_ID) &
        (pd.to_numeric(tracking["playId"], errors="coerce") == PLAY_ID)
    ].copy()

    required_cols = {"gameId", "playId", "frameId", "x", "y", "team", "s", "dir"}
    for c in ["gameId", "playId", "frameId"]:
        track[c] = pd.to_numeric(track[c], errors="coerce")

    frame_ids = (
        track["frameId"].dropna().astype(int).sort_values().unique().tolist()
    )
    if not frame_ids:
        raise ValueError("No frames found for this game/play in tracking data.")
    LAST10_START = frame_ids[-10] if len(frame_ids) >= 10 else frame_ids[0]

    # Team color setup
    home_abbr = str(game_row["homeTeamAbbr"])
    away_abbr = str(game_row["visitorTeamAbbr"])
    home_1, home_2 = TEAM_COLORS.get(home_abbr, ("#2b8cbe", "#045a8d"))
    away_1, away_2 = TEAM_COLORS.get(away_abbr, ("#de2d26", "#a50f15"))

    # LOS & first down line
    play_dir = str(track.iloc[0].get("playDirection", "right")).lower()
    abs_yl = float(
        plays.loc[
            (plays["gameId"] == GAME_ID) & (plays["playId"] == PLAY_ID),
            "absoluteYardlineNumber",
        ].iloc[0]
    )
    yards_to_go = float(
        plays.loc[
            (plays["gameId"] == GAME_ID) & (plays["playId"] == PLAY_ID),
            "yardsToGo",
        ].iloc[0]
    )

    if play_dir == "left":
        line_of_scrimmage = abs_yl
        to_go_line = line_of_scrimmage - yards_to_go
    else:
        line_of_scrimmage = 100 - abs_yl
        to_go_line = line_of_scrimmage + yards_to_go

    # Title text from play description
    desc_col = next(
        (c for c in plays.columns if c.lower() == "playdescription"), None
    )
    if desc_col:
        title_txt = str(
            plays.loc[
                (plays["gameId"] == GAME_ID) & (plays["playId"] == PLAY_ID),
                desc_col,
            ].iloc[0]
        )
    else:
        title_txt = f"Play {GAME_ID}-{PLAY_ID}"

    # Basic kinematics
    need_cols = [
        "x",
        "y",
        "s",
        "dir",
        "event",
        "displayName",
        "jerseyNumber",
        "frameId",
        "team",
        "nflId",
    ]
    track = track[[c for c in need_cols if c in track.columns]].copy()
    rad = np.deg2rad(pd.to_numeric(track["dir"], errors="coerce"))
    speed = pd.to_numeric(track["s"], errors="coerce").fillna(0.0)
    track["v_x"] = np.sin(rad) * speed
    track["v_y"] = np.cos(rad) * speed

    # Highlight columns numeric/string
    for col in {HIGHLIGHT_WR_BY, HIGHLIGHT_CB_BY} - {None}:
        if col and col in track.columns:
            if col in ["nflId", "jerseyNumber", "frameId"]:
                track[col] = pd.to_numeric(track[col], errors="coerce")
            else:
                track[col] = track[col].astype(str)

    # Map 'home'/'away' to team abbreviations
    def map_team(t):
        t = str(t)
        if t == "home":
            return home_abbr
        if t == "away":
            return away_abbr
        return t

    track["team"] = track["team"].astype(str).apply(map_team)

    home_df = track[track["team"] == home_abbr]
    away_df = track[track["team"] == away_abbr]
    ball_df = track[track["team"] == "football"]

    # First move frame for highlighted WR
    move_frame_id, move_x, move_y = find_first_move_frame(
        track,
        HIGHLIGHT_WR_BY,
        HIGHLIGHT_WR_VAL,
        angle_deg_thresh=ANGLE_THRESHOLD_DEG,
        min_speed=MIN_SPEED_YDPS,
    )

    # --------------------------------------------------------
    # Build the figure and animation
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    draw_field(ax)
    ax.plot(
        [line_of_scrimmage, line_of_scrimmage],
        [0, FIELD_H],
        color="#0d41e1",
        lw=2,
    )
    ax.plot(
        [to_go_line, to_go_line],
        [0, FIELD_H],
        color="#f9c80e",
        lw=2,
    )

    away_pts = ax.scatter(
        [], [], s=250, facecolors="#f8f9fa", edgecolors=away_2,
        linewidths=2, zorder=3
    )
    home_pts = ax.scatter(
        [], [], s=250, facecolors=home_1, edgecolors=home_2,
        linewidths=2, zorder=3
    )
    (ball_pt,) = ax.plot(
        [], [], "o", ms=8, mfc="#935e38", mec="#d9d9d9", mew=1.5, zorder=4
    )
    title = ax.set_title(title_txt, fontsize=12, weight="bold")

    # Highlighted WR
    wr_pt = ax.scatter(
        [], [], s=360, facecolors="#000000", edgecolors="#ffffff",
        linewidths=2.5, zorder=6
    )
    wr_label = ax.text(
        0, 0, "", color="#ffffff", fontsize=9, ha="center", va="bottom", zorder=7
    )
    wr_trail_x, wr_trail_y = [], []
    wr_trail, = ax.plot([], [], lw=2, color="#000000", alpha=0.5, zorder=5.5)

    # Highlighted CB
    cb_pt = ax.scatter(
        [], [], s=360, facecolors="#cc0000", edgecolors="#ffffff",
        linewidths=2.5, zorder=6
    )
    cb_label = ax.text(
        0, 0, "", color="#ffffff", fontsize=9, ha="center", va="bottom", zorder=7
    )
    cb_trail_x, cb_trail_y = [], []
    cb_trail, = ax.plot([], [], lw=2, color="#cc0000", alpha=0.5, zorder=5.5)

    # Live separation text
    sep_text = ax.text(
        2,
        FIELD_H - 2,
        "Separation: -- yd",
        fontsize=12,
        color="#111111",
        ha="left",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="#444444",
            alpha=0.8,
        ),
        zorder=10,
    )

    # WR move banner + marker
    move_text = ax.text(
        FIELD_W - 2,
        FIELD_H - 2,
        "",
        fontsize=14,
        color="#111111",
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.35",
            fc="#ffe680",
            ec="#b38f00",
            alpha=0.9,
        ),
        zorder=11,
    )
    move_marker = ax.scatter(
        [], [],
        s=220,
        facecolors="#ffe680",
        edgecolors="#b38f00",
        linewidths=2.0,
        zorder=10,
    )

    # Prediction text (bottom-right)
    pred_text = ax.text(
        FIELD_W - 2,
        2,
        "",
        fontsize=10,
        color="#111111",
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="#444444",
            alpha=0.9,
        ),
        zorder=12,
    )

    # Dynamic artists
    jersey_texts = []
    arrow_artists = []

    def init():
        wr_trail_x.clear()
        wr_trail_y.clear()
        cb_trail_x.clear()
        cb_trail_y.clear()
        wr_trail.set_data([], [])
        cb_trail.set_data([], [])


        away_pts.set_offsets(np.empty((0, 2)))
        home_pts.set_offsets(np.empty((0, 2)))
        ball_pt.set_data([], [])
        wr_pt.set_offsets(np.empty((0, 2)))
        wr_label.set_text("")
        wr_trail.set_data([], [])
        cb_pt.set_offsets(np.empty((0, 2)))
        cb_label.set_text("")
        cb_trail.set_data([], [])
        sep_text.set_text("Separation: -- yd")
        pred_text.set_text("")
        move_text.set_text("")
        move_marker.set_offsets(np.empty((0, 2)))
        return [
            away_pts,
            home_pts,
            ball_pt,
            wr_pt,
            wr_label,
            wr_trail,
            cb_pt,
            cb_label,
            cb_trail,
            sep_text,
            move_text,
            move_marker,
            pred_text,
            title,
        ]

    def clear_dynamic():
        nonlocal arrow_artists, jersey_texts
        for a in arrow_artists:
            a.remove()
        arrow_artists = []
        for t in jersey_texts:
            t.remove()
        jersey_texts = []

    def draw_team(frame_id, team_df, arrow_color, num_color):
        sub = team_df[team_df["frameId"] == frame_id]
        if sub.empty:
            return None, [], []
        pts = sub[["x", "y"]].values
        arrows, texts = [], []
        for _, r in sub.iterrows():
            arr = ax.arrow(
                r["x"],
                r["y"],
                ARROW_SCALE * r["v_x"],
                ARROW_SCALE * r["v_y"],
                head_width=0.8,
                head_length=0.8,
                length_includes_head=True,
                fc=arrow_color,
                ec=arrow_color,
                alpha=0.9,
                lw=1.0,
                zorder=2,
            )
            arrows.append(arr)
            num = "" if pd.isna(r.get("jerseyNumber")) else str(int(r["jerseyNumber"]))
            txt = ax.text(
                r["x"],
                r["y"],
                num,
                color=num_color,
                fontsize=9,
                ha="center",
                va="center",
                zorder=5,
            )
            texts.append(txt)
        return pts, texts, arrows

    def _set_highlight(frame_id, by, val, pt_artist, label_artist, trail_x, trail_y):
        if by not in track.columns or pd.isna(val):
            pt_artist.set_offsets(np.empty((0, 2)))
            label_artist.set_text("")
            return None
        sub = track[(track["frameId"] == frame_id) & (track[by] == val)]
        if sub.empty:
            pt_artist.set_offsets(np.empty((0, 2)))
            label_artist.set_text("")
            return None
        hx = float(sub["x"].values[0])
        hy = float(sub["y"].values[0])
        pt_artist.set_offsets([[hx, hy]])

        # label
        label_txt = ""
        if "jerseyNumber" in sub.columns and pd.notna(sub["jerseyNumber"].values[0]):
            label_txt = str(int(sub["jerseyNumber"].values[0]))
        elif "displayName" in sub.columns and pd.notna(sub["displayName"].values[0]):
            label_txt = str(sub["displayName"].values[0])
        label_artist.set_position((hx, hy + 1.2))
        label_artist.set_text(label_txt)

        # trail
        trail_x.append(hx)
        trail_y.append(hy)
        return (hx, hy)

    def animate(frame_id):
        nonlocal arrow_artists, jersey_texts
        clear_dynamic()

        # Teams
        pts_away, txt_away, arr_away = draw_team(
            frame_id, away_df, away_1, away_2
        )
        if pts_away is not None:
            away_pts.set_offsets(pts_away)
            away_pts.set_edgecolors(away_2)

        pts_home, txt_home, arr_home = draw_team(
            frame_id, home_df, home_2, home_2
        )
        if pts_home is not None:
            home_pts.set_offsets(pts_home)
            home_pts.set_edgecolors(home_2)

        arrow_artists.extend(arr_away + arr_home)
        jersey_texts.extend(txt_away + txt_home)

        # Ball
        b = ball_df[ball_df["frameId"] == frame_id]
        if not b.empty:
            ball_pt.set_data(b["x"].values[0], b["y"].values[0])
        else:
            ball_pt.set_data([], [])

        # Highlights
        wr_pos = _set_highlight(
            frame_id,
            HIGHLIGHT_WR_BY,
            HIGHLIGHT_WR_VAL,
            wr_pt,
            wr_label,
            wr_trail_x,
            wr_trail_y,
        )
        wr_trail.set_data(wr_trail_x, wr_trail_y)

        cb_pos = _set_highlight(
            frame_id,
            HIGHLIGHT_CB_BY,
            HIGHLIGHT_CB_VAL,
            cb_pt,
            cb_label,
            cb_trail_x,
            cb_trail_y,
        )
        cb_trail.set_data(cb_trail_x, cb_trail_y)

        # Separation
        if wr_pos is not None and cb_pos is not None:
            dx = wr_pos[0] - cb_pos[0]
            dy = wr_pos[1] - cb_pos[1]
            sep = float(np.hypot(dx, dy))
            sep_text.set_text(f"Separation: {sep:0.2f} yd")
        else:
            sep_text.set_text("Separation: -- yd")

        # Move banner & marker
        if move_frame_id is not None and frame_id >= move_frame_id:
            move_text.set_text("MOVE OCCURRED")
            if np.isfinite(move_x) and np.isfinite(move_y):
                move_marker.set_offsets([[move_x, move_y]])
        else:
            move_text.set_text("")
            move_marker.set_offsets(np.empty((0, 2)))

        title.set_text(f"{title_txt}\nFrame: {frame_id}")

        # Predicted WR win probability from the model
        color = "#118800" if prob >= 0.5 else "#cc0000"
        pred_text.set_text(f"WR Win Score {prob:.1%}")
        pred_text.set_color(color)
        pred_text.set_bbox(
            dict(
                boxstyle="round,pad=0.3",
                fc="#e8ffe8" if prob >= 0.5 else "#ffe8e8",
                ec=color,
                alpha=0.9,
            )
        )

        return [
            away_pts,
            home_pts,
            ball_pt,
            wr_pt,
            wr_label,
            wr_trail,
            cb_pt,
            cb_label,
            cb_trail,
            sep_text,
            move_text,
            move_marker,
            pred_text,
            title,
            *arrow_artists,
            *jersey_texts,
        ]

    assert len(frame_ids) > 0, "frame_ids is empty"
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frame_ids,
        interval=100,
        blit=False,
        cache_frame_data=False,
    )

    # Show and/or save
    if show and _in_notebook():
        from IPython.display import HTML, display  # type: ignore

        display(HTML(anim.to_jshtml()))

    if out_gif is not None:
        os.makedirs(out_gif.parent, exist_ok=True)
        writer = animation.PillowWriter(
            fps=fps, metadata=dict(artist="ravens_route")
        )

        def _progress(i, n):
            if i == 0 or (i + 1) == n or (i + 1) % 10 == 0:
                print(f"Writing frame {i+1}/{n}...")

        anim.save(out_gif, writer=writer, dpi=100, progress_callback=_progress)
        print(f"Saved {out_gif} (from {os.path.basename(str(selected_week))})")

    return anim


def animate_play_from_index(
    predictions_csv: Union[str, Path],
    row_num: int,
    data_dir: Union[str, Path] = "data",
    out_gif: Optional[Union[str, Path]] = None,
    fps: int = 10,
    show: bool = True,
):
    """
    Convenience wrapper: read the predictions CSV, take row_num, and call
    animate_play_from_row on that row.

    This ignores any existing 'pred_offenseWin' in the CSV and instead uses
    the packaged route model to compute the probability.
    """
    predictions_csv = Path(predictions_csv)
    df = pd.read_csv(predictions_csv)
    row = df.iloc[row_num]
    return animate_play_from_row(
        row=row,
        data_dir=data_dir,
        out_gif=out_gif,
        fps=fps,
        show=show,
    )
