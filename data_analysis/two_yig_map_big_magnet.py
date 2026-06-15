import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from pathlib import Path
import extractParameters

SINGLE_BASE = Path(r"measurements\BigMagnet_MawganRes_0.5mm_21052026")
COMBO_BASE  = Path(r"C:\Users\2514468E\OneDrive - University of Glasgow\PhD\q-SPIn\measurements\Two0.5MawganRes12-06-2026")

SINGLE_PATHS = {
    1: SINGLE_BASE / r"position 1\TFBZPK_2D_current.h5",
    2: SINGLE_BASE / r"position 2\TFC0GE_2D_current.h5",
    3: SINGLE_BASE / r"position 3\TFC0XB_2D_current.h5",
    4: SINGLE_BASE / r"position 4\TFC1GM_2D_current.h5",
    5: SINGLE_BASE / r"position 5\TFC1WW_2D_current.h5",
    6: SINGLE_BASE / r"position 6\TFC2BT_2D_current.h5",
    7: SINGLE_BASE / r"position 7\TFC2S7_2D_current.h5",
    8: SINGLE_BASE / r"position 8\TFC37C_2D_current.h5",
    9: SINGLE_BASE / r"position 9\TFC3Q1_2D_current.h5",
}

selected = []   # holds 0, 1, or 2 position numbers
buttons  = {}   # maps position number -> Button object
btn_axes = {}   # maps position number -> Axes object

COLOR_DEFAULT  = '0.85'
COLOR_SELECTED = '#4C9BE8'
COLOR_ERROR    = '#E84C4C'

def resolve_filepath(selection):
    if isinstance(selection, int):
        path = SINGLE_PATHS.get(selection)
        if path and path.exists():
            return path
        raise FileNotFoundError(f"No file for position {selection}: {path}")

    x, y = selection
    for a, b in [(x, y), (y, x)]:
        folder = COMBO_BASE / f"position-{a}-{b}"
        h5_files = list(folder.glob("*.h5"))
        if h5_files:
            if len(h5_files) > 1:
                print(f"Warning: multiple .h5 files in {folder.name}, using first.")
            return h5_files[0]

    raise FileNotFoundError(
        f"No .h5 file found in either position-{x}-{y} or position-{y}-{x} in {COMBO_BASE}"
    )

def refresh_button_colors():
    for pos, btn in buttons.items():
        if pos in selected:
            btn.color = COLOR_SELECTED
            btn.hovercolor = COLOR_SELECTED
        else:
            btn.color = COLOR_DEFAULT
            btn.hovercolor = '0.75'
        btn_axes[pos].set_facecolor(btn.color)
    plt.draw()

def on_button_click(pos):
    def handler(event):
        if pos in selected:
            selected.remove(pos)
        elif len(selected) < 2:
            selected.append(pos)
        else:
            # already have 2 — replace the first with the new one
            selected.pop(0)
            selected.append(pos)
        refresh_button_colors()
    return handler

def on_plot(event):
    if not selected:
        print("Select at least one position first.")
        return

    key = selected[0] if len(selected) == 1 else tuple(sorted(selected))

    try:
        filepath = resolve_filepath(key)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # flash buttons red briefly
        for pos in selected:
            btn_axes[pos].set_facecolor(COLOR_ERROR)
        plt.draw()
        return

    print(f"Loading: {filepath}")
    extractParameters.main(filepaths=[filepath], show_figure2=False)

def build_gui():
    fig = plt.figure(figsize=(5, 5))
    fig.canvas.manager.set_window_title("Position selector")

    # 3x3 grid for position buttons + bottom row for Plot button
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           top=0.92, bottom=0.08,
                           left=0.08, right=0.92,
                           hspace=0.4, wspace=0.3)

    positions = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    for row_idx, row in enumerate(positions):
        for col_idx, pos in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            btn = Button(ax, str(pos), color=COLOR_DEFAULT, hovercolor='0.75')
            btn.on_clicked(on_button_click(pos))
            buttons[pos]  = btn
            btn_axes[pos] = ax

    # Plot button spans full bottom row
    ax_plot = fig.add_subplot(gs[3, :])
    btn_plot = Button(ax_plot, 'Plot', color='#5CB85C', hovercolor='#4CAE4C')
    btn_plot.label.set_color('white')
    btn_plot.label.set_fontweight('bold')
    btn_plot.on_clicked(on_plot)

    fig.suptitle("Select 1 or 2 positions, then Plot", fontsize=10, color='0.3')
    plt.show()

if __name__ == "__main__":
    build_gui()