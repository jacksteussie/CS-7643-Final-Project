from ultralytics.data.split_dota import split_test, split_trainval
from const import DOTA_DIR, DOTA_MOD_DIR

if DOTA_MOD_DIR:
    # split train and val set, with labels.
    split_trainval(
        data_root=DOTA_DIR,
        save_dir=DOTA_MOD_DIR,
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
    )
    # split test set, without labels.
    split_test(
        data_root=DOTA_DIR,
        save_dir=DOTA_MOD_DIR,
        rates=[0.5, 1.0, 1.5],  # multiscale
        gap=500,
    )