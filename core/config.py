from habitat import get_config as get_task_config
from habitat.config import Config as CN

CONFIG_FILE_SEPARATOR = ","

_C = CN()

_C.TASK_CONFIG = CN()

# ==================================== for sensor =======================
_C.SENSOR = CN()
_C.SENSOR.DEPTH_MIN = 0.0
_C.SENSOR.DEPTH_MAX = 5.0
_C.SENSOR.SENSOR_HEIGHT = 1.25
_C.SENSOR.AGENT_HEIGHT = 1.5
_C.SENSOR.AGENT_RADIUS = 0.1

# ================================ for semantic map ===============================
_C.SEM_MAP = CN()
_C.SEM_MAP.ENLARGE_SIZE = 10
_C.SEM_MAP.IGNORED_MAP_CLASS = [0, ]
# for semantic segmentation, class 17 is ceiling
_C.SEM_MAP.IGNORED_SEM_CLASS = [0, 1]
_C.SEM_MAP.OBJECT_MASK_PIXEL_THRESH = 100
# explored but semantic-unrecognized pixel
_C.SEM_MAP.UNDETECTED_PIXELS_CLASS = 41
_C.SEM_MAP.CELL_SIZE = 0.05
# world model size in each dimension (left, right, top , bottom)
_C.SEM_MAP.WORLD_SIZE = 50.0
#_C.SEM_MAP.GRID_Y_SIZE = 60
_C.SEM_MAP.GRID_CLASS_SIZE = 42
_C.SEM_MAP.HABITAT_FLOOR_IDX = 2
_C.SEM_MAP.POINTS_CNT = 2
# complement the gap between the robot neighborhood and the projected occupancy map
_C.SEM_MAP.GAP_COMPLEMENT = 10

# ================================ for Frontier Exploration ===========================
_C.FE = CN()
_C.FE.COLLISION_VAL = 1
_C.FE.FREE_VAL = 2
_C.FE.UNOBSERVED_VAL = 0
_C.FE.OBSTACLE_THRESHOLD = 1


# ================================ for visualization ============================
_C.SEM_MAP.FLAG_VISUALIZE_EGO_OBS = True


# ================================ for slurm ==============================
_C.SLURM = CN()
_C.SLURM.NUM_PROCESS = 1
_C.SLURM.PROC_PER_GPU = 1


def get_config(config_paths, opts=None):
    r"""Create a unified config with default values overwritten by values from
    :ref:`config_paths` and overwritten by options from :ref:`opts`.

    Args:
            config_paths: List of config paths or string that contains comma
            separated list of config paths.
            opts: Config options (keys, values) in a list (e.g., passed from
            command line into the config. For example, ``opts = ['FOO.BAR',
            0.5]``. Argument can be used for parameter sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        for k, v in zip(opts[0::2], opts[1::2]):
            if k == "BASE_TASK_CONFIG_PATH":
                config.BASE_TASK_CONFIG_PATH = v

    config.TASK_CONFIG = get_task_config(config.BASE_TASK_CONFIG_PATH)
    if opts:
        config.CMD_TRAILING_OPTS = config.CMD_TRAILING_OPTS + opts
        config.merge_from_list(config.CMD_TRAILING_OPTS)

    config.freeze()
    return config
