from .fkd import FKD
from .fkd_rewards import get_reward_function
 
FKD_ARGS_DEFAULTS = {
    "potential_type": "diff",
    "lmbda": 1.0,
    "num_particles": 4,
    "adaptive_resampling": True,
    "resample_frequency": 5,
    "resampling_t_start": 0,
    "resampling_t_end": 60,
    "time_steps": 60,  # set as same as resampling_t_end
    "latent_to_decode_fn": lambda x: x,  # identity for EDM (already image-space)
    "get_reward_fn": "PixelReward",  # "ClassifierLoss" will be defined later
    "cls_model": None,  # it is required when using "ClassifierLoss"
    "use_smc": True,
    "output_dir": "./outputs/generated/fkd_results",  # modify
    "print_rewards": True,  # print rewards during sampling
    "visualize_intermediate": True,  # save results during sampling in output_dir
    "visualzie_x0": True,  # save x0 prediction during sampling in output_dir
}