import os
import time


def create_dir():
    """构造目录 ./scenario_name/train_data/time_struct/(plots, policy, buffer)"""
    scenario_path = "./results/"
    if not os.path.exists(scenario_path):
        os.mkdir(scenario_path)

    tm_struct = time.localtime(time.time())
    experiment_name = "%02d_%02d_%02d_%02d" % \
                      (tm_struct[1], tm_struct[2], tm_struct[3], tm_struct[4])
    experiment_path = os.path.join(scenario_path, experiment_name)
    if os.path.exists(experiment_path):
        os.remove(experiment_path)
    else:
        os.mkdir(experiment_path)

    save_paths = list()
    save_paths.append(experiment_path + "/models/")
    save_paths.append(experiment_path + "/figures/")
    save_paths.append(experiment_path + "/buffers/")
    for save_path in save_paths:
        os.mkdir(save_path)
    return save_paths[0], save_paths[1], save_paths[2]