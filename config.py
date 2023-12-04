chip_config ={
    'tau': 0.02,   # maximum tolerance service latency, second, default 0.02
    'fD': 1.7e8,    # computation frequency of mobile device, 1.5~5GHz, default 1.7e8
    'u': 1e-19,     # effective switched capacitance related to chip architecture, default 1e-19
    'C': 20e3,    # cache size constraint, test, default 20e3
}

system_config = {
    'M': 8,     # number of computation cores, default 8
    'F': 4,     # number of tasks, default 4
    'FQ': 8,    # number of computing frequency options
    'maxp': 70,     # maximum markov chain prob, percentage, default 70%
    'weight': 1,    # objective weight, default 1
}


def find_best_reactive_fD():
    from tool.data_loader import load_data
    if system_config['F'] == 4:
        task_utils = load_data('./data/task4_utils.csv')
    elif system_config['F'] == 6:
        task_utils = load_data('./data/task6_utils.csv')
    elif system_config['F'] == 8:
        task_utils = load_data('./data/task8_utils.csv')
    elif system_config['F'] == 10:
        task_utils = load_data('./data/task10_utils.csv')
    task_set = task_utils.tolist()
    tau = chip_config['tau']
    fD = chip_config['fD']
    u = chip_config['u']
    best_fDs = []
    min_values = []

    for A_t in range(system_config['F']):
        I_At = task_set[A_t][0]
        w_At = task_set[A_t][2]
        min_val = 1e10
        best_fD = 0
        for C_R_At in range(1, system_config['F'] + 1):
            B_R = I_At / (tau - I_At * w_At / (C_R_At * fD))
            E_R = u * (C_R_At * fD) ** 2 * I_At * w_At
            if B_R + E_R <= min_val:
                best_fD = C_R_At
                min_val = B_R + E_R

        best_fDs.append(best_fD)
        min_values.append(min_val)

    return best_fDs, min_values


# find the best fD choice for each task
best_fDs, _ = find_best_reactive_fD()
