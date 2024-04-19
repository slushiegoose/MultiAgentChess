from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from tqdm import tqdm
from evaluator import get_average_deep, get_average_random, get_average_simple


def plot_lf_against_avg():
    """Plot learning factor against average time to checkmate"""
    learning_factors = [0.1, 0.3, 0.5, 0.7, 0.9]
    average_times = []
    random_agent_time = get_average_random()
    print(f"Random Agent: {random_agent_time}")


    for lf in learning_factors:
        average_times.append(get_average_simple(lf))
    
    
    fig = plt.figure()
    plt.plot(learning_factors, average_times)
    plt.xlabel("Discount Factor")
    plt.ylabel("Average Time to Checkmate")
    plt.title("Discount Factor against Average Time to Checkmate")

    plt.axhline(y=random_agent_time, color="r", linestyle="--", label="Random Agent")
    plt.axhline(y=2, color="g", linestyle="--", label="Mate in 2")
    plt.savefig("learning_factor_vs_avg_time.png")

def plot_res_against_avg_deep():
    """Plot resolution against average time to checkmate for DeepQ agent"""
    resolutions = range(100, 12600, 1000)
    average_times = []


    for res in tqdm(resolutions):
        average_times.append(get_average_deep(res))

    random_agent_time = get_average_random()
    print(f"Random Agent: {random_agent_time}")

    fig = plt.figure()
    plt.plot(resolutions, average_times)
    plt.xlabel("Resolution")
    plt.ylabel("Average Time to Checkmate")
    plt.title("Resolution against Average Time to Checkmate")

    plt.axhline(y=random_agent_time, color="r", linestyle="--", label="Random Agent")
    plt.axhline(y=2, color="g", linestyle="--", label="Mate in 2")
    plt.savefig("resolution_vs_avg_time_deep.png")

def plot_res_against_avg_simple():
    """Plot resolution against average time to checkmate for SimpleQ agent"""
    resolutions = range(100, 14600, 1000)
    average_times = []
    random_agent_time = get_average_random()
    print(f"Random Agent: {random_agent_time}")

    for res in tqdm(resolutions):
        average_times.append(get_average_simple(0.9, res))

    fig = plt.figure()
    plt.plot(resolutions, average_times)
    plt.xlabel("Resolution")
    plt.ylabel("Average Time to Checkmate")
    plt.title("Resolution against Average Time to Checkmate")

    plt.axhline(y=random_agent_time, color="r", linestyle="--", label="Random Agent")
    plt.axhline(y=2, color="g", linestyle="--", label="Mate in 2")
    plt.savefig("resolution_vs_avg_time_simple.png")


def plot_res_against_checkmates_deep():
    """Plot resolution against percentage of checkmates for DeepQ agent"""
    resolutions = range(100, 12600, 1000)
    checkmates = []

    for res in tqdm(resolutions):
        checkmates.append(get_average_deep(res, False) / 100)
    fig = plt.figure()
    plt.plot(resolutions, checkmates)
    plt.xlabel("Resolution")
    plt.ylabel("Percentage of Checkmates")
    plt.title("Resolution against Percentage of Checkmates")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.savefig("resolution_vs_checkmates_deep.png")

def plot_res_against_checkmates_simple():
    """Plot resolution against percentage of checkmates for SimpleQ agent"""
    resolutions = range(100, 14600, 1000)
    checkmates = []

    for res in tqdm(resolutions):
        checkmates.append(get_average_simple(0.9, res, False) / 100)
    fig = plt.figure()
    plt.plot(resolutions, checkmates)
    plt.xlabel("Resolution")
    plt.ylabel("Percentage of Checkmates")
    plt.title("Resolution against Percentage of Checkmates")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
    plt.savefig("resolution_vs_checkmates_simple.png")

if __name__ == "__main__":
    plot_res_against_avg_deep()
    plot_res_against_avg_simple()
    plot_lf_against_avg()
    plot_res_against_checkmates_deep()
    plot_res_against_checkmates_simple()