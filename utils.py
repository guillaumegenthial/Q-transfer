def plot_rewards(training_rewards_list, file_name="plot.png", style="seaborn-paper"):
    import matplotlib.pyplot as plt
    import numpy as np
    # see styles at https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
    plt.style.use(style)
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-' , '--', '-.', ':', 'None', ' ', '' ]
    colors = ('k', 'r', 'b', 'c', 'm', 'y', 'k')
    names = []

    fig = plt.figure(figsize=[9,5])
    ax = plt.subplot(111)
    # set the basic properties
    ax.set_xlabel('Trials')
    ax.set_ylabel('Rewards')

    for l, c, (name, training_rewards) in zip(linestyles, colors, training_rewards_list):
        plt.plot(range(len(training_rewards)), training_rewards)
            # , linestyle="solid", color=c)
        names.append(" ".join(name.split("_")))

    plt.legend(names, loc='lower right')
    plt.savefig(file_name)

