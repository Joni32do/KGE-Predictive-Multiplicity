# Simple SVM Classification for XOR dataset using sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn import svm
from utils import Custom_SVM

from plot_glyph import draw_binary_glyph, explain_binary_glyph, TRUE_GREEN, FALSE_RED


cm = 1/2.54

def make_xor_dataset(n: int = 64, sampling: str="marx"):
    if sampling == "mesh":
        X = np.meshgrid(np.linspace(-1, 1, int(np.sqrt(n))),
                        np.linspace(-1, 1, int(np.sqrt(n))))
        X = np.array(X).reshape(2, -1).T
    elif sampling == "random":
        X = np.array([np.random.rand(n) * 2 - 1,
                      np.random.rand(n) * 2 - 1]).T
    elif sampling == "custom":
        X = np.meshgrid(np.linspace(-1, 1, int(np.sqrt(n))), np.linspace(-1, 1, int(np.sqrt(n))))
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    return X, y

def make_marx_dataset(n: int = 10):
    # Example dataset similar to the one in marx Figure 2
    X_m = np.array([- np.random.rand(n//2), np.random.rand(n//2)]).T
    X_p = np.array([np.random.rand(n//2), - np.random.rand(n//2)]).T
    X = np.concatenate([X_m, X_p])
    y = np.array([0] * (n//2) + [1] * (n//2))
    return X, y

def fit_baseline_and_epsilon_set(X, y, epsilon=0.1, n_representatives=3):
    # Baseline
    h0 =  svm.SVC(kernel='linear')
    h0.fit(X, y)
    h0_score = h0.score(X, y)
    print(f"Coefficients of the baseline classifier: {h0.coef_}")
    # Generate epsilon set
    eps_set = []
    max_tries = 100
    for i in range(max_tries):
        clf = svm.SVC(kernel='linear')
        clf.fit(X, y)
        # Sample 
        if np.abs(h0_score - clf.score(X, y)) < epsilon:
            eps_set.append(clf)
        if len(eps_set) == n_representatives:
            break
    return h0, eps_set

def example_baseline_and_epsilon_set():
    h0 = Custom_SVM(w=np.array([0, 1]), b=0)
    eps_set = []

    for i in range(3):
        clf = Custom_SVM(w=np.array([np.sin((i+1)*np.pi/5),
                                     np.cos((i+1)*np.pi/5)]), b=0)
        eps_set.append(clf)
    return h0, eps_set

# Plot for presentation
def plot_only_dataset():
    X, y = make_xor_dataset(n=100, sampling="mesh")
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 10*cm))
    color = np.full((len(y), 3), FALSE_RED)
    color[y] = TRUE_GREEN
    ax.scatter(X[:, 0], X[:, 1], c=color, s=4)
    
    # Shade area
    ax.fill_between([-1, 0], [0, 0], [1, 1], color=TRUE_GREEN, alpha=0.4)
    ax.fill_between([0, 1], [-1, -1], [0, 0], color=TRUE_GREEN, alpha=0.4)
    ax.fill_between([-1, 0], [-1, -1], [0, 0], color=FALSE_RED, alpha=0.4)
    ax.fill_between([0, 1], [0, 0], [1, 1], color=FALSE_RED, alpha=0.4)
    
    # Annotation
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # axs[0].grid()
    # axs[0].set_title('XOR Dataset')
    ax.set_xticks(np.arange(-1, 1.1, 1))
    ax.set_yticks(np.arange(-1, 1.1, 1))
    
    fig.savefig("../figures/xor/only_data.png", dpi=300)

def plot_with_classifier(n: int = 0, plot_glyph: bool=False):
    X, y = make_xor_dataset(n=100, sampling="mesh")
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 10*cm))
    color = np.full((len(y), 3), FALSE_RED)
    color[y] = TRUE_GREEN
    ax.scatter(X[:, 0], X[:, 1], c=color, s=4)
    
    # Shade area
    ax.fill_between([-1, 0], [0, 0], [1, 1], color=TRUE_GREEN, alpha=0.4)
    ax.fill_between([0, 1], [-1, -1], [0, 0], color=TRUE_GREEN, alpha=0.4)
    ax.fill_between([-1, 0], [-1, -1], [0, 0], color=FALSE_RED, alpha=0.4)
    ax.fill_between([0, 1], [0, 0], [1, 1], color=FALSE_RED, alpha=0.4)
        
    ## Classifiers
    h0, eps_set = example_baseline_and_epsilon_set()
    
    if plot_glyph:
        eps_set = eps_set[:n]
        idz = [16, 24, 28, 31, 54, 58, 61, 85]
        X_custom, y_custom = X[idz], y[idz]
        # Draw glyphs
        for i in range(X_custom.shape[0]):
            draw_binary_glyph(ax, X_custom[i, 0], X_custom[i, 1], [h.predict(X_custom[i]) for h in eps_set], h0.predict(X_custom[i]), y_custom[i])

    X_plot, Y_plot = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    X_concat = np.c_[X_plot.ravel(), Y_plot.ravel()]
    # Baseline
    Z = h0.decision_function(X_concat).reshape(X_plot.shape)
    ax.contour(X_plot, Y_plot, Z, colors=["k"], linestyles=['--'], levels=[0])
    # Epsilon Set
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    for i in range(n):
        clf = eps_set[i]
        Z = clf.decision_function(X_concat).reshape(X_plot.shape)
        ax.contour(X_plot, Y_plot, Z, colors=[colors[i]], linestyles=['--'], levels=[0])


    
    # Annotation
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # axs[0].grid()
    # axs[0].set_title('XOR Dataset')
    ax.set_xticks(np.arange(-1, 1.1, 1))
    ax.set_yticks(np.arange(-1, 1.1, 1))
    
    fig.savefig(f"../figures/xor/with_{n}_classifiers.png", dpi=300)

def plot_pm_for_all():
    X, y = make_xor_dataset(n=100, sampling="mesh")
    idz = [16, 24, 28, 31, 54, 58, 61, 85]
    X_custom, y_custom = X, y
    h0, eps_set = example_baseline_and_epsilon_set()

    # X, y = make_marx_dataset(n=10)
    # h0, eps_set = fit_baseline_and_epsilon_set(X, y, n_representatives=3)

    # Plot
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(15*cm, 10*cm))
    gs = GridSpec(1, 2, width_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    axs = np.array([ax0, ax1])
    
    # Draw glyphs
    for i in range(X_custom.shape[0]):
        draw_binary_glyph(axs[0], X_custom[i, 0], X_custom[i, 1], [h.predict(X_custom[i]) for h in eps_set], h0.predict(X_custom[i]), y_custom[i])
    # Annotation
    explain_binary_glyph(axs[1])

    # Draw additional points (which could be training points)
    color = np.full((len(y), 3), FALSE_RED)
    color[y] = TRUE_GREEN
    axs[0].scatter(X[:, 0], X[:, 1], c=color, s=4)

    # Calculate accuracy of predictors
    print(f"Baseline accuracy: {h0.score(X, y)}")
    for i, clf in enumerate(eps_set):
        print(f"Epsilon set classifier {i+1} accuracy: {clf.score(X, y)}")


    # Shade area
    axs[0].fill_between([-1, 0], [0, 0], [1, 1], color=TRUE_GREEN, alpha=0.4)
    axs[0].fill_between([0, 1], [-1, -1], [0, 0], color=TRUE_GREEN, alpha=0.4)
    axs[0].fill_between([-1, 0], [-1, -1], [0, 0], color=FALSE_RED, alpha=0.4)
    axs[0].fill_between([0, 1], [0, 0], [1, 1], color=FALSE_RED, alpha=0.4)


    # Show the decision boundary
    X_plot, Y_plot = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    X_concat = np.c_[X_plot.ravel(), Y_plot.ravel()]
    # Baseline classifier
    Z = h0.decision_function(X_concat).reshape(X_plot.shape)
    axs[0].contour(X_plot, Y_plot, Z, colors=["k"], linestyles=['--'], levels=[0])
    # Epsilon set classifiers
    for clf, color in zip(eps_set, ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
        Z = clf.decision_function(X_concat).reshape(X_plot.shape)
        axs[0].contour(X_plot, Y_plot, Z, colors=[color], linestyles=['--'], levels=[0])


    # Annotation
    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)
    # axs[0].grid()
    # axs[0].set_title('XOR Dataset')
    axs[0].set_xticks(np.arange(-1, 1.1, 1))
    axs[0].set_yticks(np.arange(-1, 1.1, 1))

    # Add the custom legend to the second axis
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TRUE_GREEN, markersize=5, label='true $(1)$'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=FALSE_RED, markersize=5, label='false $(-1)$'),
        plt.Line2D([0, 1], [0, 1], color='grey', linestyle='--', label='DB of $h_i$'),
    ]
    axs[1].legend(handles=legend_elements, loc='lower center', ncol=1)
    
    
    plt.tight_layout()
    # plt.show()
    fig.savefig("../figures/xor/pm_for_all.png", dpi=300)

def plot_pred_for_baseline():
    X, y = make_xor_dataset(n=100, sampling="mesh")
    fig, ax = plt.subplots(1, 1, figsize=(10*cm, 10*cm))
    color = np.full((len(y), 3), FALSE_RED)
    color[y] = TRUE_GREEN
    ax.scatter(X[:, 0], X[:, 1], c=color, s=4)
    
    ## Classifiers
    h0, eps_set = example_baseline_and_epsilon_set()

    X_plot, Y_plot = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    X_concat = np.c_[X_plot.ravel(), Y_plot.ravel()]
    # Baseline
    Z = h0.decision_function(X_concat).reshape(X_plot.shape)
    ax.contour(X_plot, Y_plot, Z, colors=["k"], linestyles=['--'], levels=[0])
    
    # Shade area
    ax.fill_between([-1, 1], [0, 0], [1, 1], color=TRUE_GREEN, alpha=0.4)
    ax.fill_between([-1, 1], [-1, -1], [0, 0], color=FALSE_RED, alpha=0.4)
    
    # Annotation
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    # axs[0].grid()
    # axs[0].set_title('XOR Dataset')
    ax.set_xticks(np.arange(-1, 1.1, 1))
    ax.set_yticks(np.arange(-1, 1.1, 1))
    fig.savefig("../figures/xor/prediction_baseline.png", dpi=300)

def main():
    X, y = make_xor_dataset(n=100, sampling="mesh")
    idz = [16, 24, 28, 31, 54, 58, 61, 85]
    X_custom, y_custom = X[idz], y[idz]
    h0, eps_set = example_baseline_and_epsilon_set()

    # X, y = make_marx_dataset(n=10)
    # h0, eps_set = fit_baseline_and_epsilon_set(X, y, n_representatives=3)

    # Plot
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(15*cm, 10*cm))
    gs = GridSpec(1, 2, width_ratios=[2, 1])

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    axs = np.array([ax0, ax1])
    
    # Draw glyphs
    for i in range(X_custom.shape[0]):
        draw_binary_glyph(axs[0], X_custom[i, 0], X_custom[i, 1], [h.predict(X_custom[i]) for h in eps_set], h0.predict(X_custom[i]), y_custom[i])
    # Annotation
    explain_binary_glyph(axs[1])

    # Draw additional points (which could be training points)
    color = np.full((len(y), 3), FALSE_RED)
    color[y] = TRUE_GREEN
    axs[0].scatter(X[:, 0], X[:, 1], c=color, s=4)

    # Calculate accuracy of predictors
    print(f"Baseline accuracy: {h0.score(X, y)}")
    for i, clf in enumerate(eps_set):
        print(f"Epsilon set classifier {i+1} accuracy: {clf.score(X, y)}")


    # Shade area
    axs[0].fill_between([-1, 0], [0, 0], [1, 1], color=TRUE_GREEN, alpha=0.4)
    axs[0].fill_between([0, 1], [-1, -1], [0, 0], color=TRUE_GREEN, alpha=0.4)
    axs[0].fill_between([-1, 0], [-1, -1], [0, 0], color=FALSE_RED, alpha=0.4)
    axs[0].fill_between([0, 1], [0, 0], [1, 1], color=FALSE_RED, alpha=0.4)


    # Show the decision boundary
    X_plot, Y_plot = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    X_concat = np.c_[X_plot.ravel(), Y_plot.ravel()]
    # Baseline classifier
    Z = h0.decision_function(X_concat).reshape(X_plot.shape)
    axs[0].contour(X_plot, Y_plot, Z, colors=["k"], linestyles=['--'], levels=[0])
    # Epsilon set classifiers
    for clf, color in zip(eps_set, ["tab:blue", "tab:orange", "tab:green", "tab:purple"]):
        Z = clf.decision_function(X_concat).reshape(X_plot.shape)
        axs[0].contour(X_plot, Y_plot, Z, colors=[color], linestyles=['--'], levels=[0])


    # Annotation
    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)
    # axs[0].grid()
    # axs[0].set_title('XOR Dataset')
    axs[0].set_xticks(np.arange(-1, 1.1, 1))
    axs[0].set_yticks(np.arange(-1, 1.1, 1))

    # Add the custom legend to the second axis
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=TRUE_GREEN, markersize=5, label='true $(1)$'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=FALSE_RED, markersize=5, label='false $(-1)$'),
        plt.Line2D([0, 1], [0, 1], color='grey', linestyle='--', label='DB of $h_i$'),
    ]
    axs[1].legend(handles=legend_elements, loc='lower center', ncol=1)
    
    
    plt.tight_layout()
    # plt.show()
    fig.savefig("../figures/xor.pdf")
    fig.savefig("../figures/xor.png", dpi=300)

if __name__ == "__main__":
    main()
    plot_only_dataset()
    plot_pred_for_baseline()
    plot_pm_for_all()
    for n in range(4):
        plot_with_classifier(n)