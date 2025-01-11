# Simple SVM Classification for XOR dataset using sklearn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn import svm
from utils import Custom_SVM

from plot_glyph import draw_binary_glyph, explain_binary_glyph, TRUE_GREEN, FALSE_RED

def make_dataset(n: int = 64, sampling: str="marx"):
    if sampling == "mesh":
        X = np.meshgrid(np.linspace(-1, 1, int(np.sqrt(n))), np.linspace(-1, 1, int(np.sqrt(n))))
        X = np.array(X).reshape(2, -1).T
    elif sampling == "random":
        rng = np.random.default_rng(424)
        X = 2*rng.random((n, 2)) - 1
    elif sampling == "custom":
        X = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5], [0.3, 0.7], [-0.75, 0.25], [-0.1, 0.9], [0.9, -0.1]])
    y = X[:, 0] > 0
    return X, y

def make_diag_dataset(n: int = 10):
    # Example dataset similar to the one in marx Figure 2
    X_m = np.array([- np.random.rand(n//2), np.random.rand(n//2)]).T
    X_p = np.array([np.random.rand(n//2), - np.random.rand(n//2)]).T
    X = np.concatenate([X_m, X_p])
    y = np.array([False] * (n//2) + [True] * (n//2))
    return X, y


def example_baseline_and_epsilon_set():
    h0 = Custom_SVM(w=np.array([1, -1]), b=0)
    eps_set = []
    for w in [[1, -4], [1, -2], [2, -1], [4, -1]]:
        clf = Custom_SVM(w=np.array(w), b=0)
        eps_set.append(clf)
    return h0, eps_set


def main():
    X_custom, y_custom = make_dataset(n=15, sampling="random")
    X, y = make_diag_dataset(n=100)
    h0, eps_set = example_baseline_and_epsilon_set()

    # X, y = make_marx_dataset(n=10)
    # h0, eps_set = fit_baseline_and_epsilon_set(X, y, n_representatives=3)

    # Plot
    cm = 1/2.54  # centimeters in inches
    fig = plt.figure(figsize=(10*cm, 10*cm))

    ax0 = fig.add_subplot()
    axs = np.array([ax0])
    
    

    # Draw additional points (which could be training points)
    color = np.full((len(y), 3), FALSE_RED)
    color[y] = TRUE_GREEN
    axs[0].scatter(X[:, 0], X[:, 1], c=color, s=4)

    # Calculate accuracy of predictors
    print(f"Baseline accuracy: {h0.score(X, y)}")
    for i, clf in enumerate(eps_set):
        print(f"Epsilon set classifier {i+1} accuracy: {clf.score(X, y)}")


    # Shade area
    axs[0].fill_between([-1, 1], [-1, 1], [1, 1], color=FALSE_RED, alpha=0.4)
    axs[0].fill_between([-1, 1], [-1, -1], [-1, 1], color=TRUE_GREEN, alpha=0.4)


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
    
    # Draw glyphs
    for i in range(X_custom.shape[0]):
        draw_binary_glyph(axs[0], X_custom[i, 0], X_custom[i, 1], [h.predict(X_custom[i]) for h in eps_set], h0.predict(X_custom[i]), y_custom[i])
    

    # Annotation axs[0]
    axs[0].set_xlabel('$x_1$')
    axs[0].set_ylabel('$x_2$')
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)
    # axs[0].grid()
    # axs[0].set_title('XOR Dataset')
    axs[0].set_xticks(np.arange(-1, 1.1, 1))
    axs[0].set_yticks(np.arange(-1, 1.1, 1))


    plt.tight_layout()
    # plt.show()
    fig.savefig("../figures/diag_border.pdf")

if __name__ == "__main__":
    main()