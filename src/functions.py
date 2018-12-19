import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.calibration import calibration_curve
from sklearn.utils import deprecated

from scipy import interp

from scikitplot.helpers import binary_ks_curve, validate_labels

def cumulative_gain_curve(y_true, y_score, pos_label=None, optimal=False):
    """This is traight up stolen from skplt.metrics, but adsjusted to the needs
    of the purposes in this project.

    This function generates the points necessary to plot the Cumulative Gain
    Note: This implementation is restricted to the binary classification task.
    Args:
        y_true (array-like, shape (n_samples)): True labels of the data.
        y_score (array-like, shape (n_samples)): Target scores, can either be
            probability estimates of the positive class, confidence values, or
            non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).
        pos_label (int or str, default=None): Label considered as positive and
            others are considered negative
    Returns:
        percentages (numpy.ndarray): An array containing the X-axis values for
            plotting the Cumulative Gains chart.
        gains (numpy.ndarray): An array containing the Y-axis values for one
            curve of the Cumulative Gains chart.
    Raises:
        ValueError: If `y_true` is not composed of 2 classes. The Cumulative
            Gain Chart is only relevant in binary classification.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    if optimal == False:
        # make y_true a boolean vector
        y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)


    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    # else:
    #     # make y_true a boolean vector
    #     #y_true = (y_true == pos_label)

    #     sorted_indices = np.argsort(y_score)[::-1]

    #     print (y_score[sorted_indices])

    #     y_true = y_true
    #     gains = np.cumsum(y_true[sorted_indices])

    #     percentages = np.arange(start=1, stop=len(y_true) + 1)

    #     gains = gains / float(np.sum(y_true))
    #     percentages = percentages / float(len(y_true))

    #     gains = np.insert(gains, 0, [0])
    #     percentages = np.insert(percentages, 0, [0])

    return percentages, gains

def plot_cumulative_gain(y_true, y_probas, title='Cumulative Gains Curve',
                         ax=None, figsize=None, title_fontsize="large",
                         text_fontsize="medium"):
    """ This is traight up stolen from skplt.metrics, but adsjusted to the needs
    of the purposes in this project.

    Generates the Cumulative Gains Plot from labels and scores/probabilities
    The cumulative gains chart is used to determine the effectiveness of a
    binary classifier. A detailed explanation can be found at
    http://mlwiki.org/index.php/Cumulative_Gain_Chart. The implementation
    here works only for binary classification.
    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.
        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.
        title (string, optional): Title of the generated plot. Defaults to
            "Cumulative Gains Curve".
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the learning curve. If None, the plot is drawn on a new set of
            axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> lr = LogisticRegression()
        >>> lr = lr.fit(X_train, y_train)
        >>> y_probas = lr.predict_proba(X_test)
        >>> skplt.metrics.plot_cumulative_gain(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_cumulative_gain.png
           :align: center
           :alt: Cumulative Gains Plot
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves

    print (' \n')

    # a = y_probas[:, 0]
    # print (a[0])

    percentages, gains_true = cumulative_gain_curve(y_true, y_true,
                                                classes[0], optimal=True)

    percentages, gains1 = cumulative_gain_curve(y_true, y_probas[:, 0],
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])

    # print (1 - gains_true[::-1])
    # gains_true = 1 - gains_true[::-1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains_true, lw=3, label='{}'.format('Optimal'))
    #ax.plot(percentages, gains1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(percentages, gains2, lw=3, label='Class {}'.format(classes[1]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.02, 1.02])

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax

def area_ratio(y_true, y_probas):

    """
    Same as 

    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves

    percentages, gains_true = cumulative_gain_curve(y_true, y_true,
                                                classes[0], optimal=True)
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas[:, 1],
                                                classes[1])
    dx = percentages[1] - percentages[0]

    gains_true_area = 0
    gains2_area = 0


    for i in range(0, len(gains2)):
        gains_true_area += dx * gains_true[i] - dx * percentages[i]
        gains2_area += dx * gains2[i] - dx * percentages[i]

    return gains2_area/float(gains_true_area)