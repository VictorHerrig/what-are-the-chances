import numpy as np
import matplotlib.pyplot as plt


def initialize(p_roll, roll_guarantee, p_roll_value, value_guarantee, p_roll_sub_value=None):
    max_steps = roll_guarantee * value_guarantee
    max_positives = (1 if p_roll_value >= 1. else value_guarantee) + 1
    p_roll = np.repeat(p_roll, roll_guarantee)
    p_roll[-1] = 1.  # 100% at pity

    if p_roll_sub_value is None:
        # Columns: No 5*, negative 5* #1, ..., positive 5*
        p0 = np.zeros((roll_guarantee, max_positives))
    else:
        # As above, but with an extra axis determining whether the sub-value has been encountered
        p0 = np.zeros((roll_guarantee, max_positives, 2))
    p0[0, 0] = 1.

    prob_series = np.zeros(max_steps)

    return prob_series, p0, p_roll


def proportions(p0):
    """

    Parameters
    ----------
    p0: np.ndarray

    Returns
    -------

    """
    prob_per_step = p0.sum(1)
    rownorm = p0 / prob_per_step.reshape(-1, 1)
    rownorm[np.isnan(rownorm)] = 0.
    return prob_per_step, rownorm


def simple_positive_roll(stepprob, rownorm, p_roll, p_roll_value):
    positive_roll = stepprob * p_roll
    negative_value = (rownorm[:, :-2] * positive_roll.reshape(-1, 1) * (1 - p_roll_value)).sum(0)
    at_least_one_positive_value = np.array([(rownorm[:, :-2] * positive_roll.reshape(-1, 1) * p_roll_value).sum() +
                                            (rownorm[:, -2:] * positive_roll.reshape(-1, 1)).sum()])
    return np.concatenate((negative_value, at_least_one_positive_value))


def double_value_positive_roll(stepprob, rownorm, p_roll, p_roll_value, p_roll_sub_value):
    negative_value, first_order_positive = np.split(simple_positive_roll(stepprob, rownorm, p_roll, p_roll_value), [1])
    second_order_negative = first_order_positive * p_roll_sub_value.sum


def negative_roll(p0, p_roll):
    return (p0 * (1 - p_roll).reshape(-1, 1))[:-1]


def simple_cumprob(p_roll, roll_guarantee, p_roll_value, value_guarantee):
    """

    Parameters
    ----------
    p_roll: float
        Probability per step.
    roll_guarantee: int
        Number of steps after which a positive is guaranteed.
    p_roll_value: float
        Probability that a positive is the desired value.
    value_guarantee: int
        Number of positives after which the desired value is guaranteed.

    Returns
    -------
    numpy.ndarray
        CDF array.
    """
    # Doesn't take any (unpublished) graduated rate change into account
    # Is there a dataset somewhere from which the graduated change can be inferred?
    prob_series, p0, p_roll = initialize(p_roll, roll_guarantee, p_roll_value, value_guarantee)

    for i in range(prob_series.shape[0]):
        # Calculate next step probability density
        p1 = np.zeros_like(p0)
        prob_per_step, rownorm = proportions(p0)
        p1[0, 1:] = simple_positive_roll(prob_per_step, rownorm, p_roll, p_roll_value)
        p1[1:] = negative_roll(p0, p_roll)

        # Log and update
        prob_series[i] = p1[:, -1].sum()
        p0 = p1

    return prob_series


if __name__ == '__main__':
    ps = simple_cumprob(0.006, 90, 1, 1)
    _ = plt.figure(figsize=(12, 10))
    plt.plot(np.arange(len(ps)) + 1, ps)
    plt.title('Chance of rolling any 5*')
    plt.xlabel('Rolls')
    plt.ylabel('Cumulative probability')
    plt.grid()
    plt.show()

    plt.clf()
    ps = simple_cumprob(0.006, 90, 0.5, 2)
    _ = plt.figure(figsize=(12, 10))
    plt.plot(np.arange(len(ps)) + 1, ps)
    plt.title('Chance of rolling the banner 5* character')
    plt.xlabel('Rolls')
    plt.ylabel('Cumulative probability')
    plt.grid()
    plt.show()

    plt.clf()
    ps = simple_cumprob(0.003, 90, 0.2, 25)
    _ = plt.figure(figsize=(12, 10))
    plt.plot(np.arange(len(ps)) + 1, ps)
    plt.title('Chance of rolling a particular 5* character from the standard banner')
    plt.xlabel('Rolls')
    plt.ylabel('Cumulative probability')
    plt.grid()
    plt.show()

    plt.clf()
    ps = simple_cumprob(0.003, 90, 0.1, 60)
    _ = plt.figure(figsize=(12, 10))
    plt.plot(np.arange(len(ps)) + 1, ps)
    plt.title('Chance of rolling a particular 5* weapon from the standard banner')
    plt.xlabel('Rolls')
    plt.ylabel('Cumulative probability')
    plt.grid()
    plt.show()
