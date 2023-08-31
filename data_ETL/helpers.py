import sklearn.linear_model as linear
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from typing import Callable
import random
import patsy
import scipy.stats as stats
import pandas as pd
import numpy as np

ALGORITHMS = {
    "linear": linear.LinearRegression,
    "ridge": linear.Ridge,
    "lasso": linear.Lasso
}


def freeman_diaconis(data):
    quartiles = stats.mstats.mquantiles(data, [0.25, 0.5, 0.75])
    iqr = quartiles[2] - quartiles[0]
    n = len(data)
    h = 2.0 * (iqr / n ** (1.0 / 3.0))
    return int(h)


def plot_hist_numeric(df: pd.DataFrame, my_col: str, backup_step=2):
    t_col = df[my_col]
    mn = int(t_col.min())
    mx = int(t_col.max())
    h = freeman_diaconis(t_col)
    if h == 0: h = backup_step
    bins = [i for i in range(mn, mx, h)]

    figure = plt.figure(figsize=(10, 6))

    axes = figure.add_subplot(1, 1, 1)
    axes.hist(t_col, bins=bins, color="darkslategray")
    axes.set_title(' '.join([my_col.title(), 'Distribution\n (Freeman Diaconis)']))
    axes.set_xlabel(my_col.title())

    plt.show()
    plt.close()
    return


def plot_hist_categorical(df: pd.DataFrame, t_col: str, do_normalize: bool):
    data = df[t_col].value_counts(normalize=do_normalize)
    x = list(data.index.sort_values())
    width = 1 / 1.5
    figure = plt.figure(figsize=(8, 6))

    axes = figure.add_subplot(1, 1, 1)
    axes.bar(x, data, width, align="center", color="darkslategray")
    axes.set_xticks(x)
    axes.set_xticklabels(data.axes[0])
    axes.set_title(' '.join(['Distribution of', t_col.title()]))
    axes.set_xlabel(t_col.title())
    axes.set_ylabel('Percent' if do_normalize else 'Count')
    axes.xaxis.grid(False)

    plt.show()
    plt.close()
    return


def get_correlations_en_masse(data, y, xs: list) -> pd.DataFrame:
    rs = []
    rhos = []
    for x in xs:
        r = stats.pearsonr(data[y], data[x])[0]
        rs.append(r)
        rho = stats.spearmanr(data[y], data[x])[0]
        rhos.append(rho)
    return pd.DataFrame({"feature": xs, "r": rs, "rho": rhos})


def get_correlations(df: pd.DataFrame, colA: str, colB: str) -> dict:
    results = {}
    results['pearson'] = stats.pearsonr(df[colA], df[colB])[0]
    results['spearman'] = stats.spearmanr(df[colA], df[colB])[0]
    return results


def describe_by_category(my_data: pd.DataFrame, numeric: str, categorical: str, transpose=False):
    t_grouped = my_data.groupby(categorical)
    t_grouped_y = t_grouped[numeric].describe()
    if transpose:
        print(t_grouped_y.transpose())
    else:
        print(t_grouped_y)
    return t_grouped


def plot_scatter(my_data: pd.DataFrame, y_col: str, x_col: str):
    figure = plt.figure(figsize=(8, 6))
    axes = figure.add_subplot(1, 1, 1)
    axes.scatter(y=my_data[y_col], x=my_data[x_col], marker='o', color='darkslategray')
    axes.set_ylabel(y_col.title())
    axes.set_xlabel(x_col.title())
    axes.set_title(' '.join([y_col, 'vs.', x_col]))

    plt.show()
    plt.close()


def plot_by_category(my_data: pd.DataFrame, response_col: str, explanatory_col: str, relative: bool):
    n_cols = 3
    h = freeman_diaconis(my_data[response_col])
    grouped = my_data.groupby(explanatory_col)
    figure = plt.figure(figsize=(20, 6))

    n_rows = math.ceil(grouped.ngroups / n_cols)

    for plot_index, k in enumerate(grouped.groups.keys()):
        axes = figure.add_subplot(n_rows, n_cols, plot_index + 1)
        axes.hist(grouped[response_col].get_group(k), bins=h, color="darkslategray", density=relative, range=(0, 40))
        axes.set_title(
            ' '.join([str(k), explanatory_col.title(), '-', response_col.title(), '\ndistribution - Freeman Diaconis']))
        axes.set_xlabel(response_col)

    figure.tight_layout()
    plt.show()
    plt.close()
    return


def linear_regression(formula, data=None, style="linear", params={}):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    params["fit_intercept"] = False

    y, X = patsy.dmatrices(formula, data, return_type="matrix")
    algorithm = ALGORITHMS[style]
    algo = algorithm(**params)
    model = algo.fit(X, y)

    result = summarize(formula, X, y, model, style)

    return result


def bootstrap_linear_regression(formula, data=None, samples=100, style="linear", params={}):
    if data is None:
        raise ValueError("The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")

    bootstrap_results = {}
    bootstrap_results["formula"] = formula

    variables = [x.strip() for x in formula.split("~")[1].split("+")]
    variables = ["intercept"] + variables
    bootstrap_results["variables"] = variables

    coeffs = []
    sigmas = []
    rs = []

    n = len(data)
    bootstrap_results["n"] = n

    for i in range(samples):
        sampling = data.sample(len(data), replace=True)
        results = linear_regression(formula, data=sampling, style=style, params=params)
        coeffs.append(results["coefficients"])
        sigmas.append(results["sigma"])
        rs.append(results["r_squared"])

    coeffs = pd.DataFrame(coeffs, columns=variables)
    sigmas = pd.Series(sigmas, name="sigma")
    rs = pd.Series(rs, name="r_squared")

    bootstrap_results["resampled_coefficients"] = coeffs
    bootstrap_results["resampled_sigma"] = sigmas
    bootstrap_results["resampled_r^2"] = rs

    result = linear_regression(formula, data=data)

    bootstrap_results["residuals"] = result["residuals"]
    bootstrap_results["coefficients"] = result["coefficients"]
    bootstrap_results["sigma"] = result["sigma"]
    bootstrap_results["r_squared"] = result["r_squared"]
    bootstrap_results["model"] = result["model"]
    bootstrap_results["y"] = result["y"]
    bootstrap_results["y_hat"] = result["y_hat"]
    return bootstrap_results

def fmt(n, sd=2):
    return (r"{0:." + str(sd) + "f}").format(n)


def boldify(xs, format):
    if format == "html":
        return ["<strong>" + x + "</strong>" if x != "" else "" for x in xs]
    if format == "markdown":
        return ["**" + x + "**" if x != "" else "" for x in xs]
    # latex
    return ["\\textbf{" + x + "}" if x != "" else "" for x in xs]


def results_table(fit, sd=2, bootstrap=False, is_logistic=False, format="html"):
    result = {}
    result["model"] = [fit["formula"]]

    variables = [v.strip() for v in [""] + fit["formula"].split("~")[1].split("+")]
    if format == 'latex':
        variables = [v.replace("_", "\\_") for v in variables]
    coefficients = []

    if bootstrap:
        bounds = fit["resampled_coefficients"].quantile([0.025, 0.975])
        bounds = bounds.transpose()
        bounds = bounds.values.tolist()
        for i, b in enumerate(zip(variables, fit["coefficients"], bounds)):
            coefficient = [b[0], f"$\\beta_{{{i}}}$", fmt(b[1], sd), fmt(b[2][0], sd), fmt(b[2][1], sd)]
            if is_logistic:
                if i == 0:
                    pass
                else:
                    coefficient.append(fmt(b[1] / 4, sd))
            coefficients.append(coefficient)
    else:
        for i, b in enumerate(zip(variables, fit["coefficients"])):
            coefficients.append([b[0], f"$\\beta_{{{i}}}$", fmt(b[1], sd)])
    result["coefficients"] = coefficients

    error = r"$\sigma$"
    r_label = r"$R^2$"
    if is_logistic:
        error = "Error (%)"
        r_label = r"Efron's $R^2$"
    if bootstrap:
        sigma_bounds = stats.mstats.mquantiles(fit["resampled_sigma"], [0.025, 0.975])
        r_bounds = stats.mstats.mquantiles(fit["resampled_r^2"], [0.025, 0.975])
        metrics = [
            [error, fmt(fit["sigma"], sd), fmt(sigma_bounds[0], sd), fmt(sigma_bounds[1], sd)],
            [r_label, fmt(fit["r_squared"], sd), fmt(r_bounds[0], sd), fmt(r_bounds[1], sd)]]
    else:
        metrics = [
            [error, fmt(fit["sigma"], sd)],
            [r_label, fmt(fit["r_squared"], sd)]]

    result["metrics"] = metrics

    title = f"Model: {result['model'][0]}"
    rows = []
    if bootstrap:
        rows.append(boldify(["", "", "", "95% BCI"], format))
    if is_logistic:
        if bootstrap:
            header = boldify(["Coefficients", "", "Mean", "Lo", "Hi", "P(y=1)"], format)
        else:
            header = boldify(["Coefficients", "", "Value"], format)
    else:
        if bootstrap:
            header = boldify(["Coefficients", "", "Mean", "Lo", "Hi"], format)
        else:
            header = boldify(["Coefficients", "", "Value"], format)
    rows.append(header)

    for row in result["coefficients"]:
        rows.append(row)

    rows.append([])

    if bootstrap:
        rows.append(boldify(["Metrics", "Mean", "Lo", "Hi"], format))
    else:
        rows.append(boldify(["Metrics", "Value"], format))
    for row in result["metrics"]:
        rows.append(row)

    return title, rows



class ResultsWrapper(object):
    def __init__(self, fit, sd=2, bootstrap=False, is_logistic=False):
        self.fit = fit
        self.sd = sd
        self.bootstrap = bootstrap
        self.is_logistic = is_logistic

    def _repr_markdown_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="markdown")
        table = tabulate(table, tablefmt="github")
        markdown = title + "\n" + table
        return markdown

    def _repr_html_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="html")
        table = tabulate(table, tablefmt="html")
        table = table.replace("&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong")
        return f"<p><strong>{title}</strong><br/>{table}</p>"

    def _repr_latex_(self):
        title, table = results_table(self.fit, self.sd, self.bootstrap, self.is_logistic, format="latex")

        title = title.replace("~", "$\\sim$").replace("_", "\\_")

        table = tabulate(table, tablefmt="latex_booktabs")
        table = table.replace("textbackslash{}", "").replace("\^{}", "^").replace("\_", "_")
        table = table.replace("\\$", "$").replace("\\{", "{").replace("\\}", "}")
        latex = "\\textbf{" + title + "}\n\n" + table
        return latex

def describe_bootstrap_lr(fit, sd=2):
    return ResultsWrapper(fit, sd, True, False)


def simple_describe_lr(fit, sd=2):
    return ResultsWrapper(fit, sd)


def summarize(formula, X, y, model, style='linear'):
    result = {}
    result["formula"] = formula
    result["n"] = len(y)
    result["model"] = model
    # I think this is a bug in Scikit Learn
    # because lasso should work with multiple targets.
    if style == "lasso":
        result["coefficients"] = model.coef_
    else:
        result["coefficients"] = model.coef_[0]
    result["r_squared"] = model.score(X, y)
    y_hat = model.predict(X)
    result["residuals"] = y - y_hat
    result["y_hat"] = y_hat
    result["y"] = y
    sum_squared_error = sum([e ** 2 for e in result["residuals"]])[0]

    n = len(result["residuals"])
    k = len(result["coefficients"])

    result["sigma"] = np.sqrt(sum_squared_error / (n - k))
    return result


def correlations(data, y, xs):
    rs = []
    rhos = []
    for x in xs:
        r = stats.pearsonr(data[y], data[x])[0]
        rs.append(r)
        rho = stats.spearmanr(data[y], data[x])[0]
        rhos.append(rho)
    return pd.DataFrame({"feature": xs, "r": rs, "rho": rhos})


def plot_residuals(data, result, variables):
    figure = plt.figure(figsize=(20, 8))

    plots = len(variables)
    rows = (plots // 3) + 1

    residuals = np.array([r[0] for r in result['residuals']])
    limits = max(np.abs(residuals.min()), residuals.max())

    n = result["n"]
    for i, variable in enumerate(variables):
        axes = figure.add_subplot(rows, 3, i + 1)

        keyed_values = sorted(zip(data[variable].values, residuals), key=lambda x: x[0])
        ordered_residuals = [x[1] for x in keyed_values]

        axes.plot(list(range(0, n)), ordered_residuals, '.', color='dimgray', alpha=0.75)
        axes.axhline(y=0.0, xmin=0, xmax=n, c='firebrick', alpha=0.5)
        axes.set_ylim((-limits, limits))
        axes.set_ylabel('residuals')
        axes.set_xlabel(variable)

    figure.tight_layout(pad=2.0)
    plt.show()
    plt.close()
    return residuals

def sse(results):
    errors = results['residuals']
    n = len(errors)
    squared_error = np.sum([e ** 2 for e in errors])
    return np.sqrt((1.0 / n) * squared_error)


def r2(results):
    return np.mean(results['r_squared'])


def sigma(results):
    return np.mean(results['sigma'])


def chunk(xs, n):
    k, m = divmod(len(xs), n)
    return [xs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def resample(data):
    n = len(data)
    return [data[i] for i in [stats.randint.rvs(0, n - 1) for _ in range(0, n)]]

def cross_validation(algorithm: Callable, formula: str, data: pd.DataFrame,
                     fold_count=10, repetitions=3) -> dict:
    indices = list(range(len(data)))
    metrics = {'sse_metric': [], 'r2_metric': [], 'sigma_metric': []}
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = chunk(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            result = algorithm(formula, data=train_data)
            t_model = result["model"]
            y, X = patsy.dmatrices(formula, test_data, return_type="matrix")
            results = summarize(formula, X, y, t_model)
            metrics['sse_metric'].append(sse(results))
            metrics['r2_metric'].append(r2(results))
            metrics['sigma_metric'].append(sigma(results))
    return metrics

