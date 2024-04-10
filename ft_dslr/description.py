import pandas
import math


def describe_csv(dtf: pandas.DataFrame, verbose: bool = True):
    describe_dtf = pandas.DataFrame()
    dtf = dtf.drop(columns='Index', inplace=False)
    nb_features = dtf.select_dtypes(include=['number']).columns.tolist()
    describe_dtf['info'] = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']

    for feature in nb_features:
        try:
            describe_dtf[feature] = describe_column_nb(dtf, feature)
        except Exception:
            pass
    if verbose:
        print(describe_dtf.to_string(index=False))
    return describe_dtf


def describe_column_nb(dtf: pandas.DataFrame, column: str) -> []:
    """Get the description of a column"""
    dtf_cleaned = pandas.DataFrame(dtf[column].copy())
    dtf_cleaned.dropna(inplace=True)
    dtf_cleaned.sort_values(by=column, ascending=True, inplace=True)
    dtf_cleaned.reset_index(drop=True, inplace=True)

    count, mean, sum_v, min_v, max_v = extract_basics_info(dtf_cleaned, column)
    std = compute_std(dtf_cleaned, column, mean, count)
    twenty_five = compute_quantile(dtf_cleaned, column, count, 0.25)
    fifty = compute_quantile(dtf_cleaned, column, count, 0.5)
    seventy_five = compute_quantile(dtf_cleaned, column, count, 0.75)

    return count, mean, std, min_v, twenty_five, fifty, seventy_five, max_v


def extract_basics_info(dtf: pandas.DataFrame, column: str) -> []:
    """Extract the maximum information of a column in a single loop.
    :return : count of items, the mean, the total sum, the min and max value"""
    count, mean, sum_v, min_v, max_v = 0, 0, 0, 0, 0
    for index, value in dtf[column].items():
        count = count + 1
        if count == 1:
            min_v = value
            max_v = value
        if value < min_v:
            min_v = value
        if value > max_v:
            max_v = value
        sum_v += value

    mean = sum_v / count
    return count, mean, sum_v, min_v, max_v


def compute_std(dtf: pandas.DataFrame, column: str, mean: float, count: int) -> []:
    """Compute the standard deviation of a column"""
    sum_tmp = 0
    for index, value in dtf[column].items():
        sum_tmp = sum_tmp + (value - mean) * (value - mean)
    std = math.sqrt(sum_tmp / count)
    return std


def compute_quantile(dtf: pandas.DataFrame, column: str, count: int, quantile: float) -> []:
    """Compute the quantile of a column"""
    quantile_idx = quantile * count                 # find the quantile index (expressed in float)
    if quantile_idx.is_integer():                   # access directly to the value if the quantile_idx can be an int
        quantile_value = dtf.loc[quantile_idx, column]
    elif quantile_idx - int(quantile_idx) < 0.5:    # find the nearst value if the quantile_idx cant be an int
        quantile_value = dtf.loc[int(quantile_idx), column]
    else:
        quantile_value = dtf.loc[math.ceil(quantile_idx), column]
    return quantile_value


def csv_to_dataframe(csv_file: str) -> pandas.DataFrame:
    """Converts a CSV file into a Pandas dataframe"""
    dtf = pandas.read_csv(csv_file)
    return dtf
