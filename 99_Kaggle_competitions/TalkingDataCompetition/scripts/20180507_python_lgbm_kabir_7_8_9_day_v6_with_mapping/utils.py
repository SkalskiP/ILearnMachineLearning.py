import gc
import numpy as np

# ======================================================================================================================
# Show feature importance
# ======================================================================================================================


def show_features(features, features_importance, to_file=True, file_name="features_importance.txt"):
    sorted_index = np.argsort(features_importance)
    features = np.array(features)[sorted_index[::-1]]
    features_importance = np.array(features_importance)[sorted_index[::-1]]
    features_data = []

    for iterator, (f_name, f_val) in enumerate(zip(features, features_importance)):
        new_line = "{}. {} - {}".format(iterator, f_name, f_val)
        features_data.append(new_line)

    output = "\n".join(features_data)
    print(output)

    if to_file:
        with open(file_name, 'w') as file:
            file.write(output)


# ======================================================================================================================
# Calculate the time to next click for each group
# ======================================================================================================================


def do_next_click(df, agg_suffix='next_click'):
    print("\n>> Extracting {} time calculation features...\n".format(agg_suffix))

    group_by_next_clicks = [
        {'groupby': ['ip', 'os', 'device', 'app']},
        {'groupby': ['ip', 'os', 'app']},
        {'groupby': ['ip', 'os', 'device']},
        {'groupby': ['ip', 'os', 'channel']},
        {'groupby': ['ip', 'device']},
        {'groupby': ['ip', 'channel']},
        {'groupby': ['ip', 'app', 'channel']},
        {'groupby': ['ip', 'device', 'app']},
        {'groupby': ['ip', 'app']},
    ]

    for spec in group_by_next_clicks:
        new_feature_name = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)
        group_of_selected_features = spec['groupby'] + ['click_time_int']

        print(">> Grouping by {}, and saving time to {} in: {}".format(spec['groupby'], agg_suffix, new_feature_name))

        df[new_feature_name] = (df[group_of_selected_features].groupby(spec['groupby'])
                                .click_time_int.shift(-1) - df.click_time_int).astype(np.float32)

        gc.collect()

    return df

# ======================================================================================================================
# Calculate the time to previous click for each group
# ======================================================================================================================


def do_prev_click(df, agg_suffix='prev_click'):
    print("\n>> Extracting {} time calculation features...\n".format(agg_suffix))

    group_by_next_clicks = [
        {'groupby': ['ip', 'os', 'device', 'app']},
        {'groupby': ['ip', 'device', 'app']},
        {'groupby': ['ip', 'device', 'channel']},
        {'groupby': ['ip', 'app', 'channel']},
        {'groupby': ['ip', 'app']},
        {'groupby': ['ip', 'device']},
    ]

    for spec in group_by_next_clicks:
        new_feature_name = '{}_{}'.format('_'.join(spec['groupby']), agg_suffix)
        group_of_selected_features = spec['groupby'] + ['click_time_int']

        print(">> Grouping by {}, and saving time to {} in: {}".format(spec['groupby'], agg_suffix, new_feature_name))

        df[new_feature_name] = (df.click_time_int - df[group_of_selected_features].groupby(spec['groupby'])
                                .click_time_int.shift(+1)).astype(np.float32)

        gc.collect()

    return df



# ======================================================================================================================
# Calculate the variance for each group
# ======================================================================================================================


def do_var(df, group_cols, counted, agg_type='float32'):
    agg_name = '{}_by_{}_var'.format(('_'.join(group_cols)), counted)

    print(">> Calculating variance of {} by {} ... and saved in {}".format(counted, group_cols, agg_name))

    gp = df[group_cols + [counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp

    gc.collect()

    df[agg_name] = df[agg_name].astype(agg_type)

    gc.collect()

    return df


# ======================================================================================================================
# Calculate the count for each group
# ======================================================================================================================


def do_count(df, group_cols, agg_type='uint32'):
    agg_name = '{}count'.format('_'.join(group_cols))

    print(">> Aggregating by {} ... and saved in {}".format(group_cols, agg_name))

    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp

    gc.collect()

    df[agg_name] = df[agg_name].astype(agg_type)

    gc.collect()

    return df


# ======================================================================================================================
# Calculate the unique count for each group
# ======================================================================================================================


def do_countuniq(df, group_cols, counted, agg_type='uint32'):
    agg_name = '{}_by_{}_countuniq'.format(('_'.join(group_cols)), counted)

    print(">> Counting unqiue ", counted, " by ", group_cols, '... and saved in', agg_name)

    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp

    gc.collect()

    df[agg_name] = df[agg_name].astype(agg_type)

    gc.collect()

    return df

# ======================================================================================================================
# Calculate the cumulative count for each group
# ======================================================================================================================


def do_cumcount(df, group_cols, counted, agg_type='uint32'):
    agg_name = '{}_by_{}_cumcount'.format(('_'.join(group_cols)), counted)

    print(">> Cumulative count by {} ... and saved in {}".format(group_cols, agg_name))

    gp = df[group_cols + [counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp

    gc.collect()

    df[agg_name] = df[agg_name].astype(agg_type)

    gc.collect()

    return df


# ======================================================================================================================
# Calculate the mean for each group
# ======================================================================================================================


def do_mean(df, group_cols, counted, agg_type='float32'):
    agg_name = '{}_by_{}_mean'.format(('_'.join(group_cols)), counted)

    print(">> Calculating mean of {} by {} ... and saved in {}".format(counted, group_cols, agg_name))

    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp

    gc.collect()

    df[agg_name] = df[agg_name].astype(agg_type)

    gc.collect()

    return df

