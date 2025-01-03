## created 9/18/24
## TO- collect statannot functions previoulsy collected in /helper functions/ file, and have under one module
from ax_modifier_functions import *
from helper_functions import * 
import scipy
from scipy import stats
from stat_tests import * 


## functinos
#common error- ValueError: Cannot set a DataFrame with multiple columns to the single column g1_num_loc- this is if you mistype the hue value or you use the wrong version of SNS 
def main_run_posthoc_tests_and_get_hue_loc_df(ax_input, plot_params, plot_obj, preset_comparisons,
                                               hue_var= None, test_name = None, hue_order = None, ax_var_is_hue=False):
    """ 
    Run posthoc tests on all axis ticks, get hue levels for each axis tick, and join this to the dataframe produced.

    Parameters:
    ax_input (matplotlib.axes.Axes): The input axis object.
    plot_params (dict): Dictionary containing plot parameters.
    plot_obj (seaborn.axisgrid.FacetGrid): The plot object.
    preset_comparisons (list): List of preset comparisons.
    hue_var (str, optional): The hue variable. Defaults to None.
    test_name (str, optional): The name of the test. Defaults to None.
    hue_order (list, optional): The order of the hue. Defaults to None.
    ax_var_is_hue (bool, optional): Whether the axis variable is the hue. Defaults to False.

    Returns:
    pandas.DataFrame: DataFrame with posthoc test results and hue locations.
    """
    if hue_var == None:
        hue_var = plot_params['hue']
    if hue_order == None:
        hue_order = plot_params['hue_order']
    if test_name == None:
        test_name = None
        # group_order- depends on if comparing within x axis, or within hues 
    if ax_var_is_hue: #you will use this to find the ordering of the hue collection points of interest
        group_order = plot_params['hue_order'] #order in collection = order in hue
    else: #if hue collection tiled over differnt x categories
        group_order = plot_params['order'] #order in collection = order in x category
    posthoc_df = run_posthoc_tests_on_all_ax_ticks(plot_params['data'], plot_obj = plot_obj, 
                                                   comparison_list =preset_comparisons, ax_grouping_col= plot_params['x'],
                                                   group_order = group_order, hue_col_name=hue_var, value_col_name = plot_params['y'],
                                                   test_name = test_name,ax_var_is_hue=ax_var_is_hue)## get df with info on post-hoc comparisons
    hue_loc_df = get_hue_point_loc_df(ax_input, hue_order) # hue_loc_df = pd.DataFrame.from_dict(get_hue_point_loc_dict(plot_ax, geno_order)).set_index('hue') #get array of numerical points and values for each hue level
    posthoc_df = get_hue_loc_on_axis(hue_loc_df, posthoc_df) #find pos in numerical ax of fig, then add as cols to df
    #manually set cat compared within to single variable if hue == axis category
    if ax_var_is_hue: #you will use this to find the ordering of the hue collection points of interest
        posthoc_df['category_compared_within']= plot_params['x']
    
    return posthoc_df
## ad hue vs ax order 
def run_posthoc_tests_on_all_ax_ticks(plot_data, plot_obj, comparison_list, ax_grouping_col, group_order, hue_col_name, value_col_name,
                                      test_name = None, ax_var_is_hue = False):
    """ 
    Run posthoc tests on all axis ticks.

    Parameters:
    plot_data (pandas.DataFrame): The plot data.
    plot_obj (seaborn.axisgrid.FacetGrid): The plot object.
    comparison_list (list): List of comparisons.
    ax_grouping_col (str): The column name for axis grouping.
    group_order (iterable): The order of the groups.
    hue_col_name (str): The hue column name.
    value_col_name (str): The value column name.
    test_name (str, optional): The name of the test. Defaults to 'MWU'.
    ax_var_is_hue (bool, optional): Whether the axis variable is the hue. Defaults to False.

    Returns:
    pandas.DataFrame: DataFrame with posthoc test results.
    """
    if test_name == None:
        test_name = 'MWU'
   
    compare_stats_df = []
    #if the ax levels = the hue levels, don't filter the plot data by what ax group col you're on
    if ax_var_is_hue:
        
        print(f"With axis variable == Hue variable:")
        for geno_pair in comparison_list: #iterate over ex. (WT VEH to HET VEH), do stats on each
            posthoc_output= run_posthoc_test_on_tick_hue_groups(plot_data,
                                                                    geno_pair[0], geno_pair[1], geno_pair,group_order,
                                                                    hue_col_name, value_col_name,test_name = test_name,ax_var_is_hue = ax_var_is_hue)
            compare_stats_df.append(posthoc_output)
    else:
    #iterate through the different categories to compare hue level values within
        for ax_category_level in plot_data[ax_grouping_col].unique():        # print(ax_category_level)
            for geno_pair in comparison_list: #iterate over ex. (WT VEH to HET VEH), do stats on each
                posthoc_output= run_posthoc_test_on_tick_hue_groups(plot_data.loc[plot_data[ax_grouping_col] ==  ax_category_level,:],
                                                                    geno_pair[0], geno_pair[1], ax_category_level, group_order,
                                                                    hue_col_name, value_col_name,test_name = test_name)
                compare_stats_df.append(posthoc_output)
    stat_table = pd.DataFrame.from_records(compare_stats_df)
    if not(ax_var_is_hue):#transform list of xticklabels to pandas df and merge ## inserted 10.16.24- automerge the xticks
        stat_table = stat_table.merge(get_x_ticks_as_df(plot_obj.get_xticklabels()), left_on = 'category_compared_within', right_on = 'tick_text') 
    return stat_table

##active_unit_df
def run_posthoc_test_on_tick_hue_groups(ax_tick_data, hue_group_1, hue_group_2, ax_category_level,group_order,
                                         hue_col_name, value_cols_name,test_name = None,ax_var_is_hue = False):
    """ 
    Run posthoc test on tick hue groups.

    Parameters:
    ax_tick_data (pandas.DataFrame): The axis tick data.
    hue_group_1 (str): The first hue group.
    hue_group_2 (str): The second hue group.
    ax_category_level (str): The axis category level.
    group_order (iterable): The order of the groups.
    hue_col_name (str): The hue column name.
    value_cols_name (str): The value column name.
    test_name (str, optional): The name of the test. Defaults to 'MWU'.
    ax_var_is_hue (bool, optional): Whether the axis variable is the hue. Defaults to False.

    Returns:
    dict: Dictionary with the posthoc test results.
    """
    ##define test to run
    if test_name == None:
        test_name = 'MWU'
    ## extract groups
    # ax_tick_data =plot_data.loc[plot_data[ax_grouping_col] ==  ax_category_level,:]
    data_group_1= ax_tick_data.loc[ax_tick_data[hue_col_name] == hue_group_1,:]
    data_group_2= ax_tick_data.loc[ax_tick_data[hue_col_name] == hue_group_2,:]
    ## get values from each grup
    data_group_1_values = data_group_1[value_cols_name].values
    data_group_2_values =data_group_2[value_cols_name].values
    result_dict = get_pair_stat_test_result(test_name, ax_category_level,group_order, hue_group_1, hue_group_2, data_group_1_values,data_group_2_values,ax_var_is_hue)
    return result_dict
    
    
    ## custom function to do stats on grups of interest
def get_pair_stat_test_result(test_name, ax_category_level,group_order, group_1_name, group_2_name, data_group_1_values,data_group_2_values,ax_var_is_hue = False):
    """ 
    Run statistical test on data groups.

    Parameters:
    test_name (str): The name of the test.
    ax_category_level (str): The axis category level.
    group_order (iterable): The order of the groups.
    group_1_name (str): The name of the first group.
    group_2_name (str): The name of the second group.
    data_group_1_values (numpy.ndarray): Values of the first group.
    data_group_2_values (numpy.ndarray): Values of the second group.
    ax_var_is_hue (bool, optional): Whether the axis variable is the hue. Defaults to False.

    Returns:
    dict: Dictionary with the statistical test results.
    """
    '''Run stats test on data_group_1_values and data_group_2_values (user input for test )
    ax order- tells you what order elements in the hue category levels are spread across the ax, so you can use later for indexing'''
    ## run stats on group values
    stat_result= []
    if test_name == 'custom':
            print('custom test ran')
    elif test_name == 'MWU':
            stat_result = stats.mannwhitneyu(data_group_1_values, data_group_2_values)
    elif test_name == 'bootstrap_sdev_overlap':
            stat_result = test_group_mean_separation(data_group_1_values, data_group_2_values)
    elif test_name == '2_sample_t_test':
            stat_result = stats.ttest_ind(data_group_1_values, data_group_2_values, equal_var = False) #equal_var = True, run 2 sample ttset, if false, run welch's test for unequal var
    elif test_name == 'permutation_test':
            stat_result =  run_permutation_test_on_diff_of_vector_means( data_group_1_values, data_group_2_values, 10000) #set to .values as original output is dict, and rounding a rdict fails 
    #record the stat values (mean, sem etc)     
    if test_name == 'bootstrap_sdev_overlap':
    
        group_mean_dict ={'group_1_mean':stat_result['group_1_mean'], 'group_1_sem':stat_result['group_1_std'],
                            'group_2_mean':stat_result['group_2_mean'], 'group_2_sem':stat_result['group_2_std']}
        stat_result = [stat_result['mean_diff_more_than_sdevs'], stat_result['pseudo_pvalue']]
    else:
        group_mean_dict = {'group_1_mean':np.nanmean(data_group_1_values), 'group_1_sem':scipy.stats.sem(data_group_1_values,nan_policy = 'omit' ),
                            'group_2_mean':np.nanmean(data_group_2_values), 'group_2_sem':scipy.stats.sem(data_group_2_values,nan_policy = 'omit')}
    #pack and return result dict
    print(group_order)
    if ax_var_is_hue: #if x categorical ticks = hue groups, find index of group1 name in hue order
        group_pos = {'group_1_order_pos': get_match_index_in_iterable(group_order, group_1_name),
                   'group_2_order_pos': get_match_index_in_iterable(group_order, group_2_name)}
    else: #else, find index of ax_category_tick name in xtick order, to pull correct point loc (say of wt-veh, at late IA tick)
        group_pos = {'group_1_order_pos': get_match_index_in_iterable(group_order, ax_category_level),
                   'group_2_order_pos': get_match_index_in_iterable(group_order, ax_category_level)}
        #get where the groups being compared, are listed in the data collections f
    result_dict = {'category_compared_within': ax_category_level, 'group_1': group_1_name, 'group_2':group_2_name,
                   'group_1_n':data_group_1_values.shape, 'group_2_n':data_group_2_values.shape,
                   **group_pos, **group_mean_dict,
                    'test_name': test_name, 'stat_result': np.round(stat_result, 5), 'pvalue': stat_result[1]}
    return result_dict
    ## build custom plotting ufnction

def get_hue_loc_on_axis(hue_loc_df, posthoc_df):
    """ 
    Add numerical and categorical axis locations to the posthoc comparison dataframe.

    Parameters:
    hue_loc_df (pandas.DataFrame): DataFrame with hue locations.
    posthoc_df (pandas.DataFrame): DataFrame with posthoc comparisons.

    Returns:
    pandas.DataFrame: Updated DataFrame with axis locations.
    """
    """ given hue_loc_df listing where each point for the given hue categories are located, use the pre-existing post hoc comparison df
    and add a new column to dataframe indicating y/numerical ax loc for each comparison 
    g1/2_num_loc: the value of position of group 1/2 (usually on the y axis) on the numerical axis used
    g1/2_cat_loc: the value of point of group 1/2 (usually on the x axis) on the categorical axis used
    group_1/2_order_pos is used to to know what the values are """
        ## get loc of points on the numerical ax (usually y but not always)
        #index into hueloc df, with index = hue Group_1 name of posthoc df ; then, get list of point vals in collection; then, 
    posthoc_df['hue_group_1_locs'] = posthoc_df.apply(lambda x: hue_loc_df.loc[x['group_1']], axis = 1) #elem index = what x tick num elem is centered on
    posthoc_df['hue_group_2_locs'] = posthoc_df.apply(lambda x: hue_loc_df.loc[x['group_2']], axis = 1) #elem index = what x tick num elem is centered on
    posthoc_df['g1_num_loc'] = posthoc_df.apply(lambda x: x['hue_group_1_locs'][x['group_1_order_pos'][0],1], axis = 1) #x['group_1_order_pos'][0] = position of group_1 being used, in ordered list of collections 
    posthoc_df['g2_num_loc'] = posthoc_df.apply(lambda x: x['hue_group_2_locs'][x['group_2_order_pos'][0],1], axis = 1) #x['group_1_order_pos'][0] = position of group_1 being used, in ordered list of collections 

    ## get location of poitns on the categorical axis (usually x but not always)
    posthoc_df['g1_cat_loc'] = posthoc_df.apply(lambda x: x['hue_group_1_locs'][x['group_1_order_pos'][0],0], axis = 1) #x['group_1_order_pos'][0] = position of group_1 being used, in ordered list of collections 
    posthoc_df['g2_cat_loc'] = posthoc_df.apply(lambda x: x['hue_group_2_locs'][x['group_2_order_pos'][0],0], axis = 1) #x['group_1_order_pos'][0] = position of group_1 being used, in ordered list of collections
    #get max of numerical ax values
    posthoc_df['max_group_loc_val'] = posthoc_df[['g1_num_loc', 'g2_num_loc']].max(axis = 1)
    return posthoc_df 

# def main_run_posthoc_tests_and_get_hue_loc_df(ax_input, plot_params, plot_obj, preset_comparisons,
#                                                hue_var= None, test_name = None, hue_order = None, ax_var_is_hue=False):
#     """ TO, 1) run posthoc test on all ax ticks, 2) get hue levels for each ax tick, 3) join this to df produced in 1"""
#     if hue_var == None:
#         hue_var = plot_params['hue']
#     if hue_order == None:
#         hue_order = plot_params['hue_order']
#     if test_name == None:
#         test_name = None
#    # group_order- depends on if comparing within x axis, or within hues 
#     if ax_var_is_hue:
#         group_order = plot_params['order']
#     else:
#         group_order = plot_params['hue_order']
#     posthoc_df = run_posthoc_tests_on_all_ax_ticks(plot_params['data'], plot_obj = plot_obj, 
#                                                    comparison_list =preset_comparisons, ax_grouping_col= plot_params['x'],
#                                                    group_order = group_order, hue_col_name=hue_var, value_col_name = plot_params['y'],
#                                                    test_name = test_name,ax_var_is_hue=ax_var_is_hue)## get df with info on post-hoc comparisons
    
#     hue_loc_df = get_hue_point_loc_df(ax_input, hue_order) # hue_loc_df = pd.DataFrame.from_dict(get_hue_point_loc_dict(plot_ax, geno_order)).set_index('hue') #get array of numerical points and values for each hue level
#     posthoc_df = get_hue_loc_on_axis(hue_loc_df, posthoc_df) #find pos in numerical ax of fig, then add as cols to df
#     return posthoc_df


# def run_posthoc_tests_on_all_ax_ticks(plot_data, plot_obj, comparison_list, ax_grouping_col, ax_order, hue_col_name, value_col_name,
#                                       test_name = None, ax_var_is_hue = False):
#     """ ax_grouping_col- str that == name of column where each unique level = 1 tick on the categorical ax to do posthoc tests within
#     plot_obj- output of seaborn plotting function """
#     if test_name == None:
#         test_name = 'MWU'
   
#     compare_stats_df = []
#     #if the ax levels = the hue levels, don't filter the plot data by what ax group col you're on
#     if ax_var_is_hue:
#         print(f"With axis variable == Hue variable:")
#         for geno_pair in comparison_list: #iterate over ex. (WT VEH to HET VEH), do stats on each
#             posthoc_output= run_posthoc_test_on_tick_hue_groups(plot_data,
#                                                                     geno_pair[0], geno_pair[1], geno_pair,ax_order,
#                                                                     hue_col_name, value_col_name,test_name = test_name)
#             compare_stats_df.append(posthoc_output)
#     else:
#     #iterate through the different categories to compare hue level values within
#         for ax_category_level in plot_data[ax_grouping_col].unique():        # print(ax_category_level)
#             for geno_pair in comparison_list: #iterate over ex. (WT VEH to HET VEH), do stats on each
#                 posthoc_output= run_posthoc_test_on_tick_hue_groups(plot_data.loc[plot_data[ax_grouping_col] ==  ax_category_level,:],
#                                                                     geno_pair[0], geno_pair[1], ax_category_level, ax_order,
#                                                                     hue_col_name, value_col_name,test_name = test_name)
#                 compare_stats_df.append(posthoc_output)
#     stat_table = pd.DataFrame.from_records(compare_stats_df)
#     if not(ax_var_is_hue):#transform list of xticklabels to pandas df and merge ## inserted 10.16.24- automerge the xticks
#         stat_table = stat_table.merge(get_x_ticks_as_df(plot_obj.get_xticklabels()), left_on = 'category_compared_within', right_on = 'tick_text') 
#     return stat_table


# def run_posthoc_test_on_tick_hue_groups(ax_tick_data, hue_group_1, hue_group_2, ax_category_level,ax_category_order,
#                                          hue_col_name, value_cols_name,test_name = None):
#     ##define test to run
#     if test_name == None:
#         test_name = 'MWU'
#     ## extract groups
#     # ax_tick_data =plot_data.loc[plot_data[ax_grouping_col] ==  ax_category_level,:]
#     data_group_1= ax_tick_data.loc[ax_tick_data[hue_col_name] == hue_group_1,:]
#     data_group_2= ax_tick_data.loc[ax_tick_data[hue_col_name] == hue_group_2,:]
#     ## get values from each grup
#     data_group_1_values = data_group_1[value_cols_name].values
#     data_group_2_values =data_group_2[value_cols_name].values
#     result_dict = get_pair_stat_test_result(test_name, ax_category_level,ax_category_order, hue_group_1, hue_group_2, data_group_1_values,data_group_2_values)
#     return result_dict
    
    
#     ## custom function to do stats on grups of interest
# def get_pair_stat_test_result(test_name, ax_category_level,ax_order, group_1_name, group_2_name, data_group_1_values,data_group_2_values):
#     '''Run stats test on data_group_1_values and data_group_2_values (user input for test )
#     ax order- tells you what order elements in the ax category levels are, so you can use later for indexing'''
#     ## run stats on group values
#     stat_result= []
#     if test_name == 'custom':
#             print('custom test ran')
#     elif test_name == 'MWU':
#             stat_result = stats.mannwhitneyu(data_group_1_values, data_group_2_values)
#     elif test_name == 'bootstrap_sdev_overlap':
#             stat_result = test_group_mean_separation(data_group_1_values, data_group_2_values)
#     elif test_name == '2_sample_t_test':
#             stat_result = stats.ttest_ind(data_group_1_values, data_group_2_values, equal_var = False) #equal_var = True, run 2 sample ttset, if false, run welch's test for unequal var
#     elif test_name == 'permutation_test':
#             stat_result =  run_permutation_test_on_diff_of_vector_means( data_group_1_values, data_group_2_values, 10000) #set to .values as original output is dict, and rounding a rdict fails 
#     #record the stat values (mean, sem etc)     
#     if test_name == 'bootstrap_sdev_overlap':
    
#         group_mean_dict ={'group_1_mean':stat_result['group_1_mean'], 'group_1_sem':stat_result['group_1_std'],
#                             'group_2_mean':stat_result['group_2_mean'], 'group_2_sem':stat_result['group_2_std']}
#         stat_result = [stat_result['mean_diff_more_than_sdevs'], stat_result['pseudo_pvalue']]
#     else:
#         group_mean_dict = {'group_1_mean':np.nanmean(data_group_1_values), 'group_1_sem':scipy.stats.sem(data_group_1_values,nan_policy = 'omit' ),
#                             'group_2_mean':np.nanmean(data_group_2_values), 'group_2_sem':scipy.stats.sem(data_group_2_values,nan_policy = 'omit')}
#     #pack and return result dict         
#     result_dict = {'category_compared_within': ax_category_level, 'group_1': group_1_name, 'group_2':group_2_name,
#                    'group_1_n':data_group_1_values.shape, 'group_2_n':data_group_2_values.shape,
#                    'group_1_order_pos': get_match_index_in_iterable(ax_order, group_1_name),
#                    'group_2_order_pos': get_match_index_in_iterable(ax_order, group_2_name),
#                     **group_mean_dict,
#                     'test_name': test_name, 'stat_result': np.round(stat_result, 5), 'pvalue': stat_result[1]}
#     return result_dict
    ## build custom plotting ufnction
     
#for combining hue loc df with posthoc ocmparisno df
# def get_hue_loc_on_axis(hue_loc_df, posthoc_df):
#     #OLD-12/30/24, NOT USABLE FOR hue = ax category type data 
#     """ given hue_loc_df listing where each point for the given hue categories are located, use the pre-existing post hoc comparison df
#     and add a new column to dataframe indicating y/numerical ax loc for each comparison 
#     g1/2_num_loc: the value of position of group 1/2 (usually on the y axis) on the numerical axis used
#     g1/2_cat_loc: the value of point of group 1/2 (usually on the x axis) on the categorical axis used
#     tick_pos is used to to know what the values are """

#     x_pos = 0
#     y_pos =1 
#     ## get loc of points on the numerical ax (usually y but not always)
#     #index into hueloc df, with index = hue Group_1 name of posthoc df ; then, get list of point vals in collection; then, 
#     posthoc_df['g1_num_loc'] =posthoc_df.apply(lambda x: hue_loc_df.loc[x['group_1']].iloc[0][x['tick_pos'][0],y_pos], axis = 1)
#     posthoc_df['g2_num_loc'] =posthoc_df.apply(lambda x: hue_loc_df.loc[x['group_2']].iloc[0][x['tick_pos'][0],y_pos], axis = 1)

#     ## get location of poitns on the categorical axis (usually x but not always)
#     posthoc_df['g1_cat_loc'] =posthoc_df.apply(lambda x: hue_loc_df.loc[x['group_1']].iloc[0][x['tick_pos'][0],x_pos], axis = 1)
#     posthoc_df['g2_cat_loc'] =posthoc_df.apply(lambda x: hue_loc_df.loc[x['group_2']].iloc[0][x['tick_pos'][0],x_pos], axis = 1)
#     #get max of numerical ax values
#     posthoc_df['max_group_loc_val'] = posthoc_df[['g1_num_loc', 'g2_num_loc']].max(axis = 1)
#     return posthoc_df 
#verify what order to use for this to make sure alignment of geno_order and collecions/hue levels

##HELPER FUNCTIONS
##get information about hue locs 
def get_hue_point_loc_df(ax_input, hue_order):
    """ 
    Get a DataFrame of the datapoints at each level of the hue variable.

    Parameters:
    ax_input (matplotlib.axes.Axes): The input axis object.
    hue_order (list): The order of the hue.

    Returns:
    pandas.DataFrame: DataFrame with hue point locations.
    """
    hue_loc_df = pd.DataFrame.from_dict(get_hue_point_loc_dict(ax_input, hue_order)).set_index('hue') #get array of numerical points and values for each hue level
    return hue_loc_df

def get_hue_point_loc_dict(ax_input, hue_order):
    """ 
    Get a dictionary of the datapoints at each level of the hue variable.

    Parameters:
    ax_input (matplotlib.axes.Axes): The input axis object.
    hue_order (list): The order of the hue.

    Returns:
    dict: Dictionary with hue point locations.
    """
    hue_point_loc_dict = [{'hue': hue_order[count], 'data_locs':x.get_offsets().data} for count, x in enumerate(ax_input.collections)]
    return hue_point_loc_dict

def get_x_ticks_as_df(ticklabel_obj):
    """ 
    Get a DataFrame of the x-tick labels and their positions.

    Parameters:
    ticklabel_obj (list): List of tick label objects.

    Returns:
    pandas.DataFrame: DataFrame with x-tick labels and positions.
    """
    ticks_df = pd.DataFrame.from_records([{'tick_text':x.get_text(), 'tick_pos': x.get_position()} for x in ticklabel_obj])
    return ticks_df
 

def get_sig_bar_x_vals(comparison_tuple):
    """ 
    Get the x-values for the significance bar.

    Parameters:
    comparison_tuple (namedtuple): Tuple with comparison information.

    Returns:
    list: List of x-values for the significance bar.
    """
    x_vals = [comparison_tuple.g1_cat_loc, comparison_tuple.g1_cat_loc,
              comparison_tuple.g2_cat_loc, comparison_tuple.g2_cat_loc]# list the 4 x coord for points that define the line
    return x_vals

def get_sig_bar_y_vals(bottom_val = None, line_height= 1.01):
    """ 
    Get the y-values for the significance bar.

    Parameters:
    bottom_val (float, optional): The bottom value for the bar. Defaults to 0.95.
    line_height (float, optional): The height of the line. Defaults to 1.01.

    Returns:
    list: List of y-values for the significance bar.
    """
    """ comparison tuple max y value is multipled by offset factor"""
    if bottom_val == None:
        bottom_val = 0.95 #for ax relative point plotting
    # bottom_val = comparison_tuple.max_group_loc_val * offset_factor #for data point plotting
    y_vals = [bottom_val,bottom_val* line_height, bottom_val*line_height,bottom_val]# list the 4 x coord for points that define the line
    return y_vals

#NEW sig bar plotting function
## bottom up, but starting from close above points
def plot_sig_bars_w_comp_df_tight(ax_input, sig_comp_df, direction_to_plot = None, tight = None, tight_offset = None, offset_constant=None, debug = None):
    """ 
    Plot significance bars with comparison dataframe, using a tight layout.
    TO- given parameters, plot vertical lines between centers of datapoints of interest (pre-sorted), with significance star (pre-calculated)
    Parameters
    ax_input (matplotlib.axes.Axes): The input axis object.
    sig_comp_df (pandas.DataFrame): DataFrame with significance comparisons.
    direction_to_plot (str, optional): Direction to plot ('top_down', 'bottom_up'). Defaults to 'bottom_up'.
    tight (bool, optional): Whether to plot bars right above their corresponding values. Defaults to True.
    tight_offset (float, optional): Offset for tight layout. Defaults to 0.075.
    offset_constant (float, optional): Constant for offset. Defaults to 0.0225.
    debug (bool, optional): Whether to print debug information. Defaults to None.
    """
    ## plotting params    #set direction to plot ('top_down', 'bottom_up') #set whether bars are plotted right above their coresponding values, or not
    #declare initial transforms of interest
    transform_ax_to_data = ax_input.transAxes + ax_input.transData.inverted() #create ax-display + display-data pipe
    transform_data_to_ax = transform_ax_to_data.inverted() # 
    #default vcalues
    if direction_to_plot == None:
        direction_to_plot = 'bottom_up'
        line_start_y_pos = 0.8 #base case- plot upwards from 0.8 of ax size 
    if tight == None:
        tight = True #set whether or not to plot bars RIGHT above datapoints
    if tight_offset == None:
        tight_offset = 0.075 #fraction of ax to put between the point of interest and the line of sig post-hoc
    #params for offsetting
    line_height = 1.00 #base case- 1.01
    if offset_constant==None:
        offset_constant = 0.0225 #what linear amount to add, in AX FRACTION AMOUNT 
    
    star_space_to_line = offset_constant*0.1
    if debug == True:
        print(f'tight format, max_numeric_ax_value = {max_numeric_ax_value}.  start y val  = {line_start_y_pos}')
        #transData transforms: (DATA) -> (DISPLAY COORDINATES)     # transAxes transforms (AXES) -> (DISPLAY)     #all transforms -> display coords 
    trans = matplotlib.transforms.blended_transform_factory(x_transform = ax_input.transData,
                                                            y_transform = ax_input.transAxes)# the x coords of this transformation are data, and the y coord are axes
    ## main loop over categorical ticks, bottom up approach 
    for cat in sig_comp_df.category_compared_within.unique():#iterate over each categorical tick value
        top_bbox = np.array([[0, 0],[0, 0]])#initialize box location for comparison # =[lower_x, lower_y] [upper_x, upper_y]
        #get max y position value for each category you're doing post-hoc comparisons within
        sig_comp_category = sig_comp_df.loc[sig_comp_df.category_compared_within == cat,:]
        if tight:
            max_numeric_ax_value = sig_comp_category.loc[:, ['g1_num_loc','g2_num_loc']].max().values.max()    #get max val in the group of interest you're running posthocs on (x ticks of interest)    
            line_start_y_pos = transform_data_to_ax.transform((0,max_numeric_ax_value))[1]+tight_offset # data -> axes 
            if debug == True:
                print(f'tight format, max_numeric_ax_value = {max_numeric_ax_value}.  start y val  = {line_start_y_pos}')
            #transData transforms: (DATA) -> (DISPLAY COORDINATES)     # transAxes transforms (AXES) -> (DISPLAY)     
        for comp in sig_comp_category.itertuples():
            x_vals = get_sig_bar_x_vals(comp) # [comp.g1_cat_loc, comp.g1_cat_loc, comp.g2_cat_loc, comp.g2_cat_loc]# list the 4 x coord for points that define the line
            y_vals =get_sig_bar_y_vals(line_start_y_pos,line_height) #  [comp.max_group_loc_val, comp.max_group_loc_val * h, comp.max_group_loc_val * h, comp.max_group_loc_val] # list 4 y coord for points that define the line
            #compare overlap of proposed y values, in data space 
            line_overlap = (top_bbox[1,1] >= y_vals[0])##check y overlap with previous bounding box,       #top right point y val in top_box defined by [1,1]
            if debug == True:
                print(f"line overlap = ({top_bbox[0,1]} >= {y_vals[0]}")
                print(f"line x_vals, y_vals: {x_vals, y_vals}")
            if line_overlap: #if the top of the prev bbox overlaps with the current line, move the current line up to ABOVE top bbox
                y_vals = get_sig_bar_y_vals(top_bbox[1,1]+offset_constant,line_height)             ## if overlaps with previous bounding box, adjust height by N

            text_x = (x_vals[0]+ x_vals[2])*.5
            text_y = y_vals[1] + star_space_to_line#what linear amount to separate star from line, in AX FRACTION AMOUNT 
            #plot sig star over line
            ax_input.plot(x_vals, y_vals, lw=annotator_default['line_width'], color = 'black', transform = trans, clip_on = False)
            star_annot = ax_input.annotate(convert_pvalue_to_asterisks(comp.pvalue), xy = (text_x, text_y), xycoords = ('data', 'axes fraction'),
                            ha='center', va='baseline', fontsize = 'x-small',fontweight = 'light')# bbox = {'boxstyle': 'Square, pad = 0.0', 'fc': 'lightblue', 'lw': 0})
            bbox_in_ax = ax_input.transAxes.inverted().transform(star_annot.get_window_extent()) #Get the artist's bounding box in display space.
            # ax.transData.inverted() is a matplotlib.transforms.Transform that goes from display coordinates to data coordinates
            top_bbox = bbox_in_ax      #detect overlap by storing, then comparing ot previous versions


## main stat annotation function- no alterations possible NOW DEFUNCT
def plot_sig_bars_w_comp_df(ax_input, sig_comp_df, direction_to_plot = None):
    """ 
    Plot significance bars with comparison dataframe.

    Parameters:
    ax_input (matplotlib.axes.Axes): The input axis object.
    sig_comp_df (pandas.DataFrame): DataFrame with significance comparisons.
    direction_to_plot (str, optional): Direction to plot ('top_down', 'bottom_up'). Defaults to 'bottom_up'.
    """
    """ TO- given parameters, plot vertical lines between centers of datapoints of interest (pre-sorted), with significance star (pre-calculated)"""
    ## plotting params
    if direction_to_plot == None:#set direction to plot ('top_down', 'bottom_up')
        direction_to_plot = 'bottom_up'

    line_height = 1.01
    offset_constant = 0.025 #what linear amount to add
    star_space_to_line = offset_constant/5
    trans = matplotlib.transforms.blended_transform_factory(x_transform = ax_input.transData,y_transform = ax_input.transAxes)# the x coords of this transformation are data, and the y coord are axes
    ## main loop over categorical ticks
    for cat in sig_comp_df.category_compared_within:#iterate over each categorical tick value
        top_bbox = np.array([[0, 0],[0, 0]])#initialize box location for comparison # =[lower_x, lower_y] [upper_x, upper_y]
        for comp in sig_comp_df.loc[sig_comp_df.category_compared_within == cat,:].itertuples():
            x_vals = get_sig_bar_x_vals(comp) # [comp.g1_cat_loc, comp.g1_cat_loc, comp.g2_cat_loc, comp.g2_cat_loc]# list the 4 x coord for points that define the line
            y_vals =get_sig_bar_y_vals(0.95,line_height) #  [comp.max_group_loc_val, comp.max_group_loc_val * h, comp.max_group_loc_val * h, comp.max_group_loc_val] # list 4 y coord for points that define the line
            line_overlap = (top_bbox[0,1] >= y_vals[0])##check overlap with previous bounding box
            if line_overlap: #if the top of the prev bbox overlaps with the current line, move the current line up to ABOVE top bbox
                y_vals = get_sig_bar_y_vals(top_bbox[1,1]+offset_constant,line_height)             ## if overlaps with previous bounding box, adjust height by N
            text_x = (x_vals[0]+ x_vals[2])*.5
            text_y = y_vals[1] + star_space_to_line
            #plot sig star over line
            ax_input.plot(x_vals, y_vals, lw=annotator_default['line_width'], color = 'black', transform = trans, clip_on = False)
            star_annot = ax_input.annotate(convert_pvalue_to_asterisks(comp.pvalue), xy = (text_x, text_y), xycoords = ('data', 'axes fraction'),
                            ha='center', va='baseline', fontsize = 'small',)# bbox = {'boxstyle': 'Square, pad = 0.0', 'fc': 'lightblue', 'lw': 0})
            bbox_in_ax = ax_input.transAxes.inverted().transform(star_annot.get_window_extent()) # to get ax coordinates of bounding box (transform from  Return the Bbox bounding the text, in display units.)
            top_bbox = bbox_in_ax      #detect overlap by storing, then comparing ot previous versions

