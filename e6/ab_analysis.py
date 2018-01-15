import sys
import pandas as pd
import numpy as np
from scipy import stats

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value: {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value: {more_searches_p:.3g}\n'
    '"Did more/less instructors use the search feature?" p-value: {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value: {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]

    data = pd.read_json(searchdata_file, orient='records', lines=True)

    # more_searches_p
    utest_p = stats.mannwhitneyu(data[(data["uid"]%2==0)]["search_count"], data[(data["uid"]%2==1)]["search_count"]).pvalue

    odd_usr_searched = data[ (data["uid"]%2==1) & (data["search_count"]>0) ]["uid"].count();
    odd_usr_never_searched = data[ (data["uid"]%2==1) & (data["search_count"]==0) ]["uid"].count();
    even_usr_searched = data[ (data["uid"]%2==0) & (data["search_count"]>0) ]["uid"].count();
    even_usr_never_searched = data[ (data["uid"]%2==0) & (data["search_count"]==0) ]["uid"].count();

    contingency = [[odd_usr_searched, odd_usr_never_searched], [even_usr_searched, even_usr_never_searched]]
    
    # more_users_p
    _,chitest_p,_,_ = stats.chi2_contingency(contingency)


    instru_data = data[(data['is_instructor'] == True)]

    # more_instr_searches_p
    instru_utest_p = stats.mannwhitneyu(instru_data[(instru_data["uid"]%2==0)]["search_count"], instru_data[(instru_data["uid"]%2==1)]["search_count"]).pvalue
    
    instru_odd_usr_searched = instru_data[ (instru_data["uid"]%2==1) & (instru_data["search_count"]>0) ]["uid"].count();
    instru_odd_usr_never_searched = instru_data[ (instru_data["uid"]%2==1) & (instru_data["search_count"]==0) ]["uid"].count();
    instru_even_usr_searched = instru_data[ (instru_data["uid"]%2==0) & (instru_data["search_count"]>0) ]["uid"].count();
    instru_even_usr_never_searched = instru_data[ (instru_data["uid"]%2==0) & (instru_data["search_count"]==0) ]["uid"].count();
    instru_contingency = [[instru_odd_usr_searched, instru_odd_usr_never_searched], [instru_even_usr_searched, instru_even_usr_never_searched]]

    # more_instr_p
    _,instru_chitest_p,_,_ = stats.chi2_contingency(instru_contingency)

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=chitest_p,
        more_searches_p=utest_p,
        more_instr_p=instru_chitest_p,
        more_instr_searches_p=instru_utest_p,
    ))


if __name__ == '__main__':
    main()