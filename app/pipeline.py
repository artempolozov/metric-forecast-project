from app.config.common_data import M_1, M_2, M_3, M_4, M_5
from app.dao.data_acquisition_layer import get_all_data
from app.dao.saver import save_results
from app.processing.main_processing import get_all_data_for_forecast, print_result


dfs = get_all_data()

# result = get_all_data_for_forecast(dfs, M_1, True)
# print_result(result)
# save_results(result, M_1)
#
# result = get_all_data_for_forecast(dfs, M_2, False)
# save_results(result, M_2)
#
# result = get_all_data_for_forecast(dfs, M_3, False)
# save_results(result, M_3)
#
result = get_all_data_for_forecast(dfs, M_4, True, extended=True)
save_results(result, M_4)

# result = get_all_data_for_forecast(dfs, M_5, True, extended=True)
# save_results(result, M_5)
