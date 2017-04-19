import numpy as np
from sklearn import metrics


def calculate_similarity(array, similarity_type, used_users,users_d):
    if used_users is None:
        size = len(array)
        result_matrix = np.ones((size, size)) * 0
        for i in range(size):
            column_i = np.nan_to_num((array[i] - np.nanmean(array[i])))
            for j in range(i, len(array)):
                column_j = np.nan_to_num((array[j] - np.nanmean(array[j])))
                # calculate the similarity by similarity_type.
                value_of_sim = similarity_type(X=column_i, Y=column_j)
                result_matrix[i][j] = value_of_sim[0][0]
                result_matrix[j][i] = value_of_sim[0][0]
        return np.array(result_matrix)
    else:
        size = len(array[:,0])
        user_dictionary_for_similarity = {}
        result_matrix = np.ones((len(used_users), size)) * 0
        for i in range(len(used_users)):
            user_dictionary_for_similarity[used_users[i]] = i
            array_list = array[user_dict[used_users[i]]]
            column_i = np.nan_to_num( array_list - np.nanmean(array_list))#
            for j in range(size):
                array_list_j = array[j]
                column_j = np.nan_to_num((array_list_j - np.nanmean(array_list_j)))#array[user_dict[used_users[j]]]
                # calculate the similarity by similarity_type.
                # f = np.dot(column_i, column_j) / (np.sqrt(np.dot(column_j, column_j)) * np.sqrt(np.dot(column_i, column_i)))
                value_of_sim = similarity_type(X=column_i, Y=column_j)
                result_matrix[i][j] = value_of_sim[0][0]
        return user_dictionary_for_similarity,np.array(result_matrix)


def prediction(sim_matrix, array_data, i_position, j_position):
    condition = [np.logical_not(np.isnan(array_data[:, j_position]))]
    choice = [array_data[:, j_position]]
    sd = np.select(condlist=condition, choicelist=choice)
    condition_sim = [sim_matrix[i_position] > 0, sim_matrix[i_position] == 1]
    choice_sim = [sim_matrix[i_position], 0]
    sd_sim = np.select(condition_sim, choice_sim)
    numenator = np.dot(sd, sd_sim)
    denominator = np.sum(sd_sim) - 1
    return numenator / denominator


def item_item_prediction(sim_matrix, array_data, user_position, item_position, user_d, course_d):
    selected_array_by_user = array_data[:, user_d[user_position]]
    condition = [np.logical_not(np.isnan(selected_array_by_user))]
    choice = [selected_array_by_user]

    rating = np.select(condlist=condition, choicelist=choice)
    rating = np.array([(rating[i] - mean_items[course_d[i+1]]) if rating[i] != 0 else 0 for i in range(len(rating))])

    condition_sim = [sim_matrix[course_d[item_position]] > 0, sim_matrix[course_d[item_position]] == 1]
    choice_sim = [sim_matrix[course_d[item_position]], 0]
    sd_sim = np.select(condition_sim, choice_sim)

    numenator = np.dot(rating, sd_sim)
    denominator = np.sum(sd_sim) - 1
    result = mean_items[course_d[item_position]] + numenator / denominator
    return result


def user_user_prediction(sim_matrix, array_data, user_position, item_position, user_dictionary,
                         item_dictionary,similarity_dict):
    users_selected = array_data[:, item_dictionary[item_position]]
    condition = [np.logical_not(np.isnan(users_selected))]
    choice = [users_selected]
    sd = np.select(condlist=condition, choicelist=choice)
    asdf = []
    for i in range(len(sd)):
        if sd[i] != 0:
            asdf.append(sd[i] - mean_users[user_dictionary[i+1]])
        else:
            asdf.append(0)
    sd = np.array([sd[i] - mean_users[user_dictionary[i+1]] if sd[i] != 0 else 0 for i in range(len(sd))])

    value_sim_selected = similarity_dict[user_position]
    condition_sim = [sim_matrix[value_sim_selected] > 0,
                     sim_matrix[value_sim_selected] == 1]
    choice_sim = [sim_matrix[value_sim_selected], 0]
    sd_sim = np.select(condition_sim, choice_sim)
    numenator = np.dot(sd, sd_sim)
    denominator = np.sum(sd_sim) - 1
    if np.count_nonzero(sd_sim) == 0:
        denominator += 1
    result = mean_users[item_position] + numenator / denominator
    return result


def loading_preparing_data():
    data = np.genfromtxt('simple.csv', delimiter=',')
    data = data[1:].astype(np.int32)
    print(np.unique(data[:, 0]).size)
    print(np.unique(data[:, 1]).size)
    course_dict = {}
    user_dict = {}
    count = 0
    for course in np.unique(data[:, 1]):
        course_dict[course] = count
        count += 1
    count = 0
    for user in np.unique(data[:, 0]):
        user_dict[user] = count
        count += 1
    train_data = data[:(len(data) * 0.9).__int__()]
    test_data = data[(len(data) * 0.9).__int__():]
    n = np.unique(data[:, 1]).size
    m = np.unique(data[:, 0]).size
    array_number = np.ones((n, m)) * np.nan
    for row in train_data:
        array_number[course_dict[row[1]]][user_dict[row[0]]] = row[2]
    return user_dict, course_dict, np.array(array_number), np.array(test_data)


def calculate_rmse(training_dataset, test_dataset, user_d, course_d, similarity_dictionary):
    user_based_result = []
    item_based_result = []
    for i in test_dataset:
        # calculation item-based CF and user-based CF.
        # calculation is going for selected item and user.
        v1 = item_item_prediction(similarity_matrix_item_item, training_dataset, i[0].__int__() , i[1].__int__(),
                                  user_d, course_d)
        v2 = user_user_prediction(similarity_matrix_user_user, training_dataset.transpose(), i[0].__int__() ,
                                  i[1].__int__() ,   user_d, course_dictionary,similarity_dictionary)
        user_based_result.append(v2)
        item_based_result.append(v1)
    rmse_users = metrics.mean_squared_error(np.nan_to_num(user_based_result), test_dataset[:,2])
    rmse_items = metrics.mean_squared_error(np.nan_to_num(item_based_result), test_dataset[:, 2])
    return rmse_users, rmse_items


def calculate_mean_to_item(data,user_d, coure_d):
    mean_value = [np.nanmean(i) for i in data]
    return mean_value


def calculate_mean_to_users(data, test_data, user_d, course_d):
    tr_data = data.transpose()
    mean_value = [np.nanmean(i) for i in tr_data]
    return mean_value


user_dict,course_dictionary, train_data, test_data = loading_preparing_data()

# calculation mean value for each user and for each item.
mean_items = calculate_mean_to_item(train_data,user_dict,course_dictionary)
mean_users = calculate_mean_to_users(train_data, np.unique(test_data[:, 0]), user_dict, course_dictionary)

# calculation similarity matricies for items and for users
similarity_matrix_item_item = calculate_similarity(train_data, metrics.pairwise.cosine_similarity, None,None)

user_dictionary_for_similarity,similarity_matrix_user_user = calculate_similarity(train_data.transpose(), metrics.pairwise.cosine_similarity,
                                                   np.unique(test_data[:, 0]),user_dict)
rmse_u, rmse_i = calculate_rmse(train_data, test_data, user_dict,course_dictionary,user_dictionary_for_similarity)
print(str(rmse_i) + " item-item")
print(str(rmse_u) + " user-user")
