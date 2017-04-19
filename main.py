import numpy as np
from sklearn import metrics

def calculate_similarity(array, similarity_type, used_users):

    #result_matrix = [[0 for j in range(len(array))] for i in range(len(array))]
    if used_users is None:
        size = len(array)
        result_matrix = np.ones((size,size)) * 0
        for i in range(size):
            column_i = np.nan_to_num((array[i] - np.nanmean(array[i])))
            for j in range(i,len(array)):
                column_j = np.nan_to_num((array[j] - np.nanmean(array[j])))
                # calculate the similarity by similarity_type.
                # f = np.dot(column_i, column_j) / (np.sqrt(np.dot(column_j, column_j)) * np.sqrt(np.dot(column_i, column_i)))
                value_of_sim = similarity_type(X=column_i, Y=column_j)
                result_matrix[i][j] = value_of_sim[0][0]
                result_matrix[j][i] = value_of_sim[0][0]
        return None,np.array(result_matrix)
    else:
        size = len(used_users)
        user_dictionary = {}
        result_matrix = np.ones((size,size)) * 0
        for i in range(size):
            user_dictionary[used_users[i]] = i
            column_i = np.nan_to_num((array[i] - np.nanmean(array[i])))
            for j in range(i,size):
                column_j = np.nan_to_num((array[j] - np.nanmean(array[j])))
                # calculate the similarity by similarity_type.
                # f = np.dot(column_i, column_j) / (np.sqrt(np.dot(column_j, column_j)) * np.sqrt(np.dot(column_i, column_i)))
                value_of_sim = similarity_type(X=column_i, Y=column_j)
                result_matrix[i][j] = value_of_sim[0][0]
                result_matrix[j][i] = value_of_sim[0][0]
        return user_dictionary, np.array(result_matrix)



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


def item_item_prediction(sim_matrix, array_data, i_position, j_position):
    condition = [np.logical_not(np.isnan(array_data[:, j_position]))]
    choice = [array_data[:, j_position]]
    sd = np.select(condlist=condition, choicelist=choice)
    sd = np.array([sd[i] - mean_items[i] if sd[i] != 0 else 0 for i in range(len(sd))])

    condition_sim = [sim_matrix[i_position] > 0, sim_matrix[i_position] == 1]
    choice_sim = [sim_matrix[i_position], 0]
    sd_sim = np.select(condition_sim, choice_sim)
    numenator = np.dot(sd, sd_sim)
    denominator = np.sum(sd_sim) - 1
    result = mean_items[i_position] + numenator / denominator
    return result


def user_user_prediction(sim_matrix, array_data, user_position, item_position):
    condition = [np.logical_not(np.isnan(array_data[:, user_position]))]
    choice = [array_data[:, user_position]]
    sd = np.select(condlist=condition, choicelist=choice)
    sd = np.array([sd[i] - mean_users[i] if sd[i] != 0 else 0 for i in range(len(sd))])

    condition_sim = [sim_matrix[item_position] > 0, sim_matrix[item_position] == 1]
    choice_sim = [sim_matrix[item_position], 0]
    sd_sim = np.select(condition_sim, choice_sim)
    numenator = np.dot(sd, sd_sim)
    denominator = np.sum(sd_sim) - 1
    result = mean_users[item_position] + numenator / denominator
    return result


def loading_preparing_data():
    data = np.genfromtxt('course-certificate.csv', delimiter=',')
    data = data[1:].astype(np.int32)
    print(np.unique(data[:,0]).size)
    print(np.unique(data[:, 1]).size)
    course_dict = {}
    count = 0
    for course in np.unique(data[:, 1]):
        course_dict[course] = count
        count += 1
    train_data = data[:(len(data) * 0.9).__int__()]
    test_data = data[(len(data) * 0.9).__int__():]
    n = np.unique(data[:, 1]).size
    m = np.unique(data[:, 0]).size
    array_number = np.ones((n, m)) * np.nan
    for row in train_data:
        array_number[course_dict[row[1]]][row[0] - 1] = row[2]
    return course_dict, np.array(array_number), np.array(test_data)


def calculate_rmse(training_dataset, test_dataset):
    rmse_nominator_item_item = 0
    rmse_nominator_user_user = 0
    for i in test_dataset:
        # calculation item-based CF and user-based CF.
        # calculation is going for selected item and user.
        v1 = item_item_prediction(similarity_matrix_item_item, training_dataset, i[1].__int__()-1, i[0].__int__()-1)
        v2 = user_user_prediction(similarity_matrix_user_user, training_dataset.transpose(), i[1].__int__()-1, i[0].__int__()-1)
        rmse_nominator_item_item += (v1 - i[2]) ** 2
        rmse_nominator_user_user += (v2 - i[2]) ** 2
    rmse_nominator_item_item = np.sqrt(rmse_nominator_item_item / len(test_dataset))
    rmse_nominator_user_user = np.sqrt(rmse_nominator_user_user / len(test_dataset))
    return rmse_nominator_item_item, rmse_nominator_user_user


def calculate_mean_to_item(data):
    mean_value = {np.nanmean(i) for i in data}
    return mean_value


def calculate_mean_to_users(data, test_data):
    mean_value = {}
    tr_data = data.transpose()
    for row in test_data:
        mean_value[row -1 ] = np.nanmean(data[:,(row-1)])
    #mean_value = [np.nanmean(i) for i in data.transpose()]
    return mean_value


course_dictionary, train_data, test_data = loading_preparing_data()

# calculation mean value for each user and for each item.
mean_items = calculate_mean_to_item(train_data)
mean_users = calculate_mean_to_users(train_data, np.unique(test_data[:,0]))

# calculation similarity matricies for items and for users
similarity_matrix_item_item = calculate_similarity(train_data, metrics.pairwise.cosine_similarity,None)

similarity_matrix_user_user = calculate_similarity(train_data.transpose(), metrics.pairwise.cosine_similarity,np.unique(test_data[:,0]))

rmse_u, rmse_i = calculate_rmse(train_data, test_data)
print(str(rmse_i)+" item-item")
print(str(rmse_u)+" user-user")
# prediction(similarity_matrix_item_item, train_data, 0, 4)
