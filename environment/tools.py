from sklearn.preprocessing import OneHotEncoder

one_hot_enc = OneHotEncoder()
one_hot_enc.fit([[0],[1],[2]])

def to_categorical_(target_on_map):
    return one_hot_enc.transform(target_on_map)
