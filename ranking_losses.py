from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

# Define Hinge Loss for Ranking
def rank_hinge_loss(num_negative=6):
    """
        Loss function for ranking

        :params:
            - num_negative: Total Number of Sampled Negative Queries per example
    """
    margin = 1.
    def _margin_loss(y_true, y_pred):
        y_pos = Lambda(lambda a: a[::(num_negative+1), :], output_shape= (1,))(y_pred)
        losses = []
        for i in range(num_negative):
          y_neg = Lambda(lambda a: a[(i+1)::(num_negative+1), :], output_shape= (1,))(y_pred)
          loss = K.maximum(0., margin + y_neg - y_pos)
          losses.append(loss)
        temp = K.concatenate(losses,-1)
        temp = K.mean(temp)
        return temp
    return _margin_loss
