from enum import Enum, auto


class RegressorType(Enum):

    '''Enum class for regressor types'''

    OLS = auto()
    OLS_SGD = auto()

    LASSO = auto()
    
    RIDGE = auto()
    RIDGE_SGD = auto()

    LOGISTIC = auto()