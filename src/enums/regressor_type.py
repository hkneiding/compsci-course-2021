from enum import Enum, auto


class RegressorType(Enum):

    '''Enum class for regressor types'''

    OLS = auto()
    LASSO = auto()
    RIDGE = auto()

