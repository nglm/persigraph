from typing import Any, Callable

def check_condition(
    condition: Callable[[Any], bool],
    variable: Any,
    var_name: str = 'Variable',
    msg_cond: str = '',
    ):
    if not condition(variable):
        msg_err = var_name + msg_cond + ": " + str(variable)
        raise ValueError(msg_err)

def check_all(
    condition: Callable[[Any], bool],
    iterable: Any,
    var_name: str = 'Variable',
    msg_cond: str = '',
    ):
    for elt in iterable:
        if not condition(elt):
            print(var_name, ': ', elt)
            msg_err = var_name + msg_cond
            raise ValueError(msg_err)

def check_O1_range(variable, var_name: str = 'Variable'):
    condition = lambda x : (x >= 0 and x <= 1)
    msg_cond =  " should be within 0-1 range"
    check_condition(condition, variable, var_name, msg_cond)

def check_positive(variable, var_name: str = 'Variable'):
    condition = lambda x : (x >= 0)
    msg_cond =  " should be >= 0"
    check_condition(condition, variable, var_name, msg_cond)

def check_all_positive(iterable, var_name: str = 'Variable'):
    condition = lambda x : (x >= 0)
    msg_cond =  " should be >= 0"
    check_all(condition, iterable, var_name, msg_cond)

def check_int(variable, var_name: str = 'Variable'):
    condition = lambda x : (x == int(x))
    msg_cond =  " should be an integer"
    check_condition(condition, variable, var_name, msg_cond)

def check_all_int(iterable, var_name: str = 'Variable'):
    condition = lambda x : (x == int(x))
    msg_cond =  " should be an integer"
    check_all(condition, iterable, var_name, msg_cond)

def check_int_positive(variable, var_name: str = 'Variable'):
    check_int(variable, var_name)
    check_positive(variable, var_name)

def check_all_int_positive(iterable, var_name: str = 'Variable'):
    check_all_int(iterable, var_name)
    check_all_positive(iterable, var_name)