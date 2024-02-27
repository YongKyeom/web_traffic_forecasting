import datetime as dt
import logging


def get_today_str(date):
    import datetime as dt
    
    return date.strftime("%Y%m%d")


def make_logg(path, name, level, date, valid_yn):

    import logging

    logpath = path + get_today_str(date) + "_" + name + '.log'

    logger = logging.getLogger(name)

    # Check handler exists
    if len(logger.handlers) > 0:
        return logger  # Logger already exists

    logger.setLevel(level)

    ## 개발
    formatter = logging.Formatter("%(asctime)s %(filename)s %(funcName)s Line %(lineno)d [%(levelname)s] %(message)s")
    # formatter = logging.Formatter("%(asctime)s %(funcName)s Line %(lineno)d [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    file_handler = logging.FileHandler(filename = logpath)

    if valid_yn is True:
        console.setLevel(logging.DEBUG)
    else:
        console.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


class Logger:
    
    def __init__(
            self,
            path = './logs/',
            name = 'Preprocessing',
            level = logging.DEBUG,
            date = None,
            valid_yn = False,
    ):
        self.__logger = make_logg(path, name, level, date, valid_yn)
        self.__path   = path
        self.__name   = name
        self.__level  = level
        self.__date   = date

    @property
    def path(self):
        return self.__path

    @property
    def name(self):
        return self.__name

    @property
    def level(self):
        return self.__level

    @property
    def logger(self):
        return self.__logger
    
    @property
    def date(self):
        return self.__date