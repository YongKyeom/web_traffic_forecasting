import os
import numpy as np
import math
import pandas as pd

from datetime import datetime
from datetime import timedelta


def check_usage_of_cpu_and_memory():
    import os
    import psutil

    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory()[2] # used memory percent

    return cpu_usage, memory_usage


def time_train_frame(training_day, date):
    """
    학습 기간 일별 데이터프레임 형성

    Args:
        training_day: 학습기간(일)
        date: 현재 일자

    Returns:
        start_date: 학습 시작일자
        end_date: 학습 종료일자
        ts_days: 학습 시작~ 종료 일자 까지의 dataframe
    """

    ## 학습 마지막일
    end_date = date + timedelta(days = -1)
    ## 학습 시작일
    start_date = end_date + timedelta(days = -(training_day - 1))

    ## 형식 변환
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')

    ## 학습일자를 담은 DataFrame
    ts_days = pd.DataFrame(pd.date_range(start_date, end_date).strftime('%Y-%m-%d'))
    ts_days.columns = ['Date']

    return start_date, end_date, ts_days


def time_predict_frame(predict_day, date):
    """
    예측 기간 일별 데이터프레임 형성
    
    Args:
        predict_day  : 예측 기간(일자)
        date: 현재 일자
    
    Returns:
        pred_startDate : 학습 시작일자
        pred_day_endDate : 학습 종료일자(일별 예측)
        pred_mon_endDate : 학습 종료일자(월별 예측)
        ts_days : 예측 시작 일자 ~ 종료 일자 까지의 dataframe
    """
    
    ## 예측 시작일
    pred_start_date = date
    ## 예측 마지막일
    pred_day_end_date = pred_start_date + timedelta(days = (predict_day - 1))

    ## 형식 변환
    pred_start_date = pred_start_date.strftime('%Y-%m-%d')
    pred_day_end_date = pred_day_end_date.strftime('%Y-%m-%d')

    ## 예측일자를 담은 DataFrame
    ts_days = pd.DataFrame(pd.date_range(pred_start_date, pred_day_end_date).strftime('%Y-%m-%d'))
    ts_days.columns = ['Date']

    return pred_start_date, pred_day_end_date, ts_days