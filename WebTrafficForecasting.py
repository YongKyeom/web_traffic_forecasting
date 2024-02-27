#########################################################################################
#########################################################################################
#### Load package
import sys
import os
import re
import gc
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

import ray
import torch

from utils.common_util import time_train_frame
from utils.common_util import time_predict_frame

from datetime import datetime
from datetime import timedelta
from utils.logger import Logger

from utils.train_predict import fn_pred_valid_arima
from utils.train_predict import fn_set_seed
from utils.train_predict import fn_pred_valid_dl

## Console color
from IPython.core.ultratb import ColorTB

sys.excepthook = ColorTB()

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

pd.options.display.float_format = "{:.2f}".format
pd.options.display.min_rows = 10
pd.options.display.max_rows = 100
pd.options.display.max_columns = None
pd.options.display.max_colwidth = 30


#########################################################################################
#########################################################################################
#### Logging 모듈


## Logging Path
CREATED_TIME = datetime.now()
LOG_PATH = "./logs/"
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)
## 로그 Naming
LOGFILE_NM = "web_traffic_forecast"
## Define Logger
logger = Logger(path=LOG_PATH, name=LOGFILE_NM, date=CREATED_TIME).logger

logger.info("Web Traffic Forecasing is Start")


#########################################################################################
#########################################################################################
#### Data 불러들이기

## File List
fileList = os.listdir("./data/")
train_ls = ["./data/" + file for file in fileList if file.startswith("train")]
key_ls = ["./data/" + file for file in fileList if file.startswith("key")]
sample_submission_ls = ["./data/" + file for file in fileList if file.startswith("sample_submission")]

## Load Data
trainData_DF = pd.concat([pd.read_csv(file) for file in train_ls], axis=0, ignore_index=True)
keyData_DF = pd.concat([pd.read_csv(file) for file in key_ls], axis=0, ignore_index=True)
submissionData_DF = pd.concat([pd.read_csv(file) for file in sample_submission_ls], axis=0, ignore_index=True)

logger.info("Train Data: {}".format(trainData_DF.shape))
trainData_DF.head(10)
logger.info("Key Data: {}".format(keyData_DF.shape))
keyData_DF.head(10)
logger.info("Submission Data: {}".format(submissionData_DF.shape))
submissionData_DF.head(10)

## Count Unique Page, ID
logger.info("N-Unique of Page in TrainData: {}".format(trainData_DF["Page"].nunique()))
logger.info("N-Unique of Page in KeyData: {}".format(keyData_DF["Page"].nunique()))
logger.info("N-Unique of ID in KeyData: {}".format(keyData_DF["Id"].nunique()))


#########################################################################################
#########################################################################################
#### 제출용 Data 가공(Page -> Page, Date)

keyData_DF.rename({"Page": "Page_Date"}, axis=1, inplace=True)
keyData_DF["Page"] = keyData_DF["Page_Date"].apply(lambda x: "_".join(x.split("_")[:-1]))
keyData_DF["Date"] = keyData_DF["Page_Date"].apply(lambda x: x.split("_")[-1])
keyData_DF.head(5)


#########################################################################################
#########################################################################################
#### 학습용 Data 생성

## Unpivot TrainData
trainData_Melt_DF = pd.melt(
    trainData_DF, id_vars="Page", var_name="Date", value_name="View", ignore_index=True
).dropna()

## 메모리 문제로 삭제
del trainData_DF
gc.collect()

logger.info("TrainData - Unpivot: {}".format(trainData_Melt_DF.shape))
trainData_Melt_DF.head(5)

## 상위 N 페이지
TOP_N = 100
pageViewMedian = trainData_Melt_DF.groupby("Page")["View"].median().reset_index(name="View_Median")
pageViewMedian.sort_values(["View_Median"], ascending=False, inplace=True)
topPageList = pageViewMedian[:TOP_N]["Page"].tolist()

## 상위 N개 페이지만 Filtering(Local PC 메모리 문제)
trainFilter_DF = (trainData_Melt_DF[trainData_Melt_DF["Page"].isin(topPageList)]).reset_index(drop=True)

logger.info("Target Page CNT: {}".format(trainFilter_DF["Page"].nunique()))


#########################################################################################
#########################################################################################
#### 예측모델 Setting
"""
    예측모델 방법론
    1. Auto_ARIMA
    2. CPT(Change Point Detection) + FREQ_Arima(통계적으로 Frequency Detection)
    3. Prophet
    4. Informer
    5. SCINet
    6. NLinear
    7. Augogleon(Automl 패키지) => Runtime이 너무 길어서 X
"""

## Train Loss 및 Prediction Plot 저장(.png) 여부
LOSS_PLOT_TF = True
## Validation Loss 계산 여부
VALID_YN = True

## DL모델의 Hyper-parameter Tuning 여부(Informer_New, SCINet, NLinear의 train_epoch)
## 운영시 True 가능 / False일 경우, 지정된 train_epoch만큼 학습하게 됨
PARA_TUNE_YN = True

DATA_PERCENTAGE = 0.7

## DL모델 종류: DL 알고리즘을 선택, options: ["Informer", "SCINet", "NLinear"]
## 운영 시 하나의 모델만 리스트로 담겨있어야 함(Ex. DL_Model_LS = ["NLinear"])
DL_Model_LS = ["Informer", "SCINet", "NLinear"]

## PyTorch Device: GPU 가용 시 변경 필요
TORCH_DEVICE = torch.device("cpu")

## 병렬 코어
CORE_CNT = os.cpu_count()
torch.set_num_threads(CORE_CNT)
torch.set_num_interop_threads(CORE_CNT)

## 분석대상 Page List
# pageList = trainFilter_DF["Page"].unique().tolist()
pageList = topPageList

## 예측기간(2017-09-11 ~ 2017-11-13)
PRED_DAY = 64  # (datetime(2017,11,13,0,0,0) - datetime(2017,9,11,0,0,0)).days + 1
TRAIN_DAY = 365


#########################################################################################
#########################################################################################
#### 모델검증 Setting

## 모델 검증을 위한 수행일자 설정
dateList = [datetime.strptime("2017-09-11", "%Y-%m-%d") + timedelta(days=-PRED_DAY * x0) for x0 in range(1, 6)]

i0 = 0
## 분석 기준일자, Ex) 2017-07-13
# ANAL_DATE = datetime.strptime("2017-09-11", "%Y-%m-%d")
ANAL_DATE = dateList[i0]

logger.info("Date: {}({}/{}) is Start".format(ANAL_DATE, i0 + 1, len(dateList)))

## 분석결과 폴더: EX) "./result/20230102/"
RESULT_PATH = "./result/{}/".format(re.sub("-", "", str(ANAL_DATE)[:10]))
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


## 학습 시작일, 마지막일, 일별 table, Ex) 2017-01-14 ~ 2017-07-12
START_DATE, END_DATE, TS_DAY = time_train_frame(TRAIN_DAY, ANAL_DATE)

## 예측 기간(일별 예측기간, 월별 예측기간), Ex) 2017-07-13 ~ 2017-09-12
(
    PRED_ST_DATE,
    PRED_DAY_END_DATE,
    predDays,
) = time_predict_frame(PRED_DAY, ANAL_DATE)

## ARIMA Plot 경로 생성
if not os.path.exists(RESULT_PATH + "/TrainLossPlot/ARIMA"):
    os.makedirs(RESULT_PATH + "/TrainLossPlot/ARIMA")

## DL모델의 Hyper-Parameter 세팅
ParaSpace = {
    "Informer": {
                "seq_len": PRED_DAY,
                "label_len": PRED_DAY,  
                "pred_len": PRED_DAY,
                "learning_rate": 1e-4,
                "train_epochs": 150 if PARA_TUNE_YN is True else 100,
                "batch_size": 16,
                "n_heads": 8,
                "e_layers": 3,
                "d_layers": 2,
                "dropout": 0,
                "d_model": 256,
                "d_ff": 256,
                "embed": "fixed",
                "freq": "d",
                "adjust_learning_rate_yn": True,
                "loss_plot_path": "{}/Informer".format(RESULT_PATH + "/TrainLossPlot"),
            },
    "SCINet": {
            "seq_len": PRED_DAY,
            "label_len": PRED_DAY,
            "pred_len": PRED_DAY,
            "train_epochs": 150 if PARA_TUNE_YN is True else 100,
            "kernel": 5,
            "embed": "timeF",  ## 'fixed', 'timeF'
            "dropout": 0.3,
            "lr": 1e-3,
            "freq": "b",
            "hidden_size": 250,
            "batch_size": 16,
            "levels": 3,
            "decompose": True,
            "adjust_learning_rate_yn": True,
            "path": "{}/SCINet".format(RESULT_PATH + "/TrainLossPlot"),
        },
    "NLinear": {
            "seq_len": PRED_DAY,
            "label_len": PRED_DAY,
            "pred_len": PRED_DAY,
            "train_epochs": 2000 if PARA_TUNE_YN is True else 1000,
            "learning_rate": 1e-4,
            "batch_size": 16,
            "enc_in": 3,
            "adjust_learning_rate_yn": True,
            "scale_yn": True,
            "loss_plot_path": "{}/NLinear".format(RESULT_PATH + "/TrainLossPlot"),
        },
}


#########################################################################################
#########################################################################################
#### ARIMA

## List를 CPU갯수 만큼 분할(Ex. [1,2,...,10], CPU 4개 => [[1,2,3], [4,5,6], [7,8,9], [10]]
def fn_split_list(mylist, split_cnt):
    return [
        mylist[i0 : (i0 + split_cnt)] for i0 in range(0, len(mylist), split_cnt)
    ]
pageSplit_LS = fn_split_list(pageList[:16], len(pageList[:16]) // CORE_CNT + 1)

TMP_ST_TIME = datetime.now()

## Parallel computing by Ray
ray.init(
    num_cpus=min(len(pageSplit_LS), CORE_CNT),
    local_mode=[True if min(len(pageSplit_LS), CORE_CNT) == 1 else False][0],
    # local_mode=True,
)

@ray.remote
def fn_predvalid_ray(
    page_nm_ls: list,
    data_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    pred_st_date: str,
    pred_day_end_date: str,
    pred_days: pd.DataFrame,
    created_time: None,
    log_path: str,
    logger_name: str,
    valid_yn: bool = False,
    data_percentage: float = 0.7,
    loss_plot_tf: bool = False,
    loss_plot_path: str = "./result/TrainLossPlot/ARIMA"
):
    """
    모델 학습 및 검증(fn_pred_valid_arima)을 병렬적으로 수행하는 함수
    Informer 등 Arima 외 모델을 적용할 경우
    해당 Page의 train/test를 담은 class를 리턴함
    (ChangePoint 함수를 두 번 수행하지 않기 위함)

    Returns:
            predict_day_df: 일별 예측치
            predict_summary_df: Arima 모델의 성능, 수행시간 등 요약정보
            dl_alg_id_ls: Arima외 모델을 적용할 경우 해당 AccountID에 대한 Class
    """

    import os
    import pandas as pd
    
    import logging
    from utils.logger import Logger
    
    from utils.train_predict import fn_pred_valid_arima

    import warnings
    
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore")

    logger = Logger(path=log_path, name=logger_name, date=created_time).logger
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    logger.info("PID:{} is Start".format(os.getpid()))

    ## 학습/예측 함수 수행
    predict_result = list(
        map(
            fn_pred_valid_arima,
            page_nm_ls,
            ['Date' for x in range(len(page_nm_ls))],
            ['View' for x in range(len(page_nm_ls))],
            [
                data_df[(data_df["Page"] == page_nm_ls[x])][["Date", "View"]]
                for x in range(len(page_nm_ls))
            ],
            [start_date for x in range(len(page_nm_ls))],
            [end_date for x in range(len(page_nm_ls))],
            [pred_st_date for x in range(len(page_nm_ls))],
            [pred_day_end_date for x in range(len(page_nm_ls))],
            [pred_days for x in range(len(page_nm_ls))],
            [logger for x in range(len(page_nm_ls))],
            [valid_yn for x in range(len(page_nm_ls))],
            [data_percentage for x in range(len(page_nm_ls))],
            [loss_plot_tf for x in range(len(page_nm_ls))],
            [loss_plot_path for x in range(len(page_nm_ls))]
        )
    )
    
    predict_day_df = pd.concat(
        [x[0] for x in predict_result], axis=0, ignore_index=True
    )
    predict_summary_df = pd.concat(
        [x[1] for x in predict_result], axis=0, ignore_index=True
    )
    dl_alg_id_ls = [x[2] for x in predict_result]

    logger.info("PID: {} is End".format(os.getpid()))

    return predict_day_df, predict_summary_df, dl_alg_id_ls

## Define map function
def fn_par_map(f, page_split_list):
    return [
        f.remote(
            page_list,
            trainData_Melt_DF[trainData_Melt_DF["Page"].isin(page_list)],
            START_DATE,
            END_DATE,
            PRED_ST_DATE,
            PRED_DAY_END_DATE,
            predDays,
            CREATED_TIME,
            LOG_PATH,
            LOGFILE_NM,
            VALID_YN,
            DATA_PERCENTAGE,
            LOSS_PLOT_TF,
            RESULT_PATH + "/TrainLossPlot/ARIMA",
        )
        for page_list in page_split_list
    ]
    
try:
    ## Do Parallel
    Result_ParMap = fn_par_map(fn_predvalid_ray, pageSplit_LS)
    ## Get Result
    ResultParallel = ray.get(Result_ParMap)

except Exception as e:
    logger.error("Error in Ray")
    raise Exception(e)

finally:
    ## Closs parallel computting
    ray.shutdown(True)

    ## 종료시간
    TMP_END_TIME = datetime.now()
    logger.info(
        "ARIMA Elapsed Time: {} Minutes".format(
            round((TMP_END_TIME - TMP_ST_TIME).seconds / 60, 2)
        )
    )

#### Get Arima Result
## 예측치
predictDay_DF = pd.concat([x[0] for x in ResultParallel], axis=0, ignore_index=True)
## Arima 모델 요약
predictSummary_DF = pd.concat(
    [x[1] for x in ResultParallel], axis=0, ignore_index=True
)
## DL 알고리즘 적용대상
dl_Alg_ID_LS = [x[2] for x in ResultParallel]
## Flatten
dl_Alg_ID_LS = [y for x in dl_Alg_ID_LS for y in x]
## Filter None
dl_Alg_ID_LS = list(filter(None, dl_Alg_ID_LS))

logger.info(
    "ARIMA SMAPE: {} w/ CNT: {}".format(
        predictSummary_DF["SMAPE"].median(), 
        predictSummary_DF.shape[1]
    )
)
logger.info("DL Target CNT: {}".format(len(dl_Alg_ID_LS)))
logger.info("Date: {}({}/{}) is End".format(ANAL_DATE, i0 + 1, len(dateList)))


#########################################################################################
#########################################################################################
#### 딥러닝

if (len(dl_Alg_ID_LS) > 0) and (len(DL_Model_LS) > 0):
    for DL_TYPE in DL_Model_LS:
        ## DL 결과 초기화
        PredictDay_DL_DF = pd.DataFrame()
        PredictSummary_DL_DF = pd.DataFrame()
        ERROR_DL = ""

        ## 시작시간
        TMP_ST_TIME = datetime.now()

        logger.info("{}(Deep learning) Start".format(DL_TYPE))

        ## 계정별 순차적으로 DL모델 학습 및 예측 진행
        for a0, dlClass in enumerate(dl_Alg_ID_LS):
            logger.info(
                "({}/{}){}: Use {}".format(
                    a0 + 1, len(dl_Alg_ID_LS), dlClass.page_nm, DL_TYPE
                )
            )

            ## Early stop 적용할 patience 수
            if DL_TYPE == "Informer":
                PATIENCE_CNT = 15
            elif DL_TYPE == "SCINet":
                PATIENCE_CNT = 30
            elif DL_TYPE == "NLinear":
                PATIENCE_CNT = 100
            else:
                PATIENCE_CNT = 30

            try:
                ## Run Informer
                fn_pred_valid_dl(
                    pred_result=dlClass,
                    device=TORCH_DEVICE,
                    loss_plot_yn=LOSS_PLOT_TF,
                    valid_yn=VALID_YN,
                    para_tune_yn=PARA_TUNE_YN,
                    patience=PATIENCE_CNT,
                    algo=DL_TYPE,
                    para_space=ParaSpace[DL_TYPE],
                    arima_loss_plot_path=RESULT_PATH + "/TrainLossPlot/ARIMA",
                    logger=logger
                )

                ## DL 모델의 예측 결과
                (
                    PredictDay_DL_TMP_DF,
                    PredictSummary_DL_TMP_DF
                ) = dlClass.model_performance(
                    valid_yn=VALID_YN,
                    loss_plot_tf=LOSS_PLOT_TF
                )

                ## Combine Result
                PredictDay_DL_DF = pd.concat(
                    [PredictDay_DL_DF, PredictDay_DL_TMP_DF],
                    axis=0,
                    ignore_index=True,
                )
                PredictSummary_DL_DF = pd.concat(
                    [PredictSummary_DL_DF, PredictSummary_DL_TMP_DF],
                    axis=0,
                    ignore_index=True,
                )

            except Exception as e:
                if VALID_YN is True:
                    ## 수동 검증 시, Error가 발생할 경우 해당 Page의 DL모델 Process를 중단함
                    raise Exception(e)
                else:
                    ## 최종 결과물의 경우, 해당 Page의 예측결과를 제외하고 다음 계정에 대해 진행
                    ## DL에서 Error가 발생하여, ARIMA 모델을 대체했음에도 에러가 발생한 경우임
                    logger.error(e)

        ## 최종결과 할당(Ex. Predict_Informer_DF, PredictSummary_Informer_DF, ERROR_INFORMER
        (
            globals()["predictDay_{}_DF".format(DL_TYPE)],
            globals()["predictSummary_{}_DF".format(DL_TYPE)]
        ) = (
            PredictDay_DL_DF.copy(),
            PredictSummary_DL_DF.copy()
        )

        ## 종료시간
        TMP_END_TIME = datetime.now()
        logger.info(
            "{} Elapsed Time: {} Minutes".format(
                DL_TYPE, round((TMP_END_TIME - TMP_ST_TIME).seconds / 60, 2)
            )
        )

## 모델별 일별 예측 결과 취합
predictDay_Final = pd.concat(
    [
        PredictDay_Informer_DF,
        PredictDay_SCINet_DF,
        PredictDay_NLinear_DF
    ],
    axis = 0,
    ignore_index=True
)
predictDay_Final['PREDICT_DATE'] = pd.to_datetime(predictDay_Final["PREDICT_DATE"])

logger.info('DataFrame of DL Prediction: {}'.format(predictDay_Final.shape))
print(predictDay_Final.head(5))

## 모델별 예측성능 요약 결과 취합
predictSummary_Final = pd.concat(
    [
        predictSummary_Informer_DF,
        predictSummary_SCINet_DF,
        predictSummary_NLinear_DF
    ],
    axis = 0,
    ignore_index=True
)

logger.info('DataFrame of DL Summary: {}'.format(predictSummary_Final.shape))
print(predictSummary_Final.head(5))
