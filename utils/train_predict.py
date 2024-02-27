import os
import re
import random
import pandas as pd
import numpy as np
import torch
import datetime as dt
import matplotlib.pyplot as plt
import ruptures as rpt

from datetime import datetime
from datetime import timedelta

from preprocessing.frequency import fn_findfrequency

from pmdarima.arima import auto_arima
from model.informer.Informer import Model_Informer
from model.scinet.models.train_module import Model_SCINet
from model.nlinear.models.NLinear import Model_NLinear

from utils.metrics import MAPE
from utils.metrics import MdAPE
from utils.metrics import SMAPE
from utils.metrics import ADJ_MAPE
from utils.metrics import MASE


def fn_set_seed(seed: int = 2024):
    """
    Seed고정을 위한 함수
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)


def fn_pred_valid_arima(
    page_nm: str,
    date_nm: str,
    value_nm: str,
    page_nm_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    pred_st_date: str,
    pred_day_end_date: str,
    pred_days: pd.DataFrame,
    logger: None,
    valid_yn: bool,
    data_percentage: float,
    loss_plot_tf: bool,
    loss_plot_path: str,
):
    """
    PredValid 클래스를 이용하여 시계열 모델 학습 및 예측을 수행하는 함수

    Args:
        page_nm: 분석 대상 Page
        date_nm: Date 컬럼 이름
        value_nm: View 컬럼 이름
        page_nm_df: 분석 데이터의 DataFrame
        start_date: 분석대상 시작일
        end_date: 분석대상 종료일
        pred_st_date: 예측 시작일
        pred_day_end_date: 예측 종료일
        pred_days: 예측 시작일부터 종료일까지를 담은 데이터 프레임,
        logger: 로깅
        valid_yn: 모델예측력 검증여부로, True일 경우 모델의 예측성능을 계산함
        data_percentage: 학습 기간 내에 데이터 건수의 비율(Ex. 70%)이상 존재하는 지 여부를 확인하기 위함
        loss_plot_tf: 모델의 학습loss와 예측 그래프를 저장할 지 여부
        loss_plot_path: loss_plot_tf가 True인 경우, 해당 그래프를 저장할 경로

    Returns:
        predict_day: 분석 대상ID의 일별 예측치(DataFrame),
        predict_summary: 분석 대상ID의 예측모델에 대한 정보(오차, 알고리즘, 수행시간)(DataFrame)
        dl_alg_id: ARIMA외 모델(딥러닝)을 적용할 분석 대상ID 클래스
    """

    # logger.debug("PID: {}, Page: {} is Start".format(os.getpid(), page_nm))
    logger.debug("Page: {} is Start".format(page_nm))

    try:
        ## 예측 및 검증 클래스 선언
        pred_result = PredValid(
            page_nm,
            date_nm,
            value_nm,
            page_nm_df,
            start_date,
            end_date,
            pred_st_date,
            pred_day_end_date,
            pred_days,
        )

        ## 예외 로직
        if len(pred_result.train_df) < 7:
            predict_day = pd.DataFrame()
            predict_summary = pd.DataFrame()
            dl_alg_id = None

            pred_result.analysis_end_time = datetime.now()

            logger.warning("Insufficient Train Data(<7): {}".format(page_nm))

            return predict_day, predict_summary, dl_alg_id

        ## 예측 데이터가 부족한 경우(검증 시에만 사용, Submission에서는 사용X)
        if (valid_yn is True) and (len(pred_result.test) == 0):
            predict_day = pd.DataFrame()
            predict_summary = pd.DataFrame()
            dl_alg_id = None

            pred_result.analysis_end_time = datetime.now()

            logger.warning("Insufficient Test Data: {}".format(page_nm))

            return predict_day, predict_summary, dl_alg_id

        ## 학습Data 결측치 0으로 대체
        pred_result.missing_imputation()

        ## 학습 Data 확인: 학습Data가 70% 이상 존재 여부
        pred_result.check_traindata(data_percentage=data_percentage)

        ## 학습Data가 전체 기간 대비 70% 이상 존재하지 않는 경우
        if pred_result.data_check_ratio_yn is False:
            predict_day = pd.DataFrame()
            predict_summary = pd.DataFrame()
            dl_alg_id = None

            pred_result.analysis_end_time = datetime.now()

            logger.warning("Insufficient in Train Data(<{}%): {}".format(str(data_percentage * 100), page_nm))

            return predict_day, predict_summary, dl_alg_id

        ## ChangePoint 함수 수행
        pred_result.last_change_point()
        
        ## 학습 데이터 분할
        pred_result.define_traindata()

        ## Frequency 탐색
        try:
            if len(pred_result.train) > 65:
                pred_result.freq_yn, pred_result.freq_para = fn_findfrequency(
                    pred_result.train[[pred_result.date_nm, pred_result.value_nm]].copy()
                )
            else:
                ## Frequency 찾기에 Data수가 부족한 경우
                pred_result.freq_yn, pred_result.freq_para = False, {"with_intercept": True}
        except Exception as e:
            if valid_yn is True:
                ## 검증(수동) 시 Error발생할 경우 Process 종료
                raise Exception(e)
            else:
                ## 운영 시 Frequency 도중 에러가 발생할 경우, 기본 auto_arima 수행
                pred_result.freq_yn, pred_result.freq_para = False, {"with_intercept": True}

        
        ## 학습 데이터가 부족한 경우
        if len(pred_result.train) == 0:
            predict_day = pd.DataFrame()
            predict_summary = pd.DataFrame()
            dl_alg_id = None

            pred_result.analysis_end_time = datetime.now()

            logger.warning("Insufficient Train Data(==0): {}".format(page_nm))

            return predict_day, predict_summary, dl_alg_id

        ## ARIMA 모델을 사용할 지 여부 결정
        pred_result.arima_valid(valid_yn)

        ## ARIMA 모델 적용 여부
        if pred_result.modeltype == "ARIMA":
            ## ARIMA 학습 및 예측
            pred_result.pred_arima(valid_yn, loss_plot_tf, loss_plot_path)
            ## Arima 모델의 예측 결과
            (
                predict_day,
                predict_summary,
            ) = pred_result.model_performance(valid_yn=valid_yn, loss_plot_tf=loss_plot_tf)
            ## Arima외 모델을 사용할 ID
            dl_alg_id = None

            return predict_day, predict_summary, dl_alg_id
        else:
            if pred_result.freq_yn is False:
                #### Frequency 미적용해서 ARIMA검증에 Fail이 난 경우, DL 적용

                ## ARIMA 외 모델을 적용할 대상ID 리턴
                predict_day = pd.DataFrame()
                predict_summary = pd.DataFrame()
                dl_alg_id = pred_result

                return (
                    predict_day,
                    predict_summary,
                    dl_alg_id,
                )

            else:
                #### Frequency 적용해서 ARIMA검증에 Fail이 난 경우, ChangePoint적용하여 다시 검증 시도

                ## Frequency 백업
                freq_yn, freq_para = pred_result.freq_yn, pred_result.freq_para
                ## Frequency 미적용
                pred_result.freq_yn, pred_result.freq_para = False, {"with_intercept": True}
                ## ChangePoint 함수 수행
                pred_result.last_change_point()
                ## 학습 데이터 분할
                pred_result.define_traindata()
                ## ARIMA 모델을 사용할 지 말지 결정
                pred_result.arima_valid(valid_yn)

                ## ARIMA 모델 적용 여부
                if pred_result.modeltype == "ARIMA":
                    ## ARIMA 학습 및 예측
                    pred_result.pred_arima(valid_yn, loss_plot_tf, loss_plot_path)
                    ## Arima 모델의 예측 결과
                    (
                        predict_day,
                        predict_summary,
                    ) = pred_result.model_performance(valid_yn=valid_yn, loss_plot_tf=loss_plot_tf)
                    ## Arima외 모델을 사용할 ID
                    dl_alg_id = None

                    return (
                        predict_day,
                        predict_summary,
                        dl_alg_id,
                    )

                else:
                    ## Frequency값 원복(기록을 위함)
                    pred_result.freq_yn, pred_result.freq_para = freq_yn, freq_para

                    ## ARIMA 외 모델을 적용할 대상ID 리턴
                    predict_day = pd.DataFrame()
                    predict_summary = pd.DataFrame()
                    dl_alg_id = pred_result

                    return (
                        predict_day,
                        predict_summary,
                        dl_alg_id,
                    )

    except Exception as e:
        if valid_yn is True:
            ## 수동 검증 시, Error가 발생할 경우 해당 Process를 종료
            ## Error를 발생시켜 에러이력을 남김
            raise Exception("Error in Page: {} | Algorithm: Arima \n{}".format(page_nm, e))
        else:
            ## 운영 시, Error가 발생한 Page를 제외한 다른 Page들을 연산하기 위해 빈 결과를 리턴하도록 함
            ## 에러가 발생한 Page는 에러이력을 남김
            predict_day = pd.DataFrame()
            predict_summary = pd.DataFrame()
            dl_alg_id = None

            logger.error("Error in {} with ARIMA, \nError message: {}".format(page_nm, e))

            return (
                predict_day,
                predict_summary,
                dl_alg_id,
            )


def fn_pred_valid_dl(
    pred_result: None,
    device: None,
    loss_plot_yn: bool = True,
    valid_yn: bool = False,
    para_tune_yn: bool = False,
    patience: int = 30,
    algo: str = "Informer",
    para_space: dict = {},
    arima_loss_plot_path: str = "",
    logger: None = None,
):
    """
    'PredValid' 클래스를 이용한 결과, ARIMA외 Informer, SCINet, NLinear 등의 모델을 수행하는 PageNM에 대한
    클라우드 비용예측 모델의 예측 및 모델성능(검증) 결과를 수행하는 함수

    Args:
        pred_result: Page별 ChangePoint, 학습/검증 데이터 등의 정보를 담은 Class
        device: CPU 혹은 GPU 사용을 위한 PyTorch 세팅
        loss_plot_yn: Loss Plot을 저장할 지 여부
        valid_yn: Validation Loss를 계산할 지 여부, 수동으로 분석할 때만 사용할 것
        para_tune_yn: Informer,SCINet, NLinear 모델의 hyper-parameter tuning 여부(현재는 train_epoch만 가능)
        patience: para_tune_yn = True인 경우, Early stopping으로 조절한 인자
        algo: options = ["Informer", "SCINet", "NLinear"]
        para_space: DL모델의 Hyper parameter를 담은 Dictionary
        arima_loss_plot_path: 딥러닝모델에서 에러가 발생했을 경우, Arima모델로 대신 학습 및 추론을 함
                             이 때, loss_plot을 저장할 경우 저장할 경로 지정

    Returns:
        predict_day: 분석 대상ID의 일별 예측치(DataFrame),
        predict_summary: 분석 대상ID의 예측모델에 대한 정보(오차, 알고리즘, 수행시간)(DataFrame)
        err_msg: 에러 메시지(String)
    """

    from datetime import datetime

    ## 시작시간
    st_time = datetime.now()

    try:
        ## DL모델 학습 및 예측 수행
        if algo == "Informer":
            ## Infomer
            pred_result.pred_informer(
                device=device,
                loss_plot_yn=loss_plot_yn,
                valid_yn=valid_yn,
                para_tune_yn=para_tune_yn,
                patience=patience,
                **para_space
            )
        elif algo == "SCINet":
            ## SCINet
            pred_result.pred_scinet(
                device=device,
                loss_plot_yn=loss_plot_yn,
                valid_yn=valid_yn,
                para_tune_yn=para_tune_yn,
                patience=patience,
                **para_space
            )
        elif algo == "NLinear":
            pred_result.pred_nlinear(
                device=device,
                loss_plot_yn=loss_plot_yn,
                valid_yn=valid_yn,
                para_tune_yn=para_tune_yn,
                patience=patience,
                **para_space
            )
        else:
            raise KeyError("The Algorithm(algo) must be one of ['Informer', 'SCINet', 'NLinear']")

        ## Model Type
        pred_result.modeltype = algo

    except Exception as e:
        if valid_yn is True:
            ## 수동 검증 시, Error가 발생할 경우 Process를 종료
            ## Error를 발생시켜 에러이력을 남김
            raise Exception("Error in PageNM: {} | Algorithm: {} \n{}".format(pred_result.page_nm, algo, e))
        else:
            try:
                ## 최종 스크립트의 경우, Error발생 시, ARIMA모델로 대체
                ## 에러가 발생한 Page은 에러이력을 남김
                pred_result.pred_arima(
                    valid_yn=valid_yn,
                    loss_plot_yn=loss_plot_yn,
                    loss_plot_path=arima_loss_plot_path,
                )
                pred_result.modeltype = "ARIMA"

                logger.error(
                    "Error in {} with {}, Replace with ARIMA Model \nError message: {}".format(
                        pred_result.page_nm, algo, e
                    )
                )
            except:
                ## DL에러 발생후, ARIMA모델로 대체했음에도 에러가 발생하는 경우 Error를 발생시킴
                raise Exception(
                    "Error in PageNM: {} with Algorithm: {} & ARIMA \nError message:{}".format(
                        pred_result.page_nm, algo, e
                    )
                )

    finally:
        ## 수행시간
        pred_result.end_time = datetime.now()
        run_time = (pred_result.end_time - st_time).seconds

        ## ChangePoint/Arima모델 동작 시간을 더해줌
        pred_result.run_time = pred_result.arima_run_time + run_time


class PredValid:
    """
    클라우드 비용예측 모델의 예측 및 모델성능(검증) 결과를 수행하는 클래스
    병렬수행을 위해 ARIMA 모델만 수행

    Args:
        pange_nm: 분석 대상 Page명
        date_nm: x컬럼 이름
        value_nm: y컬럼 이름
        page_nm_df: Page에 대한 일별 데이터
        start_date: 분석대상 시작일(Ex. '20221001')
        end_date: 분석대상 종료일(Ex. '20221229')
        r: ryp2
        pred_st_date: 예측 시작일(Ex. '20221230'),
        pred_day_end_date: 예측 종료일(Ex. '202301228'),,
        pred_days: 예측 시작일부터 종료일까지를 담은 데이터 프레임,
        created_time: 분석 수행일(Ex. datetime.datetime(2023, 3, 2, 13, 55, 28, 72432))

    Returns:
        predict_day: 분석 대상ID의 일별 예측치(DataFrame),
        predict_summary: 분석 대상ID의 예측모델에 대한 정보(오차, 알고리즘, 수행시간)(DataFrame)
        err_msg: 에러 메시지(String)
    """

    def __init__(
        self,
        page_nm: str,
        date_nm: str,
        value_nm: str,
        page_nm_df: pd.DataFrame,
        start_date: str,
        end_date: str,
        pred_st_date: str,
        pred_day_end_date: str,
        pred_days: pd.DataFrame,
    ) -> None:
        self.st_time = datetime.now()
        self.page_nm = page_nm
        self.date_nm = date_nm
        self.value_nm = value_nm
        self.start_date = start_date
        self.end_date = end_date
        self.pred_st_date = pred_st_date
        self.pred_day_end_date = pred_day_end_date
        self.pred_days = pred_days

        ## DL모델의 Train epoch 초기화
        self.best_epoch = 100

        ## 학습/검증 데이터 필터링
        self.make_traindata(page_nm_df)

    def make_traindata(self, df):
        """
        학습 데이터로 필터링 하는 메소드
        """

        df[self.date_nm] = df[self.date_nm].astype(str).apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))

        df[self.value_nm] = round(df[self.value_nm], 0)

        ## 학습 전체 데이터
        self.train_df = df[
            (df[self.date_nm] >= pd.to_datetime(self.start_date)) & (df[self.date_nm] <= pd.to_datetime(self.end_date))
        ]

        ## 검증 데이터
        self.test = df[
            (df[self.date_nm] >= pd.to_datetime(self.pred_st_date))
            & (df[self.date_nm] <= pd.to_datetime(self.pred_day_end_date))
        ]

        ## 정렬
        self.train_df = self.train_df.sort_values(by=[self.date_nm], ascending=True).reset_index(drop=True)
        self.test = self.test.sort_values(by=[self.date_nm], ascending=True).reset_index(drop=True)

    def missing_imputation(self):
        """
        Data 내에 결측치가 있는 경우, 0으로 대체함(학습 데이터의 경우, 중간에 비는 날짜에 한함)
            Ex) 학습 기간이 10/1 ~ 12/29(90일)이고, TrainData가 10/16 ~ 12/29(75일) 존재하는 경우
                => 10/16 ~ 12/29 사이에 빈 값만 채움
            Ex) 테스트 기간의 경우, 테스트 시작일부터 마지막일 까지 빈 값을 채움
        """

        ## 학습Data 시작일
        train_date_min = self.train_df[self.date_nm].min()
        ## 학습Data 마지막일
        train_date_max = self.train_df[self.date_nm].max()
        ## 기준 Table
        train_date_df = pd.DataFrame({self.date_nm: pd.date_range(train_date_min, train_date_max)})

        ## 결측치 존재여부 확인
        if len(train_date_df) > len(self.train_df):
            ## 결측치가 있는 경우, 해당 일자의 View를 0으로 대체
            ## Left join
            self.train_df = pd.merge(train_date_df, self.train_df, how="left", on=self.date_nm)
            ## 결측치 대체
            self.train_df = self.train_df.fillna(0)
            ## 정렬
            self.train_df = self.train_df.sort_values([self.date_nm], ascending=True).reset_index(drop=True)

        ## 검증 Date기준 Table
        test_date_df = pd.DataFrame({self.date_nm: pd.date_range(self.pred_st_date, self.pred_day_end_date)})

        ## 결측치 존재여부 확인
        if len(test_date_df) > len(self.test):
            ## 결측치가 있는 경우, 해당 일자의 View를 0으로 대체
            ## Left join
            self.test = pd.merge(test_date_df, self.test, how="left", on=self.date_nm)
            ## 결측치 대체
            self.test = self.test.fillna(0)
            ## 정렬
            self.test = self.test.sort_values([self.date_nm], ascending=True).reset_index(drop=True)

    def last_change_point(self):
        """
        Change Point 함수를 수행한 후, 학습/예측 데이터셋을 생성하는 메소드
        """
        
        ## Pelt 알고리즘(CPT)
        cpt_pelt = rpt.Pelt(model="rbf", min_size=30).fit(self.train_df[self.value_nm].values)
        pelt_result = cpt_pelt.predict(pen = 3 * np.log(self.train_df.shape[0]))

        if len(pelt_result) == 1:
            ## ChangePoint가 없는 경우
            self.last_cp_pelt_yn = False
            self.last_cp_pelt = "No Cpt"
            self.last_cp = self.start_date
        else:
            ## ChangePoint가 있는 경우
            if (self.train_df.shape[0] - (pelt_result[-2] + 1)) < 7:
                ## CP가 있지만, 학습 데이터 마지막 7일 이내인 경우 미반영
                self.last_cp_pelt_yn = False
                self.last_cp_pelt = "No Cpt"
                self.last_cp = self.start_date
            else:
                self.last_cp_pelt_yn = True
                self.last_cp_pelt = pelt_result[-2]
                self.last_cp = str(self.train_df.iloc[self.last_cp_pelt][self.date_nm])[:10]

    def check_traindata(self, data_percentage: float = 0.0):
        """
        학습Data를 확인하여 예측을 수행할 지 여부를 결정하는 메소드

        Check Point:
            Data건 수가 전체 학습 기간 대비 70% 있는 지 여부

        Args:
             data_percentage: 학습Data 전체 기간 대비 최소 비율
        """

        ## 1) Data건 수가 전체 학습 기간 대비 70% 이상 존재
        if data_percentage > 0:
            if (
                len(self.train_df)
                >= len(
                    pd.date_range(
                        datetime.strptime(self.start_date, "%Y-%m-%d"),
                        datetime.strptime(self.end_date, "%Y-%m-%d"),
                    )
                )
                * data_percentage
            ):
                self.data_check_ratio_yn = True
            else:
                self.data_check_ratio_yn = False
        else:
            ## data_percentage == 0인 경우, 해당 Check로직 수행하지 않음
            self.data_check_ratio_yn = True

    def define_traindata(self):
        """
        학습/검증 데이터셋을 분리하는 메소드
            - Frequency 적용 시 학습 기간 전체를 학습 데이터로 사용함
            - Frequency 미적용 시 ChangePoint에 따라 학습 데이터를 분리하여 사용함
        """

        if self.last_cp_pelt_yn is True:
            self.train = self.train_df.iloc[self.last_cp_pelt :]
        else:
            self.train = self.train_df

        ## 정렬
        self.train.sort_values(self.date_nm, ascending=True, inplace=True)
        self.test.sort_values(self.date_nm, ascending=True, inplace=True)
        ## Index initilization
        self.train.reset_index(drop=True, inplace=True)
        self.test.reset_index(drop=True, inplace=True)

    def arima_valid(self, valid_yn):
        """
        Auto_Arima 모델을 사용할 지 검증하는 메소드
        Arima 모델 사용 조건
         - 학습 데이터 기간이 129일 미만(딥러닝 학습 불가)
         - 직전 7Day 기준 Arima 모델의 SMAPE가 15이상일 경우

         Args:
             valid_yn: 검증의 경우, ARIMA 모델의 예측성능을 계산함(딥러닝 모델과 비교를 위함)
        """

        try:
            ## (1) 학습기간이 충분하지 않으면 ARIMA로 확정
            if len(self.train) < 129:
                self.modeltype = "ARIMA"

            ## (2) 학습기간이 충분하면 train의 1주일치 데이터로 Validation 수행
            else:
                ## 검증 모델
                model_vali = auto_arima(self.train[self.value_nm][:-7], random_state=2024, **self.freq_para)
                ## 검증 데이터셋 예측치
                pred_values_vali = model_vali.predict(n_periods=7)

                ## 예측 성능 검증
                self.smape_vali = (
                    SMAPE(
                        pred=pred_values_vali.values,
                        true=self.train[self.value_nm][-7:].values,
                    )
                    * 100
                )

                ## (3) PASS 시에는 그대로 ARIMA 사용, FAIL 시에는 DL 수행
                if self.smape_vali < 15:
                    self.modeltype = "ARIMA"
                else:
                    self.modeltype = ""
        except:
            #### TypeError 발생할 경우

            ## Frequency 미적용(Default)
            self.freq_yn, self.freq_para = False, {"with_intercept": True}
            ## ChangePoint 함수 재수행
            self.last_change_point()
            ## 학습 데이터 분할
            self.define_traindata()

            ## (1) 학습기간이 충분하지 않으면 ARIMA로 확정
            if len(self.train) < 129:
                self.modeltype = "ARIMA"

            ## (2) 학습기간이 충분하면 train의 1주일치 데이터로 Validation 수행
            else:
                try:
                    ## 검증 모델
                    model_vali = auto_arima(
                        self.train[self.value_nm][:-7],
                        random_state=2024,
                        **self.freq_para
                    )
                    ## 검증 데이터셋 예측치
                    pred_values_vali = model_vali.predict(n_periods=7)

                except:
                    raise TypeError("Type Error in ARIMA(Last 7-Days Valid)")

                ## 예측 성능 검증
                self.smape_vali = (
                    SMAPE(
                        pred=pred_values_vali.values,
                        true=self.train[self.value_nm][-7:].values,
                    )
                    * 100
                )

                ## (3) PASS 시에는 그대로 ARIMA 사용, FAIL 시에는 DL 수행
                if self.smape_vali < 15:
                    self.modeltype = "ARIMA"
                else:
                    self.modeltype = ""

        ## 모델Type과 상관없이, ARIMA모델 학습 후 예측값 생성(+ ARIMA 예측구간 생성)
        self.pred_arima(valid_yn=valid_yn)

        ## 수행시간
        self.end_time = datetime.now()
        self.run_time = (self.end_time - self.st_time).seconds
        self.arima_run_time = self.run_time

    def pred_arima(
        self,
        valid_yn: bool = False,
        loss_plot_yn: bool = False,
        loss_plot_path: str = ".result/TrainLossPlot/ARIMA",
    ):
        """
        Arima모델을 학습한 후 예측결과를 리턴하는 메소드

        Args:
             valid_yn: 검증의 경우, ARIMA 모델의 예측성능을 계산함(딥러닝 모델과 비교를 위함)
        """

        try:
            ## ARIMA 모델링
            model = auto_arima(self.train[self.value_nm], random_state=2024, **self.freq_para)

            ## Prediction
            self.pred_values = model.predict(n_periods=len(self.pred_days))

            ## 기본 ARIMA
            def fn_default_arima():
                default_arima = auto_arima(self.train_df[self.value_nm], random_state=2024)

                return default_arima.predict(n_periods=len(self.pred_days))

            ## 기본 ARIMA 모델
            if self.freq_yn is True:
                ## Freq적용 시, 기본 auto_arima와 비교하기 위함

                ## Prediction
                self.pred_default_arima_values = fn_default_arima()
            else:
                ## Freq 미적용시
                if self.last_cp_pelt_yn is True:
                    ## ChangePointDetection 적용시, 기본 auto_arima와 비교 위함
                    ## Prediction
                    self.pred_default_arima_values = fn_default_arima()
                else:
                    ## Freq, CPT 둘 다 미적용인 경우
                    self.pred_default_arima_values = self.pred_values

        except:
            raise TypeError("Type Error in ARIMA(Final Prediction) in {}".format(self.page_nm))

        if valid_yn is True:
            ## ARIMA 성능
            self.mape_arima = MAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.mdape_arima = MdAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.smape_arima = SMAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.mase_arima = MASE(self.test[self.value_nm].values, self.pred_values)
            self.adj_mape_arima = ADJ_MAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.rev_mape_arima = MAPE(pred=self.test[self.value_nm].values, true=self.pred_values) * 100
            self.corr_arima = np.corrcoef(self.pred_values, self.test[self.value_nm].values)[0, 1]

        if loss_plot_yn is True:
            ## Disable output showing
            plt.ioff()
            ## Define Plot
            fig = plt.figure(figsize=(7, 7))

            ## Arima Hyper-Parameter
            arima_para = str(self.freq_para)

            ## Title
            if self.freq_yn is True:
                plt.title("ARIMA with Freq_YN: True \n'{}'".format(arima_para))
            else:
                plt.title("ARIMA with Freq_YN: False")

            ## Train
            plt.plot(
                self.train_df[self.date_nm],
                self.train_df[self.value_nm],
                color="black",
                label="Train",
            )
            plt.axvline(pd.to_datetime(self.last_cp), color="red", linestyle="--")
            plt.axvline(
                pd.to_datetime(self.end_date),
                color="red",
                linestyle="--",
            )
            ## Real
            if valid_yn is True:
                plt.plot(
                    self.test[self.date_nm],
                    self.test[self.value_nm],
                    color="blue",
                    label="Test_Real",
                )
            ## Prediction
            plt.plot(
                self.test[self.date_nm],
                self.pred_values,
                color="red",
                label="Test_Predict",
            )
            ## FREQ Arima 사용한 경우, Default Arima 예측치도 함께 Plot
            if (self.freq_yn is True) | (self.last_cp_pelt_yn is True):
                plt.plot(
                    self.test[self.date_nm],
                    self.pred_default_arima_values,
                    color="darkgray",
                    label="Test_Default_Arima",
                )

            ## y축 지수표현 없애기
            plt.ticklabel_format(axis="y", useOffset=False, style="plain")
            ## 범례
            plt.legend()
            ## plot layout
            plt.tight_layout()

            ## Save Plot
            try:
                plt.savefig(
                    "{}/ARIMA_Loss_{}({}_{}).png".format(
                        loss_plot_path,
                        re.sub(r'[:,\\|*?<>]', '_', self.page_nm),
                        self.last_cp.replace("-", ""),
                        self.end_date.replace("-", ""),
                    )
                )
            except:
                pass
            plt.close(fig)

    def pred_informer(self, device, loss_plot_yn, valid_yn, para_tune_yn, patience, **kwargs):
        """
        Informer모델을 학습한 후 예측결과를 리턴하는 메소드(개선 모델)
        """

        ## Hyper-parameter Tuning 여부
        if para_tune_yn is True:
            if len(self.train) >= 70:
                ## Seet Seed
                fn_set_seed()

                ## Define Informer Model
                informer_model_tuned = Model_Informer(
                    page_nm=self.page_nm,
                    start_date=self.last_cp,
                    end_date=self.end_date,
                    traindata=self.train.iloc[:-7],
                    testdata=self.train.iloc[-7:],
                    x_col_nm=self.date_nm,
                    y_col_nm=self.value_nm,
                    valid_yn=True,
                    patience=patience,
                    loss_plot_yn=False,
                    para_tune_yn=para_tune_yn,
                    device=device,
                    **kwargs
                )
                ## Train
                informer_model_tuned.train()
                ## Optimal Hyper-parameter
                kwargs["train_epochs"] = informer_model_tuned.best_epoch
                self.best_epoch = informer_model_tuned.best_epoch
            else:
                ## 학습 데이터가 부족한 경우, train_epochs은 80으로 고정
                kwargs["train_epochs"] = 80
                self.best_epoch = 80

        ## Seet Seed
        fn_set_seed()

        ## Initialize View Columns
        self.pred_days["View"] = 0

        ## Define Informer model
        informer_model = Model_Informer(
            page_nm=self.page_nm,
            start_date=self.last_cp,
            end_date=self.end_date,
            traindata=self.train,
            testdata=[self.test if valid_yn is True else self.pred_days][0],
            x_col_nm=self.date_nm,
            y_col_nm=self.value_nm,
            valid_yn=valid_yn,
            loss_plot_yn=loss_plot_yn,
            para_tune_yn=False,
            device=device,
            **kwargs
        )
        ## Train
        informer_model.train()
        ## Predict
        self.pred_values = informer_model.test()
    
    def pred_scinet(self, device, loss_plot_yn, valid_yn, para_tune_yn, patience, **kwargs):
        """
        SCINet모델을 학습한 후 에측결과를 리턴하는 메소드
        """

        ## Hyper-parameter Tuning 여부
        if para_tune_yn is True:
            if len(self.train) >= 70:
                ## Seet Seed
                fn_set_seed()

                ## Define SCINet Model
                scinet_model_tuned = Model_SCINet(
                    page_nm=self.page_nm,
                    start_date=self.last_cp,
                    end_date=self.end_date,
                    data=self.train.iloc[:-7],
                    testdata=self.train.iloc[-7:],
                    x_col_nm=self.date_nm,
                    y_col_nm=self.value_nm,
                    valid_yn=True,
                    patience=patience,
                    loss_plot_yn=False,
                    para_tune_yn=para_tune_yn,
                    device=device,
                    **kwargs
                )
                ## Train
                scinet_model_tuned.train()
                ## Optimal Hyper-parameter
                kwargs["train_epochs"] = scinet_model_tuned.best_epoch
                self.best_epoch = scinet_model_tuned.best_epoch
            else:
                ## 학습 데이터가 부족한 경우, train_epochs은 80으로 고정
                kwargs["train_epochs"] = 80
                self.best_epoch = 80

        ## Seet Seed
        fn_set_seed()

        ## Initialize View Columns
        self.pred_days["View"] = 0

        ## Define SCINet model
        scinet_model = Model_SCINet(
            page_nm=self.page_nm,
            start_date=self.last_cp,
            end_date=self.end_date,
            data=self.train,
            testdata=[self.test if valid_yn is True else self.pred_days][0],
            x_col_nm=self.date_nm,
            y_col_nm=self.value_nm,
            valid_yn=valid_yn,
            loss_plot_yn=loss_plot_yn,
            device=device,
            para_tune_yn=False,
            **kwargs
        )
        ## Train
        scinet_model.train()
        ## Predict
        self.pred_values = scinet_model.test().reshape(-1)

    def pred_nlinear(self, device, loss_plot_yn, valid_yn, para_tune_yn, patience, **kwargs):
        """
        NLinear 모델을 학습한 후 예측결과를 리턴하는 메소드
        """

        ## Hyper-parameter Tuning 여부
        if para_tune_yn is True:
            if len(self.train) >= 70:
                ## Seet Seed
                fn_set_seed()

                ## Define NLinear Model
                nlinear_model_tuned = Model_NLinear(
                    page_nm=self.page_nm,
                    start_date=self.last_cp,
                    end_date=self.end_date,
                    traindata=self.train.iloc[:-7],
                    testdata=self.train.iloc[-7:],
                    x_col_nm="Date",
                    y_col_nm="View",
                    valid_yn=True,
                    patience=patience,
                    loss_plot_yn=False,
                    para_tune_yn=para_tune_yn,
                    device=device,
                    **kwargs
                )
                ## Train
                nlinear_model_tuned.train()
                ## Optimal Hyper-parameter
                kwargs["train_epochs"] = nlinear_model_tuned.best_epoch
                self.best_epoch = nlinear_model_tuned.best_epoch
            else:
                ## 학습 데이터가 부족한 경우, train_epochs은 800으로 고정
                kwargs["train_epochs"] = 800
                self.best_epoch = 800

        ## Seet Seed
        fn_set_seed()

        ## Initialize View Columns
        self.pred_days["View"] = 0

        ## Define NLinear Model
        nlinear_model = Model_NLinear(
            page_nm=self.page_nm,
            start_date=self.last_cp,
            end_date=self.end_date,
            traindata=self.train,
            testdata=[self.test if valid_yn is True else self.pred_days][0],
            x_col_nm="Date",
            y_col_nm="View",
            valid_yn=valid_yn,
            loss_plot_yn=loss_plot_yn,
            device=device,
            patience=patience,
            para_tune_yn=False,
            **kwargs
        )
        ## Train
        nlinear_model.train()
        ## Predict
        self.pred_values = nlinear_model.test()

    def model_performance(
        self,
        valid_yn: bool = False,
        loss_plot_tf: bool = False,
    ):
        """
        Page별 모델, 예측값, 성능 등을 리턴하는 메소드
        """

        ## 예측결과 보정(음수)
        self.pred_values = np.where(self.pred_values < 0, 0, self.pred_values)

        ## 모델 결과(DataFrame)
        df_predict = pd.DataFrame(index=range(len(self.pred_days)))
        df_predict[self.date_nm] = self.pred_days[[self.date_nm]]
        df_predict[self.value_nm] = self.pred_values

        ## Page별 일별 예측값 결과
        predict_day = pd.DataFrame(index=range(len(self.pred_days)))
        predict_day["PAGE"] = self.page_nm
        predict_day["ALG_GUB"] = self.modeltype
        predict_day["PREDICT_DATE"] = df_predict[self.date_nm]
        predict_day["PREDICT_VIEW"] = np.round(df_predict[self.value_nm], 0)
        predict_day["REAL_VIEW"] = self.test[self.value_nm]

        ## Page별 모델정보 요약 Table
        if valid_yn is True:
            ## 모델 성능
            self.mape = MAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.mdape = MdAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.smape = SMAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.mase = MASE(self.test[self.value_nm].values, self.pred_values)
            self.adj_mape = ADJ_MAPE(pred=self.pred_values, true=self.test[self.value_nm].values) * 100
            self.rev_mape = MAPE(true=self.pred_values, pred=self.test[self.value_nm].values) * 100
            self.corr = np.corrcoef(self.pred_values, self.test[self.value_nm].values)[0, 1]

            ## Page별 모델정보 요약 Table
            predict_summary = pd.DataFrame(
                {
                    "PAGE": self.page_nm,
                    "ALG_GUB": self.modeltype,
                    "FREQ_YN": 1 if self.freq_yn is True else 0,
                    "FREQ_PARA": str(self.freq_para) if self.freq_yn is True else None,
                    "TRAIN_EPOCH": np.nan if self.modeltype == "ARIMA" else self.best_epoch,
                    "TRAINING_DAY_CNT": self.train.shape[0],
                    "MAPE": self.mape,
                    "SMAPE": self.smape,
                    "MDAPE": self.mdape,
                    "ADJ_MAPE": self.adj_mape,
                    "REV_MAPE": self.rev_mape,
                    "MASE": self.mase,
                    "CORRELATION": self.corr,
                    "MAPE_ARIMA": np.nan if self.modeltype == "ARIMA" else self.mape_arima,
                    "SMAPE_ARIMA": np.nan if self.modeltype == "ARIMA" else self.smape_arima,
                    "MDAPE_ARIMA": np.nan if self.modeltype == "ARIMA" else self.mdape_arima,
                    "ADJ_MAPE_ARIMA": np.nan if self.modeltype == "ARIMA" else self.adj_mape_arima,
                    "REV_MAPE_ARIMA": np.nan if self.modeltype == "ARIMA" else self.rev_mape_arima,
                    "MASE_ARIMA": np.nan if self.modeltype == "ARIMA" else self.mase_arima,
                    "CORRELATION_ARIMA": np.nan if self.modeltype == "ARIMA" else self.corr_arima,
                    "TRAINING_START": self.last_cp,
                    "TRAINING_END": self.end_date,
                    "PELT": self.last_cp_pelt_yn,
                    "RUN_TIME": self.run_time,
                },
                index=[0],
            )
        else:
            ## 검증을 수행하지 않는 경우
            predict_summary = pd.DataFrame(
                {
                    "PAGE": self.page_nm,
                    "ALG_GUB": self.modeltype,
                    "FREQ_YN": 1 if self.freq_yn is True else 0,
                    "FREQ_PARA": str(self.freq_para) if self.freq_yn is True else None,
                    "TRAIN_EPOCH": np.nan if self.modeltype == "ARIMA" else self.best_epoch,
                    "TRAINING_DAY_CNT": self.train.shape[0],
                    "TRAINING_START": self.last_cp,
                    "TRAINING_END": self.end_date,
                    "PELT": self.last_cp_pelt_yn,
                    "RUN_TIME": self.run_time,
                },
                index=[0],
            )

        ## Loss Plot File 이름
        if loss_plot_tf is True:
            loss_plot_file_nm = "TrainLossPlot/{}/{}_Loss_{}({}_{}).png".format(
                self.modeltype,
                self.modeltype,
                re.sub(r'[:,\\|*?<>]', '_', self.page_nm),
                self.last_cp.replace("-", ""),
                self.end_date.replace("-", "")
            )

            predict_summary["LOSS_PLOT_FILE"] = loss_plot_file_nm

        return predict_day, predict_summary
