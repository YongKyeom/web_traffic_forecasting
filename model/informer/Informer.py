import re
import os

import re
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from model.informer.model import Informer
from model.informer.tools import adjust_learning_rate
from model.informer.tools import StandardScaler
from model.informer.tools import _process_one_batch
from model.informer.timefeatures import time_features


import matplotlib.pyplot as plt
from utils.metrics import SMAPE
from utils.metrics import RMSE


class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_epoch = 0

    def __call__(self, epoch, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.counter = 0


class Dataset_pred(Dataset):
    def __init__(self, dataframe, size=None, scale=True):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.dataframe = dataframe

        self.scale = scale
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.dataframe
        df_raw["date"] = pd.to_datetime(df_raw["date"])

        delta = df_raw["date"].iloc[1] - df_raw["date"].iloc[0]
        if delta >= timedelta(hours=1):
            self.freq = "h"
        else:
            self.freq = "t"

        border1 = 0
        border2 = len(df_raw)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Model_Informer:
    """
    Informer 모델을 학습한 후 예측하는 클래스

    Args:
        page_nm: 페이지명
        start_date: 학습 시작일
        end_date: 학습 종료일
        traindata: 학습 데이터
        testdata: (수동) 검증 시 검증 데이터
        device: Pytorch 설정(CPU, GPU)
        loss_plot_yn: Plot을 png파일로 저장할 지 여부
        valid_yn: 수동으로 검증하는 지 여부, True일 경우 testdata를 입력으로 받아 loss_plot을 생성함(Only)
        para_tune_yn: Train Epoch 최적화 여부(Early Stopping 기능 활용)
        patience: Early stopping으로 하이퍼 파라미터 최적화 시 사용
        loss_plot_path: loss_plot을 저장할 경로
        seq_len: Input sequence length of encoder, look back window
        lebel_len: Start token length of Decoder
        pred_len: Prediction sequence length
        learning_rate: 학습률
        train_epochs: 총 학습 횟수
        batch_size: mini-batch 크기
        n_heads: attension head의 갯수
        e_layers: 인코딩 레이어 수
        d_layers: 디코딩 레이어 수
        dropout: dropout 비율
        d_model: 임베딩 모델 차원수
        d_ff: d_ff
        embed: Time Feature Embedding, options: [timeF, fixed, learned]
        freq: For time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]
        adjust_learning_rate_yn: Adjusted learning rate 적용 여부
    """

    def __init__(
            self,
            page_nm: str,
            start_date: str,
            end_date: str,
            traindata: pd.DataFrame,
            testdata: pd.DataFrame,
            x_col_nm: str,
            y_col_nm: str,
            device,
            loss_plot_yn: None = False,
            valid_yn: None = False,
            para_tune_yn: bool = False,
            patience: int = 100,
            loss_plot_path: str = "./logs/TrainLoss/Informer",
            pred_len: int = 64,
            seq_len: int = 30,
            label_len: int = 30,
            learning_rate: float = 1e-4,
            train_epochs: int = 100,
            batch_size: int = 10,
            n_heads: int = 8,
            e_layers: int = 3,
            d_layers: int = 2,
            dropout: float = 0,
            d_model: int = 512,
            d_ff: int = 512,
            embed: str = "fixed",
            freq: str = "d",
            adjust_learning_rate_yn: bool = False
    ):
        self.page_nm = page_nm
        self.start_date = start_date
        self.end_date = end_date
        self.traindata = traindata
        self.testdata = testdata
        self.x_col_nm = x_col_nm
        self.y_col_nm = y_col_nm
        self.device, = device,
        self.loss_plot_yn = loss_plot_yn
        self.valid_yn = valid_yn
        self.para_tune_yn = para_tune_yn
        self.patience = patience
        self.loss_plot_path = loss_plot_path
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.label_len = label_len
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        self.d_model = d_model
        self.d_ff = d_ff
        self.embed = embed
        self.freq = freq
        self.adjust_learning_rate_yn = adjust_learning_rate_yn

        ## 학습/추론 데이터 생성(for pytorch)
        self._make_data()

    def _make_data(self):
        data = self.traindata.copy()
        data = data.reset_index(level=0)

        # 변수명 변경 처리
        data["date"] = data[self.x_col_nm]
        data["date"] = pd.to_datetime(data["date"])
        data["value"] = data[self.y_col_nm]

        # Normalization
        self.min_max_scaler = MinMaxScaler()
        data["value"] = self.min_max_scaler.fit_transform(
            data["value"].to_numpy().reshape(-1, 1)
        ).reshape(-1)
        data = data[["date", "value"]]

        self.data_train = data.copy()

        ## Parameter setting
        shuffle_flag = True
        num_workers = 0
        drop_last = True

        ## Dataset for torch
        self.dataset = Dataset_pred(
            dataframe = self.data_train,
            scale = True,
            size = (self.seq_len, self.label_len, self.pred_len)
        )
        self.data_loader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = shuffle_flag,
            num_workers = num_workers,
            drop_last = drop_last,
        )

    ## Informer 예측함수 선언
    def fn_predict_informer(self):
        scaler = self.dataset.scaler
        df_test = self.data_train.copy()
        df_test["value"] = scaler.transform(df_test["value"])
        df_test["date"] = pd.to_datetime(df_test["date"].values)

        delta = df_test["date"][1] - df_test["date"][0]
        for i in range(self.pred_len):
            # df_test = df_test.append({"date": df_test['date'].iloc[-1] + delta}, ignore_index=True)
            df_test = pd.concat(
                [df_test, pd.DataFrame({"date": [df_test["date"].iloc[-1] + delta]})],
                ignore_index = True,
            )
        df_test = df_test.fillna(0)

        df_test_x = df_test.iloc[-self.seq_len - self.pred_len: -self.pred_len].copy()
        df_test_y = df_test.iloc[-self.label_len - self.pred_len:].copy()

        df_test_numpy = df_test.to_numpy()[:, 1:].astype("float")
        test_time_x = time_features(df_test_x, freq=self.dataset.freq)
        test_data_x = df_test_numpy[-self.seq_len - self.pred_len: -self.pred_len]

        test_time_y = time_features(df_test_y, freq=self.dataset.freq)
        test_data_y = df_test_numpy[-self.label_len - self.pred_len:]
        test_data_y[-self.pred_len:] = np.zeros_like(test_data_y[-self.pred_len:])

        test_time_x = test_time_x
        test_time_y = test_time_y
        test_data_x = test_data_x.astype(np.float64)
        test_data_y = test_data_y.astype(np.float64)

        _test = [(test_data_x, test_data_y, test_time_x, test_time_y)]
        _test_loader = DataLoader(_test, batch_size=1, shuffle=False)

        preds = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(_test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            outputs = self.model(batch_x, batch_x_mark, batch_y, batch_y_mark)
            preds = outputs.detach().cpu().numpy()

        preds = self.dataset.scaler.inverse_transform(preds[0])

        df_test.iloc[-self.pred_len:, 1:] = preds

        ## Prediction Value
        result = df_test["value"].iloc[-self.pred_len:].to_numpy()
        result = self.min_max_scaler.inverse_transform(result.reshape(-1, 1)).reshape(-1)

        return result

    def test(self):
        ## Final Prediction
        return self.fn_predict_informer()

    def train(self):
        ## Set default Parameter
        enc_in = 1
        dec_in = 1
        c_out = 1

        ## Define Model
        self.model = Informer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=self.pred_len,
            device=self.device,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            dropout=self.dropout,
            d_model=self.d_model,
            d_ff=self.d_ff,
            embed=self.embed,
            freq=self.freq
        ).to(self.device)

        ## Point Estimator
        criterion = nn.MSELoss()
        ## Optimizer
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        ## Train Epochs
        if self.para_tune_yn is True:
            ## Hyper-Parameter Tuning시, loss를 출력하지 않음
            progress = range(self.train_epochs)
        else:
            progress = tqdm(range(self.train_epochs))
        ## Train start
        self.model.train()
        ## Loss
        train_loss_ls = []
        val_loss_ls = []
        ## Early stop for Tuning Train Epochs
        early_stopping = EarlyStopping(patience=self.patience)

        for epoch in progress:
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(self.data_loader):
                model_optim.zero_grad()
                pred, true = _process_one_batch(
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    device=self.device,
                    pred_len=self.pred_len,
                    label_len=self.label_len,
                    model=self.model,
                )
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            ## Train Loss
            train_loss = np.average(train_loss)
            ## Append loss result
            train_loss_ls.append(train_loss)

            ## Validation Loss
            if self.valid_yn is True:
                valid_pred = self.fn_predict_informer()
                if self.para_tune_yn is True:
                    ## Hyper-parameter 튜닝시 RMSE 계산(최적화용) -> 학습 데이터의 직전 7일치에 대해서만 최적 Epoch 찾음
                    valid_loss = RMSE(
                        pred=valid_pred[:7], true=self.testdata[self.y_col_nm].values
                    )
                else:
                    valid_loss = SMAPE(
                        pred=valid_pred, true=self.testdata[self.y_col_nm].values
                    )
                ## Append loss result
                val_loss_ls.append(valid_loss)

                if self.para_tune_yn is True:
                    ## 최소 학습 Epoch: 30
                    if (epoch + 1) >= 30:
                        early_stopping(epoch + 1, valid_loss)
                        if early_stopping.early_stop:
                            ## Best Epochs
                            self.best_epoch = early_stopping.best_epoch
                            break
                        elif (epoch + 1) == self.train_epochs:
                            self.best_epoch = self.train_epochs
                else:
                    ## Hyper-Parameter Tuning시, loss를 출력하지 않음
                    progress.set_description(
                        "Train loss: {:0.6f}, Valid loss: {:0.6f}".format(
                            train_loss, valid_loss
                        )
                    )

            else:
                ## Print
                progress.set_description("Train loss: {:0.6f}".format(train_loss))

            ## Adjust learning rate
            if self.adjust_learning_rate_yn is True:
                ## 20 epoch 이후 적용
                if epoch > 20:
                    ## epoch 30 이후 적용
                    adjust_learning_rate(model_optim, epoch + 1, "type1", self.learning_rate)

        ## Loss Plot
        if self.loss_plot_yn is True:
            ## Disable output showing
            plt.ioff()
            ## Define Plot
            fig = plt.figure(figsize=(7, 7))

            ## Plot1: View ~ Date
            if self.valid_yn is True:
                ax1 = fig.add_subplot(311)
            else:
                ax1 = fig.add_subplot(211)
            ax1.set_title(f"{self.y_col_nm} ~ {self.x_col_nm}")
            ax1.set_xlabel(self.x_col_nm)
            ax1.set_ylabel(self.y_col_nm)
            ax1.plot(
                self.traindata[self.x_col_nm],
                self.traindata[self.y_col_nm],
                color="black", label="Train"
            )
            self.testdata["PREDICT_VIEW"] = self.test()
            if self.valid_yn is True:
                ax1.plot(
                    pd.to_datetime(self.testdata[self.x_col_nm]),
                    self.testdata[self.y_col_nm],
                    color="blue",
                    label="Test_Real",
                )
                ax1.plot(
                    pd.to_datetime(self.testdata[self.x_col_nm]),
                    self.testdata["PREDICT_VIEW"],
                    color="red",
                    label="Test_Predict",
                )
                ax1.legend()
            else:
                ax1.plot(
                    pd.to_datetime(self.testdata[self.x_col_nm]),
                    self.testdata["PREDICT_VIEW"],
                    color="red",
                    label="Test_Predict",
                )
            ## 지수표현 없애기
            ax1.ticklabel_format(axis='y',useOffset=False, style='plain')

            ## Plot2: Train Loss
            if self.valid_yn is True:
                ax2 = fig.add_subplot(312)
            else:
                ax2 = fig.add_subplot(212)
            ax2.set_title(
                "Informer Train Loss(MSE): {}({}_{})".format(
                    self.page_nm,
                    self.start_date.replace('-', ''),
                    self.end_date.replace('-', ''),
                )
            )
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.plot(
                [x + 1 for x in range(len(train_loss_ls))],
                train_loss_ls,
                color="black",
                label="Train loss",
            )
            ax2.legend()

            ## Plot3: Validation Loss
            if self.valid_yn is True:
                ax3 = fig.add_subplot(313)
                ax3.set_title(
                    "Informer Validation Loss(SMAPE): {}({}_{})".format(
                        self.page_nm,
                        self.start_date.replace('-', ''),
                        self.end_date.replace('-', ''),
                    )
                )
                ax3.set_xlabel("Epoch")
                ax3.set_ylabel("Loss")
                ax3.plot(
                    [x + 1 for x in range(len(val_loss_ls))],
                    val_loss_ls,
                    color="red",
                    label="Valid loss",
                )
                ax3.legend()

            plt.tight_layout()

            ## Plot 경로 생성
            if not os.path.exists(self.loss_plot_path):
                os.makedirs(self.loss_plot_path)
            ## Save Plot
            plt.savefig(
                "{}/Informer_Loss_{}({}_{}).png".format(
                    self.loss_plot_path,
                    re.sub(r'[:,\\|*?<>]', '_', self.page_nm),
                    self.start_date.replace('-', ''),
                    self.end_date.replace("-", ""),
                )
            )

            plt.close(fig)

