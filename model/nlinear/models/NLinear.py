import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

from model.nlinear.utils.tools import _process_one_batch
from model.nlinear.data_process.data_process import Dataset_pred
from model.nlinear.utils.tools import adjust_learning_rate

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


class NLinear(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len, pred_len, enc_in, individual):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        
        return x  # [Batch, Output length, Channel]


class Model_NLinear:
    """
    NLinear 모델을 학습한 후 예측하는 클래스

    Args:
        page_nm: 페이지명
        start_date: 분석 시작일
        end_date: 분석 종료일
        traindata: 학습 데이터
        testdata: 예측 데이터
        x_col_nm: 날짜 컬럼명
        y_col_nm: Y 컬럼명
        device: PyTorch 장치(CPU, GPU)
        loss_plot_yn: Loss Plot을 저장할지 여부
        valid_yn: 예측 데이터에 대한 Loss를 계산할지 여부
        para_tune_yn: Train Epoch 최적화 여부(Early Stopping 기능 활용)
        patience: Early stopping으로 하이퍼 파라미터 최적화 시 사용
        loss_plot_path: loss_plot을 저장할 경로
        seq_len: Input sequence length of SCINet encoder, look back window
        lebel_len: Start token length of Decoder
        pred_len: Prediction sequence length
        learning_rate: 학습률
        train_epochs: 총 학습 횟수
        batch_size: mini-batch 크기
        enc_in: Linear Channel 수
        individual: individual
        scale_yn: 학습 시 Y값 Scale 적용 여부
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
        device: None,
        loss_plot_yn: bool = False,
        valid_yn: bool = False,
        para_tune_yn: bool = False,
        patience: int = 100,
        loss_plot_path: str = "./logs/TrainLoss/NLinear",
        seq_len: int = 30,
        label_len: int = 30,
        pred_len: int = 30,
        learning_rate: float = 1e-4,
        train_epochs: int = 1000,
        batch_size: int = 4,
        enc_in: int = 2,
        individual: None = None,
        scale_yn: bool = False,
        adjust_learning_rate_yn: bool = False,
    ):
        self.page_nm = page_nm
        self.start_date = start_date
        self.end_date = end_date
        self.traindata = traindata
        self.testdata = testdata
        self.x_col_nm = x_col_nm
        self.y_col_nm = y_col_nm
        self.device = device
        self.loss_plot_yn = loss_plot_yn
        self.valid_yn = valid_yn
        self.para_tune_yn = para_tune_yn
        self.patience = patience
        self.loss_plot_path = loss_plot_path
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.enc_in = enc_in
        self.individual = individual
        self.scale_yn = scale_yn
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

        ## Normalization <-- NLinear 학습 시 자동으로 normalization을 수행함
        if self.scale_yn is True:
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

        ## Dataset
        self.dataset = Dataset_pred(
            dataframe=self.data_train,
            scale=self.scale_yn,
            size=(self.seq_len, self.label_len, self.pred_len),
        )
        ## Data Loader
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_flag,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    ## 예측함수
    def fn_predict_NLinear(self):
        scaler = self.dataset.scaler
        df_test = self.data_train.copy()
        df_test["value"] = scaler.transform(df_test["value"])
        df_test["date"] = pd.to_datetime(df_test["date"].values)

        delta = df_test["date"][1] - df_test["date"][0]
        for i in range(self.pred_len):
            df_test = pd.concat(
                [df_test, pd.DataFrame({"date": [df_test["date"].iloc[-1] + delta]})],
                ignore_index=True,
            )
        df_test = df_test.fillna(0)

        df_test_numpy = df_test.to_numpy()[:, 1:].astype("float")

        test_data_x = df_test_numpy[-self.seq_len - self.pred_len : -self.pred_len]
        test_data_y = df_test_numpy[-self.label_len - self.pred_len :]
        test_data_y[-self.pred_len :] = np.zeros_like(test_data_y[-self.pred_len :])
        test_data_x = test_data_x.astype(np.float64)
        test_data_y = test_data_y.astype(np.float64)

        _test = [(test_data_x, test_data_y)]
        _test_loader = DataLoader(_test, batch_size=1, shuffle=False)

        preds = []

        for i, (batch_x, batch_y) in enumerate(_test_loader):
            batch_x = batch_x.float().to(self.device)
            outputs = self.model(batch_x)
            preds = outputs.detach().numpy().reshape(1, -1)

        # df_test.iloc[-pred_len:, 1] = preds
        if self.scale_yn is True:
            preds = scaler.inverse_transform(preds[0])
            result = self.min_max_scaler.inverse_transform(
                preds.reshape(-1, 1)
            ).reshape(-1)
        else:
            result = preds[0]

        return result

    def test(self):
        ## Final Prediction
        return self.fn_predict_NLinear()

    def train(self):
        ## Define NLinear Model
        self.model = NLinear(self.seq_len, self.pred_len, self.enc_in, self.individual).to(self.device)
        ## 최적화 Loss
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
            for i, (batch_x, batch_y) in enumerate(self.data_loader):
                model_optim.zero_grad()
                pred, true = _process_one_batch(
                    batch_x,
                    batch_y,
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
            train_loss_ls.append(train_loss)

            ## Validation Loss
            if self.valid_yn is True:
                valid_pred = self.fn_predict_NLinear()
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
                    ## 최소 학습 Epoch: 250
                    if (epoch + 1) >= 250:
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
                        "Train loss: {}, Valid loss: {}".format(train_loss, valid_loss)
                    )

            else:
                ## Print
                progress.set_description("Train loss: {}".format(train_loss))

            ## Adjust learning rate
            if self.adjust_learning_rate_yn is True:
                ## 125 epoch 이후 적용
                if (epoch + 1) >= 125:
                    adjust_learning_rate(model_optim, epoch + 1, "type1", self.learning_rate)

        ## Loss Plot
        if self.loss_plot_yn is True:
            ## Disable output showing
            plt.ioff()
            ## Define Plot
            fig = plt.figure(figsize=(7, 7))

            ## Plot1 : View ~ Date
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
                color="black",
                label="Train",
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

            ## Plot2 : Train Loss
            if self.valid_yn is True:
                ax2 = fig.add_subplot(312)
            else:
                ax2 = fig.add_subplot(212)
            ax2.set_title(
                "NLinear Train Loss(MSE): {}({}_{})".format(
                    self.page_nm,
                    self.start_date.replace("-", ""),
                    self.end_date.replace("-", "")
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
                    "NLinear Validation Loss(SMAPE): {}({}_{})".format(
                        self.page_nm,
                        self.start_date.replace("-", ""),
                        self.end_date.replace("-", "")
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
                "{}/NLinear_Loss_{}({}_{}).png".format(
                    self.loss_plot_path,
                    re.sub(r'[:,\\|*?<>]', '_', self.page_nm),
                    self.start_date.replace("-", ""),
                    self.end_date.replace("-", "")
                )
            )

            plt.close(fig)



