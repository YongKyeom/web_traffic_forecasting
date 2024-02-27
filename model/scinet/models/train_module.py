import os
import warnings
warnings.filterwarnings("ignore")

import re
import pandas as pd
import numpy as np
import time
import matplotlib.pylab as plt

import torch
import torch.nn as nn

from tqdm import tqdm
from utils.metrics import SMAPE
from utils.metrics import RMSE

from torch.utils.data import DataLoader
from model.scinet.data_process.data_loader import Dataset_Custom
from model.scinet.data_process.data_loader import Dataset_Pred
from model.scinet.utils.tools import adjust_learning_rate
from model.scinet.models.SCINet import SCINet
from model.scinet.models.SCINet_decompose import SCINet_decompose


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


class Model_SCINet:
    """
    SCINet 모델을 학습한 후 예측하는 클래스

    Args:
        page_nm: 페이지명
        seq_len: Input sequence length of SCINet encoder, look back window
        lebel_len: Start token length of Decoder
        pred_len: Prediction sequence length
        window_size: Window 크기
        hidden_size: Hidden layer 갯수
        stacks: Stack 갯수
        levels
        num_decoder_layer: Decoder layer 갯수
        concat_len
        groups
        kernel: Kernal 사이즈
        dilation: dilation
        dropout: Dropout 비율
        single_step
        single_step_output_One: Only output the single final step
        lastWeight
        positionalEcoding
        RIN
        lr: Learning rate
        embed: Time Feature Embedding, options: [timeF, fixed, learned]
        lradj: Adjust learning rate
        freq: For time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]
        features: S: 단변량, M: 다변량
        inverse: Denorm the output data
    """

    def __init__(
            self,
            page_nm: str,
            start_date: str,
            end_date: str,
            device: None = None,
            decompose: bool = True,
            seq_len: int = 32,
            label_len: int = 30,
            pred_len: int = 30,
            hidden_size: int = 250,
            INN: int = 1,
            stacks: int = 1,
            levels: int = 3,
            num_decoder_layer: int = 1,
            concat_len: int = 0,
            groups: int = 1,
            kernel: int = 5,
            dilation: int = 1,
            dropout: float = 0.3,
            single_step: int = 0,
            single_step_output_One: int = 0,
            lastWeight: float = 1.0,
            positionalEcoding: bool = False,
            RIN: bool = False,
            lr: float = 1e-3,
            loss: str = "mse",
            train_epochs: int = 100,
            batch_size: int = 8,
            embed: str = "timeF",
            lradj: int = 1,
            inverse: bool = False,
            freq: str = "b",
            features: str = "S",
            data: pd.DataFrame = None,
            testdata: pd.DataFrame = None,
            x_col_nm: str = "Date",
            y_col_nm: str = "View",
            valid_yn: bool = True,
            para_tune_yn: bool = False,
            patience: int = 25,
            loss_plot_yn: bool = True,
            path: str = "./logs/TrainLoss/SCINet",
            adjust_learning_rate_yn: bool = True,
        ):
        self.page_nm = page_nm
        self.start_date = start_date
        self.end_date = end_date
        self.device = device
        self.decompose = decompose
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.INN = INN
        self.stacks = stacks
        self.levels = levels
        self.num_decoder_layer = num_decoder_layer
        self.concat_len = concat_len
        self.groups = groups
        self.kernel = kernel
        self.dilation = dilation
        self.dropout = dropout
        self.single_step = single_step
        self.single_step_output_One = single_step_output_One
        self.lastWeight = lastWeight
        self.positionalEcoding = positionalEcoding
        self.RIN = RIN
        self.lr = lr
        self.loss = loss
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.embed = embed
        self.lradj = lradj
        self.inverse = inverse
        self.freq = freq
        self.features = features

        self.data = data.copy()
        self.testdata = testdata.copy()
        self.x_col_nm = x_col_nm
        self.y_col_nm = y_col_nm
        self.valid_yn = valid_yn
        self.loss_plot_yn = loss_plot_yn
        self.para_tune_yn = para_tune_yn
        self.patience = patience

        self.path = path
        self.adjust_learning_rate_yn = adjust_learning_rate_yn

        ## Rename of Date Columns
        if x_col_nm != "date":
            self.data.rename(columns = {x_col_nm: "date"}, inplace = True)
            self.testdata.rename(columns={x_col_nm: "date"}, inplace=True)
            self.x_col_nm = 'date'

        ## Define Mddel
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        if self.decompose:
            model = SCINet_decompose(
                output_len=self.pred_len,
                input_len=self.seq_len,
                input_dim=[1 if self.features == "S" else 7][0],
                hid_size=self.hidden_size,
                num_stacks=self.stacks,
                num_levels=self.levels,
                # num_decoder_layer=self.num_decoder_layer,
                concat_len=self.concat_len,
                groups=self.groups,
                kernel=self.kernel,
                dropout=self.dropout,
                single_step_output_One=self.single_step_output_One,
                positionalE=self.positionalEcoding,
                modified=True,
                RIN=self.RIN,
            )
        else:
            model = SCINet(
                output_len=self.pred_len,
                input_len=self.seq_len,
                input_dim=[1 if self.features == "S" else 7][0],
                hid_size=self.hidden_size,
                num_stacks=self.stacks,
                num_levels=self.levels,
                num_decoder_layer=self.num_decoder_layer,
                concat_len=self.concat_len,
                groups=self.groups,
                kernel=self.kernel,
                dropout=self.dropout,
                single_step_output_One=self.single_step_output_One,
                positionalE=self.positionalEcoding,
                modified=True,
                RIN=self.RIN,
            )

        return model

    def _get_data(self, flag: str = "train", embed: str = "fixed"):
        timeenc = 0 if embed != "timeF" else 1

        Data = Dataset_Custom
        if flag == "train":
            shuffle_flag = True
            drop_last = True
            batch_size = self.batch_size

        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            Data = Dataset_Pred

        else:
            shuffle_flag = False
            drop_last = True
            batch_size = self.batch_size

        data_set = Data(
            self.data,
            size = [self.seq_len, self.label_len, self.pred_len],
            features = self.features,
            target = self.y_col_nm,
            inverse = self.inverse,
            timeenc = timeenc,
            freq = self.freq,
        )

        data_loader = DataLoader(
            data_set, batch_size=batch_size, shuffle=shuffle_flag, drop_last=drop_last
        )

        return data_set, data_loader

    def _select_optimizer(self):
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr
        )

    def _select_criterion(self, losstype="mse"):
        if losstype == "mse":
            criterion = nn.MSELoss()
        elif losstype == "mae":
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()

        return criterion

    def train(self):
        train_data, train_loader = self._get_data(flag='train')

        train_steps = len(train_loader)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.loss)

        ## Train Epochs
        if self.para_tune_yn is True:
            ## Hyper-Parameter Tuning시, loss를 출력하지 않음
            progress = range(self.train_epochs)
        else:
            progress = tqdm(range(self.train_epochs))
        ## Loss
        train_loss_ls = []
        valid_loss_ls = []
        ## Early stop for Tuning Train Epochs
        early_stopping = EarlyStopping(patience=self.patience)

        for epoch in progress:
            iter_count = 0
            train_loss = []

            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                    train_data, batch_x, batch_y)

                if self.stacks == 1:
                    loss = criterion(pred, true)
                elif self.stacks == 2:
                    loss = criterion(pred, true) + criterion(mid, true)
                else:
                    print('Error!')

                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            train_loss_ls.append(train_loss)

            if self.valid_yn is True:
                valid_pred = self.test().reshape(-1)
                if self.para_tune_yn is True:
                    ## Hyper-parameter 튜닝시 RMSE 계산(최적화용) -> 학습 데이터의 직전 7일치에 대해서만 최적 Epoch 찾음
                    valid_loss = RMSE(
                        pred = valid_pred[:7],
                        true = self.testdata[self.y_col_nm].values
                    )
                else:
                    ## Hyper-parameter 튜닝이 아닐 경우 SMAPE 계산(Only 확인용)
                    valid_loss = SMAPE(
                        pred = valid_pred,
                        true = self.testdata[self.y_col_nm].values
                    )
                valid_loss_ls.append(valid_loss)

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
                        "Epoch: {}, Steps: {} | Train Loss: {:.7f}, Valid Loss: {:.7f}".format(
                            epoch + 1, train_steps, train_loss, valid_loss
                        )
                    )
            else:
                progress.set_description(
                    "Epoch: {}, Steps: {} | Train Loss: {:.7f}".format(
                    epoch + 1, train_steps, train_loss
                    )
                )
            ## Adjust learning rate
            if self.adjust_learning_rate_yn is True:
                ## 125 epoch 이후 적용
                if (epoch + 1) >= 125:
                    lr = adjust_learning_rate(model_optim, epoch + 1, self.lradj, self.lr)

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
            ax1.set_title("{} ~ {}".format(self.y_col_nm, self.x_col_nm))
            ax1.set_xlabel(self.x_col_nm)
            ax1.set_ylabel(self.y_col_nm)
            ax1.plot(
                self.data[self.x_col_nm],
                self.data[self.y_col_nm],
                color="black",
                label="Train"
            )

            self.testdata["PREDICT_VIEW"] = self.test()[0]
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
                    pd.to_datetime(
                        self.testdata[self.x_col_nm].apply(
                            lambda x: x[:4] + "-" + x[4:6] + "-" + x[6:]
                        )
                    ),
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
                "SCINet Train Loss(MSE): {}({}_{})".format(
                    self.page_nm, self.start_date.replace('-', ''), self.end_date
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
                    "SCINet Validation Loss(SMAPE): {}({}_{})".format(
                        self.page_nm, self.start_date.replace('-', ''), self.end_date
                    )
                )
                ax3.set_xlabel("Epoch")
                ax3.set_ylabel("Loss")
                ax3.plot(
                    [x + 1 for x in range(len(valid_loss_ls))],
                    valid_loss_ls,
                    color="red",
                    label="Valid loss",
                )
                ax3.legend()

            plt.tight_layout()
            ## Plot 경로 생성
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            ## Save Plot
            plt.savefig(
                "{}/SCINet_Loss_{}({}_{}).png".format(
                    self.path,
                    re.sub(r'[:,\\|*?<>]', '_', self.page_nm),
                    self.start_date.replace('-', ''),
                    self.end_date.replace('-', ''),
                )
            )

            plt.close(fig)

    def test(self):
        test_data, test_loader = self._get_data(flag='pred')

        self.model.eval()

        preds = []
        trues = []
        mids = []
        pred_scales = []
        true_scales = []
        mid_scales = []

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            pred, pred_scale, mid, mid_scale, true, true_scale = self._process_one_batch_SCINet(
                test_data, batch_x, batch_y)

            if self.stacks == 1:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())
            elif self.stacks == 2:
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())
                mids.append(mid.detach().cpu().numpy())
                pred_scales.append(pred_scale.detach().cpu().numpy())
                mid_scales.append(mid_scale.detach().cpu().numpy())
                true_scales.append(true_scale.detach().cpu().numpy())

            else:
                print('Error!')

        if self.stacks == 1:
            preds = np.array(preds)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

            pred_scales = np.array(pred_scales)
            pred_scales = pred_scales.reshape(-1, pred_scales.shape[-2], pred_scales.shape[-1])

            return pred_scales

        elif self.stacks == 2:
            preds = np.array(preds)
            mids = np.array(mids)
            mid_scales = np.array(mid_scales)

            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            mids = mids.reshape(-1, mids.shape[-2], mids.shape[-1])
            mid_scales = mid_scales.reshape(-1, mid_scales.shape[-2], mid_scales.shape[-1])

            return mid_scales
        else:
            print('Error!')


    def _process_one_batch_SCINet(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        if self.stacks == 1:
            outputs = self.model(batch_x)
        elif self.stacks == 2:
            outputs, mid = self.model(batch_x)
        else:
            print('Error!')

        # if self.inverse:
        outputs_scaled = dataset_object.inverse_transform(outputs)
        if self.stacks == 2:
            mid_scaled = dataset_object.inverse_transform(mid)
        f_dim = -1 if self.features == 'MS' else 0
        batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)
        batch_y_scaled = dataset_object.inverse_transform(batch_y)

        if self.stacks == 1:
            return outputs, outputs_scaled, 0, 0, batch_y, batch_y_scaled
        elif self.stacks == 2:
            return outputs, outputs_scaled, mid, mid_scaled, batch_y, batch_y_scaled
        else:
            print('Error!')
