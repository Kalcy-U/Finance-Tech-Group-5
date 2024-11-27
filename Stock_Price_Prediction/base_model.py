from operator import imod
import numpy as np
import pandas as pd
import copy

from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric


class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= ''):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        self.fitted = False

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix


    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    def train_epoch(self, data_loader):
        self.model.train()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label           
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def test_epoch(self, data_loader):
        self.model.eval()
        losses = []

        for data in data_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            pred = self.model(feature.float())
            loss = self.loss_fn(pred, label)
            losses.append(loss.item())

        return float(np.mean(losses))

    def _init_data_loader(self, data, shuffle=True, drop_last=True):
        sampler = DailyBatchSamplerRandom(data, shuffle)
        data_loader = DataLoader(data, sampler=sampler, drop_last=drop_last)
        return data_loader

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = True

    def fit(self, dl_train, dl_valid):
        train_loader = self._init_data_loader(dl_train, shuffle=True, drop_last=True)
        valid_loader = self._init_data_loader(dl_valid, shuffle=False, drop_last=True)

        self.fitted = True
        best_param = None
        
        train_loss_list=[]
        val_loss_list=[]
        for step in range(self.n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.test_epoch(valid_loader)

            print("Epoch %d, train_loss %.6f, valid_loss %.6f " % (step, train_loss, val_loss))
            best_param = copy.deepcopy(self.model.state_dict())
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
            if train_loss <= self.train_stop_loss_thred:
                break
        torch.save(best_param, f'{self.save_path}{self.save_prefix}master_{self.seed}.pkl')

        plot_loss(train_loss_list,val_loss_list)

    def predict(self, dl_test):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        test_loader = self._init_data_loader(dl_test, shuffle=False, drop_last=False)

        preds = []
        ic = []
        ric = []

        self.model.eval()
        for data in test_loader:
            data = torch.squeeze(data, dim=0)
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1]
            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label.detach().numpy())
            ic.append(daily_ic)
            ric.append(daily_ric)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())
        y_gt=dl_test.iloc[:,-1]
        dl_test.data.iloc[:,-1]
        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }
        return predictions, metrics

def plot_loss(train_loss,val_loss):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig("tain_loss.png")
    plt.figure(figsize=(10, 6))
    plt.plot(val_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig("valid_loss.png")
def _prepare_data(data):
    # 将(N, T, F)的数据转换为lightgbm需要的格式
    sampler = DailyBatchSamplerRandom(data, shuffle=False)
    dl = DataLoader(data, sampler=sampler, drop_last=False)
    
    data = []
    labels = []
    for batch in dl:
        batch = torch.squeeze(batch, dim=0)
        features = batch[:, :, 0:-1].numpy()  # (N, T, F-1)
        label = batch[:, -1, -1].numpy()  # 最后一个时间点的label
        
        # 将时序特征展平
        flat_features = features.reshape(features.shape[0], -1)
        data.append(flat_features)
        labels.append(label)
        
    return np.vstack(data), np.concatenate(labels)
class LGBModel():
    def __init__(self, params=None, seed=None, save_path='model/', save_prefix=''):
        self.params = params if params is not None else {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': seed
        }
        self.seed = seed
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.fitted = False
        

        
    def fit(self, dl_train, dl_valid):
        import lightgbm as lgb
        
        # 准备训练和验证数据
        X_train, y_train = _prepare_data(dl_train)
        X_valid, y_valid = _prepare_data(dl_valid)
        
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        
        # 训练模型
        
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=1000,

        )
        
        self.fitted = True
        # 保存模型
        self.model.save_model(f'{self.save_path}{self.save_prefix}lgb_{self.seed}.txt')
        
    def predict(self, dl_test):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
            
        # 准备测试数据
        X_test, y_test =_prepare_data(dl_test)
        
        # 预测
        preds = self.model.predict(X_test)
        
        # 计算IC和RIC
        ic = []
        ric = []
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 12))
        # 按日期分组计算IC
        dates = dl_test.get_index().get_level_values('datetime').unique()
        stock_count=3
        stocks_pred=[[]for _ in range(stock_count)]
        stocks_true=[[]for _ in range(stock_count)]
        i=0
        curr_pos = 0
        for date in dates:
            date_mask = dl_test.get_index().get_level_values('datetime') == date
            n_stocks = sum(date_mask)
            
            daily_pred = preds[curr_pos:curr_pos + n_stocks]
            daily_label = y_test[curr_pos:curr_pos + n_stocks]
            for i in range(stock_count):
                stocks_pred[i].append(daily_pred[i])
                stocks_true[i].append(daily_label[i])
            
            daily_ic, daily_ric = calc_ic(daily_pred, daily_label)
            ic.append(daily_ic)
            ric.append(daily_ric)
            
            curr_pos += n_stocks
            
        predictions = pd.Series(preds, index=dl_test.get_index())
        
        # # 绘制前3支股票的预测值和真实值对比
        # for i in range(stock_count):
        #     plt.subplot(stock_count, 1, i+1)
        #     plt.plot(stocks_pred[i], label='pred')
        #     plt.plot(stocks_true[i], label='true') 
        #     plt.legend()
        #     plt.grid(True)
        # plt.savefig(f'{self.save_path}{self.save_prefix}pred_vs_true.png')
        # plt.close()
        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }
        
    
        return predictions, metrics
    
    def load_param(self, param_path):
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=param_path)
        self.fitted = True
        return self
    
def simulate_trading( dl_test,predictions):
    """模拟每日交易,选择预测收益率最高的前30支股票"""
    
    dates = dl_test.get_index().get_level_values('datetime').unique()
    daily_returns = []  # 策略每日收益率
    benchmark_returns = []  # 基准收益率(全市场等权重)
    X_test, y_test =_prepare_data(dl_test)
    
    for date in dates:
        # 获取当天的预测值和真实收益率
        date_mask = predictions.index.get_level_values('datetime') == date
        daily_pred = predictions[date_mask]  # 当天所有股票的预测收益率
        daily_label = y_test[date_mask]  # 当天所有股票的实际收益率
        
        # 选择预测收益率最高的前30支股票构建等权重组合
        top_30_mask = daily_pred.argsort()[-30:]  # 获取前30支股票的位置
        portfolio_return = daily_label[top_30_mask].mean()  # 计算组合收益率
        benchmark_return = daily_label.mean()  # 计算基准收益率
        
        daily_returns.append(portfolio_return)
        benchmark_returns.append(benchmark_return)
        
    # 计算策略表现
    excess_returns = np.array(daily_returns) - np.array(benchmark_returns)  # 每日超额收益率
    ann_excess_return = np.sum(excess_returns) * 252  # 年化超额收益率(假设一年252个交易日)
    ir = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # 信息比率(年化超额收益率/年化波动率)
    
    return {
        'AR': ann_excess_return,
        'IR': ir
    }
    
