import numpy as np
import torch, os, pickle
from copy import deepcopy

def save_output(path, fn, output):
    with open(os.path.join(path, fn), 'wb') as f:
        pickle.dump(output, f)

class CrossValidation:
    def __init__(self, x, y=None, cv=5, return_index=False, seed=100):
        self._x = x
        self._y = y
        # index만 return
        self._index = return_index
        
        n_total = len(x)
        np.random.seed(seed)

        # data index 생성 및 shuffle
        idxs = np.arange(n_total)
        np.random.shuffle(idxs)

        chunk_size = int(n_total / cv) + 1

        # True 로 되어있는 array 생성
        cv_mask = np.ones_like(idxs, dtype=bool)
        self._mask = []
        for i in range(cv):
            mask = cv_mask.copy()
            # 일부 구간만 False로 변경
            mask[idxs[i*chunk_size:(i+1)*chunk_size]] = False
            self._mask.append(mask)
    
    def __getitem__(self, i):
        mask = self._mask[i] # i 번째 mask
        if self._index:
            return np.where(mask)[0], np.where(~mask)[0]
        else:
            return self._x[mask], self._y[mask], self._x[~mask], self._y[~mask]

class Trainer:
    def __init__(self, model, opt, crit):
        self._model = model
        self._opt = opt
        self._crit = crit

    def train(self, dataloader):
        self._model.train()
        train_loss = 0
        for feat, target in dataloader:
            pred = self._model(feat) # 예측
            loss = self._crit(pred, target) # loss 계산
            
            self._opt.zero_grad() # gradient 초기화
            loss.backward() # loss 역전파
            self._opt.step()
            
            train_loss += loss.detach().item()
        return train_loss / len(dataloader)
    
    def eval(self, dataloader):
        self._model.eval()
        eval_loss = 0
        eval_output = []
        with torch.no_grad():
            for feat, target in dataloader:
                pred = self._model(feat)
                loss = self._crit(pred, target)

                eval_loss += loss.detach().item()
                eval_output.append(pred)
        return eval_loss / len(dataloader), torch.concat(eval_output, dim=0).cpu()
    
    def pred(self, dataloader):
        self._model.eval()
        pred_output = []
        with torch.no_grad():
            for feat, _ in dataloader:
                pred = self._model(feat)
                pred_output.append(pred)
        return torch.concat(pred_output, dim=0).cpu()

class PyGTrainer:
    def __init__(self, model, opt, crit):
        self._model = model
        self._opt = opt
        self._crit = crit

    def train(self, dataloader):
        self._model.train()
        train_loss = 0
        for batch in dataloader: ### PyG Data
            pred = self._model(batch) # 예측 ###
            loss = self._crit(pred, batch.y) # loss 계산 ###
            
            self._opt.zero_grad() # gradient 초기화
            loss.backward() # loss 역전파
            self._opt.step()
            
            train_loss += loss.detach().item()
        return train_loss / len(dataloader)
    
    def eval(self, dataloader):
        self._model.eval()
        eval_loss = 0
        eval_output = []
        with torch.no_grad():
            for batch in dataloader:  ###
                pred = self._model(batch)  ###
                loss = self._crit(pred, batch.y)  ###

                eval_loss += loss.detach().item()
                eval_output.append(pred)
        return eval_loss / len(dataloader), torch.concat(eval_output, dim=0).cpu()
    
    def pred(self, dataloader):
        self._model.eval()
        pred_output = []
        with torch.no_grad():
            for batch in dataloader: ###
                pred = self._model(batch) ###
                pred_output.append(pred)
        return torch.concat(pred_output, dim=0).cpu()

class StandardGraphScaler:
    def __init__(self, attrs=['x','y']):
        self._attrs = attrs
        self._mean = {attr:0 for attr in attrs}
        self._std = {attr:1 for attr in attrs}
    
    def fit(self, data):
        for attr in self._attrs:
            dtype = str(type(getattr(data[0], attr)))
            value = [getattr(d, attr) for d in data]
            if 'torch' in dtype:
                dat = torch.concat(value, dim=0)
                self._mean[attr] = dat.mean(dim=0)
                self._std[attr] = dat.std(dim=0)
            else:
                dat = np.vstack(value)
                self._mean[attr] = dat.mean(axis=0)
                self._std[attr] = dat.std(axis=0)

    def transform_vector(self, data, attr):
        return (data - self.mean[attr]) / self.std[attr]

    def transform(self, data):
        # 원본 데이터 보존
        output = deepcopy(data)
        for d in output:
            for attr in self._attrs:
                v = getattr(d, attr)
                s = self.transform_vector(v, attr)
                setattr(d, attr, s)
        return output
        
    def inverse_transform_vector(self, data, attr):
        return data * self.std[attr] + self.mean[attr]
        
    def inverse_transform(self, data):
        # 원본 데이터 보존
        output = deepcopy(data)
        for d in output:
            for attr in self._attrs:
                v = getattr(d, attr)
                s = self.inverse_transform_vector(v, attr)
                setattr(d, attr, s)
        return output        

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        return self._std