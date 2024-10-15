import time
import torch
import numpy as np
from tqdm import tqdm
from src.utils.metrics import masked_mae
from src.utils.metrics import masked_mape
from src.utils.metrics import masked_rmse
from src.utils.metrics import compute_all_metrics

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

class Engine():
    def __init__(self, device, model, optimizer, scheduler, 
                 s_mask, optimizer_s, scheduler_s,
                 node_training, node_test, node_frechet, node_sood_num, ssie, dataloader, 
                 scaler, sampler, loss_fn, log_dir, logger, args):
        super().__init__()
        self._device = device
        self.model = model
        self.model.to(self._device)
        self._optimizer = optimizer
        self._lr_scheduler = scheduler


        self.s_mask = s_mask
        # self.s_mask.to(self._device)

        self._optimizer_s = optimizer_s
        self._lr_scheduler_s = scheduler_s

        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._lrate = args.lrate
        self._clip_grad_value = args.clip_grad_value
        self._max_epochs = args.max_epochs
        self._patience = args.patience
        self._iter_cnt = 0
        self._mask_patience = args.mask_patience
        self._mask_cnt = 0

        self._node_training = node_training
        self._node_test = node_test
        self._node_sood_num = node_sood_num
        self._node_frechet = node_frechet

        self._alpha = args.alpha
        self._beta = args.beta
        self._seed = args.seed
        self._year = int(args.years)
        self._maxratio = args.max_increase_ratio
        self._c = args.c
        self._core = args.core
        self._head = args.head
        self._bs = args.bs
        self._seq_len = args.seq_len
        self._horizon = args.horizon

        self._tood = args.tood
        self._checkall = args.checkall

        self._save_path = log_dir
        self._logger = logger
        self._logger.info('The number of parameters: {}'.format(self.model.param_num()))


    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)


    def _to_numpy(self, tensors):
        if isinstance(tensors, list):
            return [tensor.detach().cpu().numpy() for tensor in tensors]
        else:
            return tensors.detach().cpu().numpy()


    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(array, dtype=torch.float32) for array in nparray]
        else:
            return torch.tensor(nparray, dtype=torch.float32)


    def _inverse_transform(self, tensors, cat='train'):
        def inv(tensor):
            return self._scaler[cat].inverse_transform(tensor)

        if isinstance(tensors, list):
            return [inv(tensor) for tensor in tensors]
        else:
            return inv(tensors)


    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_s{}_{}_maxratio{}_head{}_core{}.pt'.format(self._seed, 
                                                                 self._year, 
                                                                 self._maxratio, 
                                                                 self._head,
                                                                 self._core)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))


    def load_model(self, save_path):
        filename = 'final_model_s{}_{}_maxratio{}_head{}_core{}.pt'.format(self._seed, 
                                                                 self._year, 
                                                                 self._maxratio, 
                                                                 self._head,
                                                                 self._core)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))   


    def train_batch(self):
        self.model.train()
        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        # self._dataloader['train_loader'].shuffle()
        for _, data in tqdm(enumerate(self._dataloader['train_loader']), desc='Training'):
            self._optimizer.zero_grad()
            pm25, feature, _, day, week = data
            X = np.repeat(pm25[:, self._seq_len:self._seq_len+1], self._horizon, axis=1)
            feature = feature[:, self._seq_len:]
            day = day[:, self._seq_len:]
            week = week[:, self._seq_len:]
            X = np.concatenate([X, feature, day, week], axis=-1)
            label = pm25[:, self._seq_len:]

            # X (b, t, n, f), label (b, t, n, 1)
            X = X[:, :, self._node_training, :]
            label = label[..., self._node_training, :]
            X, label = self._to_device(self._to_tensor([X, label]))

            # t_mask, log_p_t = self.t_mask()
            # s_mask, log_p_s = self.s_mask()

            pred_corr = self.model(X)
            pred_corr, label = self._inverse_transform([pred_corr, label])

            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if label.min() < 1:
                mask_value = label.min()
            if self._iter_cnt == 0:
                print('Check mask value', mask_value)
            loss = self._loss_fn(pred_corr, label, mask_value)
    

            loss.backward()
            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._clip_grad_value)
            self._optimizer.step()

            mae = masked_mae(pred_corr, label, mask_value).item()
            mape = masked_mape(pred_corr, label, mask_value).item()
            rmse = masked_rmse(pred_corr, label, mask_value).item()
            train_loss.append(loss.item())
            train_mae.append(mae)
            train_mape.append(mape)
            train_rmse.append(rmse)

            self._iter_cnt += 1
            self._mask_cnt += 1
        return np.mean(train_loss), np.mean(train_mae), np.mean(train_mape), np.mean(train_rmse)


    def train(self):
        self._logger.info('Start training!')
        wait = 0
        min_loss = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse = self.train_batch()
            t2 = time.time()

            v1 = time.time()
            mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse = self.evaluate('val')
            v2 = time.time()

            if self._lr_scheduler is None:
                cur_lr = self._lrate
            else:
                cur_lr = self._lr_scheduler.get_last_lr()[0]
                self._lr_scheduler.step()

            # self._lr_scheduler_t.step()
            # self._lr_scheduler_s.step()

            message = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Train Time: {:.4f}s/epoch, Valid Time: {:.4f}s, LR: {:.4e}'
            self._logger.info(message.format(epoch + 1, mtrain_loss, mtrain_mae, mtrain_rmse, mtrain_mape, \
                                             mvalid_loss, mvalid_mae, mvalid_rmse, mvalid_mape, \
                                             (t2 - t1), (v2 - v1), cur_lr))

            if mvalid_loss < min_loss:
                self.save_model(self._save_path)
                self._logger.info('Val loss decrease from {:.4f} to {:.4f}'.format(min_loss, mvalid_loss))
                min_loss = mvalid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info('Early stop at epoch {}, loss = {:.6f}'.format(epoch + 1, min_loss))
                    break

        self.evaluate('test')


    def evaluate(self, mode):
        if mode == 'test':
            self._logger.info('Start test!')
            self.load_model(self._save_path)
            self._logger.info('Calculating done!')
            node_eval = self._node_test
            tood = 0 # self._tood
        else:
            node_eval = self._node_training
            tood = 0
        self.model.eval()

        preds = []
        labels = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(self._dataloader[mode + '_loader']), desc='Training'):
                pm25, feature, _, day, week = data
                X = np.repeat(pm25[:, self._seq_len:self._seq_len+1], self._horizon, axis=1)
                feature = feature[:, self._seq_len:]
                day = day[:, self._seq_len:]
                week = week[:, self._seq_len:]
                X = np.concatenate([X, feature, day, week], axis=-1)
                label = pm25[:, self._seq_len:]

                # X (b, t, n, f), label (b, t, n, 1)
                X = X[:, :, node_eval, :]
                label = label[..., node_eval, :]
                X, label = self._to_device(self._to_tensor([X, label]))
                pred = self.model(X)
                pred, label = self._inverse_transform([pred, label])
                preds.append(pred.squeeze(-1).cpu())
                labels.append(label.squeeze(-1).cpu())

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

        if mode == 'val':
            # handle the precision issue when performing inverse transform to label
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
            loss = self._loss_fn(preds, labels, mask_value).item()
            mae = masked_mae(preds, labels, mask_value).item()
            mape = masked_mape(preds, labels, mask_value).item()
            rmse = masked_rmse(preds, labels, mask_value).item()
            return loss, mae, mape, rmse

        elif mode == 'test':
            # handle the precision issue when performing inverse transform to label
            ## All 
            mask_value = torch.tensor(0)
            if labels.min() < 1:
                mask_value = labels.min()
            print('Check mask value ', mask_value)
            test_mae = []
            test_mape = []
            test_rmse = []
            ## Fix
            preds1 = preds[..., :-self._node_sood_num] if self._node_sood_num else preds
            labels1 = labels[..., :-self._node_sood_num]  if self._node_sood_num else preds
            mask_value1 = torch.tensor(0)
            if labels1.min() < 1:
                mask_value1 = labels1.min()
            print('Check mask value1', mask_value1)
            test_mae1 = []
            test_mape1 = []
            test_rmse1 = []
            print('All node number', labels.shape[-1])
            print('Fix node number', labels1.shape[-1])
            ## UnFix
            if self._node_sood_num: 
                preds2 = preds[..., -self._node_sood_num:]
                labels2 = labels[..., -self._node_sood_num:]
                mask_value2 = torch.tensor(0)
                if labels2.min() < 1:
                    mask_value2 = labels2.min()
                print('Check mask value2', mask_value2)
                test_mae2 = []
                test_mape2 = []
                test_rmse2 = []
                print('Unf node number', labels2.shape[-1])

            if self._checkall:
                for i in range(self.model.horizon):
                    res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                    test_mae.append(res[0])
                    test_mape.append(res[1])
                    test_rmse.append(res[2])

                    res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                    test_mae1.append(res1[0])
                    test_mape1.append(res1[1])
                    test_rmse1.append(res1[2])

                    if self._node_sood_num: 
                        res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                        test_mae2.append(res2[0])
                        test_mape2.append(res2[1])
                        test_rmse2.append(res2[2])

                    if (i == 5 or i == 11 or i == 23):
                        log = '{:.4f},{:.4f},{:.4f}'
                        self._logger.info(log.format(res[0], res[2], res[1]))

                        log = '{:.4f},{:.4f},{:.4f}'
                        self._logger.info(log.format(res1[0], res1[2], res1[1]))

                        if self._node_sood_num: 
                            log = '{:.4f},{:.4f},{:.4f}'
                            self._logger.info(log.format(res2[0], res2[2], res2[1]))

                        print('Clip')

                log = '{:.4f},{:.4f},{:.4f}'
                self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
                log = '{:.4f},{:.4f},{:.4f}'
                self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))
                if self._node_sood_num:
                    log = '{:.4f},{:.4f},{:.4f}'
                    self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))
            else:
                for i in range(self.model.horizon):
                    res = compute_all_metrics(preds[:,i,:], labels[:,i,:], mask_value)
                    test_mae.append(res[0])
                    test_mape.append(res[1])
                    test_rmse.append(res[2])

                    res1 = compute_all_metrics(preds1[:,i,:], labels1[:,i,:], mask_value1)
                    test_mae1.append(res1[0])
                    test_mape1.append(res1[1])
                    test_rmse1.append(res1[2])

                    if self._node_sood_num: 
                        res2 = compute_all_metrics(preds2[:,i,:], labels2[:,i,:], mask_value2)
                        test_mae2.append(res2[0])
                        test_mape2.append(res2[1])
                        test_rmse2.append(res2[2])

                    if (i == 5 or i == 11 or i == 23):
                        log = 'All Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                        self._logger.info(log.format(i + 1, res[0], res[2], res[1]))

                        log = 'Fix Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                        self._logger.info(log.format(i + 1, res1[0], res1[2], res1[1]))

                        if self._node_sood_num: 
                            log = 'UnF Horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                            self._logger.info(log.format(i + 1, res2[0], res2[2], res2[1]))

                        print('Clip')

                log = 'All Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(np.mean(test_mae), np.mean(test_rmse), np.mean(test_mape)))
                log = 'Fix Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                self._logger.info(log.format(np.mean(test_mae1), np.mean(test_rmse1), np.mean(test_mape1)))
                if self._node_sood_num:
                    log = 'UnF Average Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}'
                    self._logger.info(log.format(np.mean(test_mae2), np.mean(test_rmse2), np.mean(test_mape2)))