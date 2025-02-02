import torch
import time
import torch.nn as nn
import numpy as np
from utils_ethucy import *


class Processor():
    def __init__(self, args):
        self.args = args
        
        self.lr=self.args.learning_rate
        self.trainloader_gt = DataLoader(DataSet_bytrajec2_ethucy(args,is_train=True), batch_size=1, shuffle=True, num_workers=0)
        self.testloader_gt = DataLoader(DataSet_bytrajec2_ethucy(args,is_train=False), batch_size=1, shuffle=False, num_workers=0)
        model = import_class(args.model)
        self.net = model(args)
        if self.args.phase == "train":
            print("self.args.phase",self.args.phase)
            self.net.train()
        else:
            self.net.eval()
        self.init_lr = self.args.learning_rate
        self.step_ratio = self.args.step_ratio
        self.lr_step=self.args.lr_step
        self.set_optimizer()
        self.epoch = 0
        self.load_model()
        # self.save_model(self.epoch)
        if self.args.using_cuda:
            self.net=self.net.cuda()
        else:
            self.net=self.net.cpu()
        self.net_file = open(os.path.join(self.args.model_dir, 'net.txt'), 'a+')
        self.net_file.write(str(self.net))
        self.net_file.close()
        self.log_file_curve = open(os.path.join(self.args.model_dir, 'log_curve.txt'), 'a+')



    def save_model(self,epoch):
        model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' +\
                                   str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path,_use_new_zipfile_serialization=False)


    def load_model(self):
        if self.args.load_model > 0:
            self.args.model_save_path = self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + \
                                        str(self.args.load_model) + '.tar'
            if os.path.isfile(self.args.model_save_path):
                print('Loading checkpoint')
                checkpoint = torch.load(self.args.model_save_path,map_location={'cuda:0': 'cuda:'+str(self.args.gpu)})
                # print("self.args.model_save_path",self.args.model_save_path)
                model_epoch = checkpoint['epoch']
                self.epoch = int(model_epoch) + 1
                self.net.load_state_dict(checkpoint['state_dict'])
                print('Loaded checkpoint at epoch', model_epoch)
                for i in range(self.args.load_model):
                    self.scheduler.step()


    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        self.criterion = nn.MSELoss(reduce=False)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,\
        T_max = self.args.num_epochs, eta_min=self.args.eta_min)

    def playtest(self):
        print('Testing begin')
        print('Model:', self.args.load_model)
        test_error, test_final_error, _ = self.test_epoch(self.args.load_model)
        for k in test_error.keys():
            print('*** agent class: ', k, '***')
            print('ade=%.4f, fde=%.4f' % (test_error[k], test_final_error[k]))
        print(self.args.load_model)

    def playtrain(self):
        print('Training begin')
        test_error, test_final_error,first_erro_test=0,0,0
        for epoch in range(self.epoch, self.args.num_epochs+1):
            print('Epoch-{0} lr: {1}'.format(epoch, self.optimizer.param_groups[0]['lr']))

            train_loss = self.train_epoch(epoch)
            
            self.scheduler.step()
            if epoch == self.args.num_epochs or epoch % 50 == 0:
                self.save_model(epoch)
            if epoch == self.args.num_epochs or epoch % 10 == 0:
                test_error, test_final_error, _= self.test_epoch(epoch)
                for k in test_error.keys():
                    print('*** type: ', k, '***')
                    print('ade=%.4f, fde=%.4f' % (test_error[k], test_final_error[k]))
            print('----epoch {}'.format(epoch))
            model_path= self.args.save_dir + '/' + self.args.train_model + '/' + self.args.train_model + '_' + str(epoch) + '.tar'



    def train_epoch(self,epoch):
        """   batch_abs: the (orientated) batch
              batch_norm: the batch shifted by substracted the last position. ??? What is the impact of zeros
              shift_value: the last observed position
              seq_list: [seq_length, num_peds], mask for position with actual values at each frame for each ped
              nei_list: [seq_length, num_peds, num_peds], mask for neigbors at each frame
              nei_num: [seq_length, num_peds], neighbors at each frame for each ped
              batch_pednum: list, number of peds in each batch"""
        self.net.train()
        loss_epoch=0
        for batch, (inputs_gt, batch_split, nei_lists, edge_pair) in enumerate(self.trainloader_gt):
            start = time.time()
            
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])

            batch_abs_gt, batch_norm_gt, shift_value_gt, shift_value_max, seq_list_gt, nei_num = inputs_gt

            batch_abs_gt = batch_abs_gt[0].float()
            # batch_abs_gt = batch_abs_gt[0][:,:,1:].float()
            batch_norm_gt = batch_norm_gt[0].float()

            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split, shift_value_max
            self.net.zero_grad()

            
            L2_loss, full_pre_tra, _= self.net.forward(inputs_fw, edge_pair)

            tot_loss = L2_loss
            loss_epoch += tot_loss.item()
            tot_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optimizer.step()
            end= time.time()
            if batch%self.args.show_step == 0:
                print('train-{}/{} (epoch {}), loss = {:.5f}'.\
                format(batch,len(self.trainloader_gt), epoch, L2_loss.item()))
        train_loss_epoch = loss_epoch / len(self.trainloader_gt)

        return train_loss_epoch



    def test_epoch(self,epoch):
        self.net.eval()

        error_epoch, final_error_epoch, first_erro_epoch = {}, {}, {}
        total_time = 0
        error_cnt_epoch, final_error_cnt_epoch, first_erro_cnt_epoch = {}, {}, {}

        avg_ade, avg_fde = 0, 0
        avg_ade_cnt, avg_fde, avg_fde_cnt = 1e-5, 1e-5, 1e-5

        
        for batch, (inputs_gt, batch_split, nei_lists, edge_pair) in enumerate(self.testloader_gt):
            if batch%100 == 0:
                print('testing batch',batch,len(self.testloader_gt))
                
            inputs_gt = tuple([torch.Tensor(i) for i in inputs_gt])
            if self.args.using_cuda:
                inputs_gt = tuple([i.cuda() for i in inputs_gt])
            batch_abs_gt, batch_norm_gt, shift_value_gt, shift_value_max, seq_list_gt, nei_num = inputs_gt

            batch_abs_gt = batch_abs_gt[0].float()
            # batch_abs_gt = batch_abs_gt[0][:,:,1:].float()
            batch_norm_gt = batch_norm_gt[0].float()
            batch_class = torch.zeros_like(batch_abs_gt[-1, :, -1]).long()
            batch_class_unique = torch.unique(batch_class)
        
            inputs_fw = batch_abs_gt, batch_norm_gt, nei_lists, nei_num, batch_split, shift_value_max

            start_time = time.perf_counter()
            _, full_pre_tra, multi_traj = self.net.forward(inputs_fw, edge_pair)
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            total_time += inference_time
            multi_traj = multi_traj.transpose(1,2)
            full_pre_tra = [x * shift_value_max for x in full_pre_tra]
            multi_shift_value_max = shift_value_max.unsqueeze(0)
            multi_traj = multi_traj * multi_shift_value_max
            batch_norm_gt = batch_norm_gt * shift_value_max
            for clss in batch_class_unique:
                clss = int(clss.item())

                mask = (batch_class == clss)
                error_epoch_min, final_error_epoch_min, first_erro_epoch_min = [], [], []
                for pre_tra in full_pre_tra:
                    error, error_cnt, final_error, final_error_cnt, first_erro,first_erro_cnt = \
                    L2forTest(pre_tra[:, mask, ...], batch_norm_gt[1:, mask, :2], self.args.obs_length)
                    

                    error_epoch_min.append(error)
                    final_error_epoch_min.append(final_error)
                    first_erro_epoch_min.append(first_erro)

                error_epoch_min = min(error_epoch_min)
                final_error_epoch_min = min(final_error_epoch_min)
                first_erro_epoch_min = min(first_erro_epoch_min)

                error_epoch.setdefault(clss, 0)
                final_error_epoch.setdefault(clss, 0)
                first_erro_epoch.setdefault(clss, 0)

                error_cnt_epoch.setdefault(clss, 1e-5)
                final_error_cnt_epoch.setdefault(clss, 1e-5)
                first_erro_cnt_epoch.setdefault(clss, 1e-5)

                
                first_erro_epoch[clss] += first_erro_epoch_min
                final_error_epoch[clss] += final_error_epoch_min
                error_epoch[clss] += error_epoch_min

                error_cnt_epoch[clss] += error_cnt
                final_error_cnt_epoch[clss] += final_error_cnt
                first_erro_cnt_epoch[clss] += first_erro_cnt

                avg_ade += error_epoch_min
                avg_fde += final_error_epoch_min
                avg_ade_cnt += error_cnt
                avg_fde_cnt += final_error_cnt

        for k in error_epoch.keys():
            error_epoch[k] /= error_cnt_epoch[k]
            final_error_epoch[k] /= final_error_cnt_epoch[k]
            first_erro_epoch[k] /= first_erro_cnt_epoch[k]

        error_epoch['avg'] = avg_ade / avg_ade_cnt
        final_error_epoch['avg'] = avg_fde / avg_fde_cnt
        print('------------------',total_time/(batch*32),'--------------')
        return error_epoch, final_error_epoch, first_erro_epoch
