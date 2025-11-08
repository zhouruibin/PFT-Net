import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder_sort import Res101Encodernew

import ot


class OT_Attn_assem(nn.Module):
    def __init__(self, impl='pot-uot-l2', ot_reg=0.1, ot_tau=0.5) -> None:
        super(OT_Attn_assem, self).__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", self.impl)

    def normalize_feature(self, x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)

        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)

            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()

            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(),
                               M=cost_map_detach / cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow
            dist = torch.sum(dist)
            return flow, dist

        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(
                ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)

            if weight1.dim() == 1:
                weight1 = weight1.unsqueeze(0)  # 变成 (1, D)
            if weight2.dim() == 1:
                weight2 = weight2.unsqueeze(0)  # 变成 (1, D)

            self.cost_map = torch.cdist(weight1, weight2) ** 2  # (N, M)

            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach / cost_map_detach.max()

            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b,
                                                           M=M_cost.double(), reg=self.ot_reg, reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()

            dist = self.cost_map * flow  # (N, M)
            dist = torch.sum(dist)  # (1,) float
            return flow, dist

        else:
            raise NotImplementedError

    def forward(self, x, y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze()
        y = y.squeeze()

        x = self.normalize_feature(x)
        y = self.normalize_feature(y)

        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3", dataset='SABS'):
        super().__init__()

        # Encoder
        self.encoder = Res101Encodernew(replace_stride_with_dilation=[True, True, False],
                                        pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()

        self.num_layers = 2
        # ot
        ot_reg = 0.1
        ot_tau = 0.5
        # ot_impl = "pot-sinkhorn-l2"
        ot_impl = "pot-uot-l2"

        self.coattn = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

    def forward(self, supp_imgs, supp_mask, qry_imgs, qry_mask=None, train=False, t_loss_scaler=1, n_iters=20):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """

        self.n_ways = len(supp_imgs)  # 1
        self.n_shots = len(supp_imgs[0])  # 1
        self.n_queries = len(qry_imgs)  # 1
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]  # 1
        supp_bs = supp_imgs[0][0].shape[0]  # 1
        img_size = supp_imgs[0][0].shape[-2:]
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        # encoder output
        img_fts, tao, t1 = self.encoder(imgs_concat)

        supp_fts = img_fts[:self.n_ways * self.n_shots * supp_bs].view(  # B x Wa x Sh x C x H' x W'
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts.shape[-2:])

        qry_fts = img_fts[self.n_ways * self.n_shots * supp_bs:].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts.shape[-2:])

        ##### Get threshold #######
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.thresh_pred = [self.t for _ in range(self.n_ways)]
        ##### Get rate #######
        self.t1 = t1[self.n_ways * self.n_shots * supp_bs:]  # t for query features
        self.t1 = torch.sigmoid(self.t1)

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        mse_loss = torch.zeros(1).to(self.device)
        outputs0 = []
        outputs1 = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            if supp_mask[epi][0].sum() == 0:
                # supp_mask = supp_mask.view(1,1,1,512,*img_fts.shape[-2:])
                supp_fts_ = [[[self.getFeatures(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                             range(len(supp_fts))]

                fg_prototypes = torch.zeros(1, 512).cuda()
                qry_fts_clone = qry_fts[epi].clone()

                qry_pred = torch.stack(  # (1, 512, 64, 64) (1, 512) (1, 1)
                    [self.getPred(qry_fts_clone, fg_prototypes.view(1, 512), self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)

                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

                preds = qry_pred_up
                preds = torch.cat((1.0 - preds, preds), dim=1)
                outputs0.append(preds)

            else:
                bs, c, h, w = qry_fts[epi].shape
                features_supp = [[self.get_features(supp_fts[[epi], way, shot], supp_mask[[epi], way, shot])
                                  for shot in range(self.n_shots)] for way in range(self.n_ways)]

                prototypes = self.get_all_prototypes(features_supp)

                fg_prototypes = [self.get_mean_prototype(prototypes[way]) for way in
                                 range(self.n_ways)]
                select_protos = [self.sort_select_fts(prototypes[way], fg_prototypes[way])
                                 for way in range(self.n_ways)]

                qry_fts_clone = qry_fts[epi].clone()  # [N, c, h_q, w_q]

                fg_prototypes2 = [self.get_mean_prototype(select_protos[way]) for way in
                                  range(self.n_ways)]
                A_fg = [F.cosine_similarity(qry_fts_clone, fg_prototypes2[way][..., None, None]) for way in
                        range(self.n_ways)] # list[(1,h_q,w_q)]
                A_fg = [A_fg[way].expand(1, 1, h, w) for way in range(self.n_ways)]
                qry_fts_clone_fg = qry_fts_clone * A_fg[0]
                qry_fts_clone_fg_flat = qry_fts_clone_fg.flatten(-2).permute(2, 0, 1)# (hw,n,c)

                A_coattn, _ = self.coattn(select_protos[0].unsqueeze(1), qry_fts_clone_fg_flat)

                fg_protos_new = torch.mm(A_coattn.T, qry_fts_clone_fg_flat.squeeze(1))
                global_proto = self.get_mean_prototype(select_protos[0])

                qry_pred = torch.stack(  # (1, 512, 64, 64) (1, 512) (1, 1)
                    [self.getPred(qry_fts_clone, global_proto.view(1, 512), self.thresh_pred[way])
                     for way in range(self.n_ways)], dim=1)
                qry_pred_up = F.interpolate(qry_pred, size=img_size, mode='bilinear', align_corners=True)

                pred_0 = qry_pred_up  # (1,1,256,256)
                pred_0 = torch.cat((1.0 - pred_0, pred_0), dim=1)
                outputs0.append(pred_0)

        out_0 = torch.stack(outputs0, dim=1)
        out_0 = out_0.view(-1, *out_0.shape[2:])
        return out_0

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        if len(fts.shape) == 3:  # 如果是 [C, H, W], 添加 batch 维度
            fts = fts.unsqueeze(0)
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        sim = sim.mean(0, keepdim=True)
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))  # ([1, 64, 64])

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        if len(fts.shape) == 3:
            fts = fts.unsqueeze(0)
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear', align_corners=True)

        # masked fg features
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        return masked_fts

    def get_features(self, features, mask):

        features_trans = F.interpolate(features, size=mask.shape[-2:], mode='bilinear',
                                       align_corners=True)
        features_trans = features_trans.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0)
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features_trans = features_trans[indx]

        return features_trans

    def get_all_prototypes(self, fg_fts):

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        prototypes = [sum([shot for shot in way]) / n_shots for way in fg_fts]
        return prototypes

    def get_mean_prototype(self, prototypes):

        return torch.mean(prototypes, dim=0).unsqueeze(0)

    def sort_select_fts(self, fts, prototype):  # (n,512) (1,512)
        for i in range(self.num_layers):
            sim = F.cosine_similarity(fts, prototype, dim=1)  # (n,1)
            rate = (sim.max() + sim.mean() + sim.min()) / 3
            index = sim >= rate.squeeze(0)
            if index.sum() != 0:
                fts = fts[index]
            # fts = self.MHA(fts.unsqueeze(0), fts.unsqueeze(0), fts.unsqueeze(0))
            # fts = self.MLP(fts).squeeze(0)
            prototype = fts.mean(0, keepdim=True)

        return fts




