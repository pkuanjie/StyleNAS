import torch

def whiten_and_color(cF,sF):
    cFSize = cF.size()
    c_mean = torch.mean(cF,1) # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    c_gram = torch.mm(cF,cF.t()).div(cFSize[1]-1)
    c_choose = torch.eye(cFSize[0]).double().cuda()

    contentConv = (1 - c_choose) * c_gram + 2.0 * c_choose * c_gram + 0.5 * c_gram
    c_u,c_e,c_v = torch.svd(contentConv,some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.000001:
            k_c = i
            break

    sFSize = sF.size()
    s_mean = torch.mean(sF,1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF,sF.t()).div(sFSize[1]-1) + 0.0 * torch.eye(sFSize[0]).double().cuda()
    s_u,s_e,s_v = torch.svd(styleConv,some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.000001:
            k_s = i
            break
    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:,0:k_c],torch.diag(c_d))
    step2 = torch.mm(step1,(c_v[:,0:k_c].t()))
    whiten_cF = torch.mm(step2,cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:,0:k_s],torch.diag(s_d)),(s_v[:,0:k_s].t())),whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    return targetFeature

def transform(cF,sF,alpha):
    cF = cF.double()
    sF = sF.double()
    if len(cF.size()) == 4:
        cF = cF[0]
    if len(sF.size()) == 4:
        sF = sF[0]
    C,W,H = cF.size(0),cF.size(1),cF.size(2)
    _,W1,H1 = sF.size(0),sF.size(1),sF.size(2)
    cFView = cF.view(C,-1)
    sFView = sF.view(C,-1)

    targetFeature = whiten_and_color(cFView,sFView)
    targetFeature = targetFeature.view_as(cF)
    csF = alpha * targetFeature + (1.0 - alpha) * cF
    csF = csF.float().unsqueeze(0)
    return csF
