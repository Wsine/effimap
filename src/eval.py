from tqdm import tqdm
import torch
import torch.nn as nn
from vendor.MSRN.util import AveragePrecisionMeter

from dataset import load_dataloader
from model import get_device, load_model
from arguments import parser
from utils import *


dispatcher = AttrDispatcher('task')


@torch.no_grad()
@dispatcher.register('multilabels')
def evaluate_multilabels_task(_, model, testloader, device):
    model.eval()

    criterion = nn.MultiLabelSoftMarginLoss().to(device)

    running_loss = 0
    ap_meter = AveragePrecisionMeter(difficult_examples=True)
    for batch_idx, (inputs, targets) in enumerate(pbar := tqdm(testloader, 'Eval')):
        targets_gt = targets.clone()
        targets[targets == -1] = 0
        imgs, inps = inputs
        imgs, inps, targets = imgs.to(device), inps.to(device), targets.to(device)

        with torch.cuda.amp.autocast():  # type: ignore
            outputs, group_loss = model(imgs, inps)
            loss = criterion(outputs, targets) + group_loss

        running_loss += loss.item()
        ap_meter.add(outputs.data, targets_gt)

        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix(loss=avg_loss)

    mAP = 100 * ap_meter.value().mean()  # type: ignore
    OP, OR, OF1, CP, CR, CF1, ce, lrap, lrl = ap_meter.overall()  # type: ignore
    OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, ce_k, lrap_k, lrl_k = ap_meter.overall_topk(3)
    print('Test:\tmAP {map:.3f}'.format(map=mAP))
    print('OP: {OP:.4f}\tOR: {OR:.4f}\tOF1: {OF1:.4f}\tCP: {CP:.4f}\tCR: {CR:.4f}\t'
          'CF1: {CF1:.4f}\tce: {ce:.4f}\tlrap: {lrap:.4f}\tlrl: {lrl:.4f}'.format(
              OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1, ce=ce, lrap=lrap, lrl=lrl))
    print('OP_3: {OP:.4f}\tOR_3: {OR:.4f}\tOF1_3: {OF1:.4f}\tCP_3: {CP:.4f}\tCR_3: {CR:.4f}\t'
          'CF1_3: {CF1:.4f}\tce_3: {ce_3:.4f}\tlrap_3: {lrap_3:.4f}\tlrl_3: {lrl_3:.4f}'.format(
              OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k, ce_3=ce_k, lrap_3=lrap_k, lrl_3=lrl_k))


def main():
    opt = parser.parse_args()
    print(opt)

    device = get_device(opt)
    model = load_model(opt).to(device)
    testloader = load_dataloader(opt, split='val+test')

    dispatcher(opt, model, testloader, device)


if __name__ == '__main__':
    main()
