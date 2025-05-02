from torchmetrics import Metric
import torch

# [TODO] Implement this!
# class MyF1Score(Metric):
#     pass

class MyF1Score(Metric):
    def __init__(self, num_classes: int, eps: float = 1e-9):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

        self.add_state('tp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds_idx = torch.argmax(preds, dim=1).view(-1)
        target = target.view(-1)

        cm = torch.bincount(
            preds_idx * self.num_classes + target,
            minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes).float()

        tp = cm.diag()               
        fp = cm.sum(dim=1) - tp      
        fn = cm.sum(dim=0) - tp    

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall    = self.tp / (self.tp + self.fn + self.eps)
        f1        = 2 * precision * recall / (precision + recall + self.eps)
        return f1




class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total',   default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        preds_idx = torch.argmax(preds, dim=1)

        if preds_idx.shape != target.shape:
            preds_idx = preds_idx.view_as(target)

        correct = torch.sum(preds_idx == target)

        self.correct += correct
        self.total   += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

# class MyAccuracy(Metric):
#     def __init__(self):
#         super().__init__()
#         self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
#         self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

#     def update(self, preds, target):
#         # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence


#         # [TODO] check if preds and target have equal shape


#         # [TODO] Cound the number of correct prediction


#         # Accumulate to self.correct
#         self.correct += correct

#         # Count the number of elements in target
#         self.total += target.numel()

#     def compute(self):
#         return self.correct.float() / self.total.float()
