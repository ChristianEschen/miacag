import torch
from miacag.model_utils.eval_utils import getListOfLogits


def test_getListOflogtis():

    logits = [
        [torch.tensor([0.9, 0.1]), torch.tensor([0.2, 0.8])],
        [torch.tensor([0.5, 0.5]), torch.tensor([0.4, 0.6])]]
    output_liste_logits = [
        torch.tensor([[0.9, 0.1], [0.5, 0.5]]),
        torch.tensor([[0.2, 0.8], [0.4, 0.6]])
        ]

    label_names = ['label1', 'label2']
    list_logits = getListOfLogits(logits, label_names, 2)
    for count, _ in enumerate(list_logits):
        assert torch.equal(list_logits[count], output_liste_logits[count])


def test_getListOflogtisMultiClass():

    logits = [
        [torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.5, -0.8]])],
        [torch.tensor([[0.4, 0.23], [0.1, 0.2]])]]
    output_liste_logits = [torch.tensor(
        [[0.9, 0.1], [0.2, 0.8], [0.5, -0.8],
         [0.4, 0.23], [0.1, 0.2]])]
        

    label_names = ['label1', 'label2']
    list_logits = getListOfLogits(logits)
    for count, _ in enumerate(list_logits):
        assert torch.equal(list_logits[count], output_liste_logits[count])



if __name__ == '__main__':
    test_getListOflogtisMultiClass()
    #test_getListOflogtis()
