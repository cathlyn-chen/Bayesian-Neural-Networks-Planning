import numpy as np
import torch
from torch.autograd import Variable


def test2(net, test_loader, hp):
    net.eval()
    with torch.no_grad():
        correct = 0

        for data, target in test_loader:
            data, target = data.to(hp.device), target.to(hp.device)

            outputs = torch.zeros(hp.test_size, hp.classes).to(hp.device)
            outputs = net(data, sample=False)

            pred = torch.max(outputs, 1)[1].data.numpy()[0]
            # print(pred, target.data.numpy())
            correct += (pred == target.data.numpy()[0])

        acc = correct / hp.test_size

        return acc


def test1(net, test_data, test_label, hp):
    with torch.no_grad():
        X_test = Variable(torch.Tensor(test_data))

        # Predict with softmax
        pred_test = net(X_test)
        _, out_test = torch.max(pred_test, 1)

        out_test = out_test.data.numpy()
        test_label = np.array(test_label)

        # Evaluate
        acc = np.count_nonzero(
            [out_test[i] == test_label[i]
             for i in range(out_test.shape[0])]) / out_test.shape[0]
        err = 1 - acc

        return acc, err


def test_ensemble(net, test_loader, hp):
    net.eval()
    correct = 0
    corrects = np.zeros(hp.test_samples + 1, dtype=int)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(hp.device), target.to(hp.device)
            outputs = torch.zeros(hp.test_samples + 1, hp.test_batch_size,
                                  hp.classes).to((hp.device))
            for i in range(hp.test_samples):
                outputs[i] = net(data, sample=True)
            outputs[hp.test_samples] = net(data, sample=False)
            output = outputs.mean(0)
            preds = preds = outputs.max(2, keepdim=True)[1]
            pred = output.max(1,
                              keepdim=True)[1]  # index of max log-probability
            corrects += preds.eq(
                target.view_as(pred)).sum(dim=1).squeeze().cpu().numpy()
            correct += pred.eq(target.view_as(pred)).sum().item()
    for index, num in enumerate(corrects):
        if index < hp.test_samples:
            print('Component {} Accuracy: {}%'.format(index, num /
                                                      hp.test_size * 100))
        else:
            print('Posterior Mean Accuracy: {}%'.format(num / hp.test_size *
                                                        100))
    print('Test Ensemble Accuracy: {}%'.format(correct / hp.test_size * 100))
