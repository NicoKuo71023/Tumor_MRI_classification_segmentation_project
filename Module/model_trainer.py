
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class SimpleTrainer:
    """A model trainer for classification models."""

    def __init__(self, model, loss_fn, optimizer):
        self.model = self.to_gpu(model)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def to_gpu(self, obj, device="cuda:0"):
        """將張量或模型送至GPU。"""
        return obj.to(device)
    

    def train_step(self, dataloader):
        """訓練一個epoch。"""
        self.model.train()  # 設定成訓練模式 (有些網路層訓練時和推理時有不同的行為, 例如Dropout)
        for iteration, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = self.to_gpu(batch_x)
            batch_y = self.to_gpu(batch_y)

            self.optimizer.zero_grad()  # 請優化器清空模型內所有權重的梯度

            pred_y = self.model(batch_x)  # 正向傳遞得到模型預測

            loss_value = self.loss_fn(
                pred_y, batch_y
            )  # 將正確(Ground Truth)和預測(Prediction) 標籤做比較，得到誤差
            loss_value.backward()  # 倒傳遞得到誤差

            self.optimizer.step()  # 讓優化器更新模型權重乙次

        return self.test_step(dataloader, mode="train")

    def test_step(self, dataloader, mode="test"):
        """結束一個epoch的訓練後，測試模型表現。"""
        self.model.eval()  # 設定成推理模式

        size = len(dataloader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for iteration, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = self.to_gpu(batch_x)
                batch_y = self.to_gpu(batch_y)

                pred_y = self.model(batch_x)

                test_loss += self.loss_fn(pred_y, batch_y).item()
                correct += (
                    (pred_y.argmax(axis=1) == batch_y).type(torch.float).sum().item()
                )

        test_loss /= size
        correct /= size

        print(
            "{}_loss={:.4f}, {}_accuracy={:.2f}".format(mode, test_loss, mode, correct)
        )
        return correct

    def fit(self, dataloader_train, dataloader_test, num_epochs):
        # 開始訓練
        metrics = {"train_acc": [], "test_acc": []}
        for epoch in range(num_epochs):
            print(epoch)
            train_acc = self.train_step(dataloader_train)
            test_acc = self.test_step(dataloader_test)

            metrics["train_acc"].append(train_acc)
            metrics["test_acc"].append(test_acc)

        return metrics

    def __call__(self, x):
        self.model.eval()  # 啟動推理 (Inference) 模式
        return self.model(x)  # 執行推理