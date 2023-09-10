import math
import numpy as np
import pandas as pd


class NaiveBayes:
    def __init__(self, data_path):
        self.data_path = data_path
        self.dataset = None
        self.position_samples: list = list()
        self.negative_samples: list = list()
        self.column_idx: dict = dict()
        self.attribute_kinds: dict = dict()
        self.X, self.y = None, None
        self.gmm_cache: dict = dict()  # 记忆一些参数的连续数据,以避免重复计算
        self.p_map: dict = dict()
        self._initialized = False

    def get_dataset(self):
        self.dataset = pd.read_csv(self.data_path, delimiter=",")
        del self.dataset['编号']
        self.dataset["密度"] = self.dataset["密度"].apply(lambda row: round(row, 3))
        self.dataset["含糖率"] = self.dataset["含糖率"].apply(lambda row: round(row, 3))

        self.position_samples, self.negative_samples = list(self.dataset[self.dataset["好瓜"] == "是"].index), \
                                                       list(self.dataset[self.dataset["好瓜"] == "否"].index),

        self.X = self.dataset.values[:, :-1]
        self.y = self.dataset.values[:, -1]
        self.column_idx = {self.dataset.columns[i]: i for i in range(len(self.dataset.columns))}
        self.attribute_kinds = {i: len(set(self.X[:, i])) for i in range(self.X.shape[1])}
        self._initialized = True

    def bayes_main(self, column_id, attribute, C):  # P(colName=attribute|C) P(色泽=青绿|是)
        if not self._initialized:
            self.get_dataset()
        if (column_id, attribute, C) in self.p_map:
            return self.p_map[(column_id, attribute, C)]
        pred = self.position_samples if C == '是' else self.negative_samples
        ans = 0
        if column_id >= 6:
            if (column_id, C) in self.gmm_cache:
                curPara = self.gmm_cache[(column_id, C)]
                mean = curPara[0]
                std = curPara[1]
            else:
                curData = self.X[pred, column_id]
                mean = curData.mean()
                std = curData.std()

                self.gmm_cache[(column_id, C)] = (mean, std)
            ans = 1 / (math.sqrt(math.pi * 2) * std) * math.exp((-(attribute - mean) ** 2) / (2 * std * std))
        else:
            for i in pred:
                if self.X[i, column_id] == attribute:
                    ans += 1
            # 拉普拉斯修正
            ans = (ans + 1) / (len(pred) + self.attribute_kinds[column_id])
        self.p_map[(column_id, attribute, C)] = ans

        return ans

    def predict(self, single):
        # 这边为二分类，所以存在正负样本，概率均等，所以分子+1，分母+2，概率均为0.5
        ansYes = math.log2((len(self.position_samples) + 1) / (len(self.y) + 2))
        ansNo = math.log2((len(self.negative_samples) + 1) / (len(self.y) + 2))
        # 书上是连乘，但在实践中要把“连乘”通过取对数的方式转化为“连加”以避免数值下溢
        for i in range(len(single)):
            ansYes += math.log2(self.bayes_main(i, single[i], '是'))
            ansNo += math.log2(self.bayes_main(i, single[i], '否'))
        return '是' if ansYes > ansNo else '否'

    def predict_main(self):
        if not self._initialized:
            self.get_dataset()
        return [self.predict(item) for item in self.X]

    def get_confusion_matrix(self, predicts):
        confusion_matrix = np.zeros((2, 2))
        for i in range(len(self.y)):
            if predicts[i] == self.y[i]:
                if self.y[i] == '是':
                    confusion_matrix[0, 0] += 1
                else:
                    confusion_matrix[1, 1] += 1
            else:
                if self.y[i] == '否':
                    confusion_matrix[0, 1] += 1
                else:
                    confusion_matrix[1, 0] += 1
        return confusion_matrix


def f1_score(confusion_matrix):
    precision_score = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    recall_score = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    print("precision_score:", precision_score)
    print("recall_score:", recall_score)

    return 2*precision_score * recall_score / (precision_score + recall_score)


if __name__ == '__main__':
    data_path = "./watermelon_3.csv"
    NB = NaiveBayes(data_path)
    result = NB.predict_main()
    f1 = f1_score(NB.get_confusion_matrix(result))
    print("f1:", f1)
