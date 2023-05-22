import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
import emoji
import matplotlib.pyplot as plt
from sklearn import svm

# implementing a class for KNN classifier algorithm
class kNN:
    def __init__(self, k, data_file):
        self.k = k
        self.drug_response_data = pd.read_csv(data_file, sep='\t')
        self.drug_response_data.drop(index=self.drug_response_data.index[5:], inplace=True)
        self.expression_data = pd.read_csv(data_file, sep='\t')
        self.expression_data.drop([0,1,2,3,4], inplace=True)
    
    def pearson_similarity(self, x, y):
        r, _ = pearsonr(x,y)
        return r
    
    def classifier(self, drug_idx, cell):
        drug_data = self.drug_response_data.loc[drug_idx]
        valid_idx = drug_data.dropna(axis=0).index
        drug_data = drug_data.dropna(axis=0)
        valid_expression_data = self.expression_data[valid_idx]
        input_profile = self.expression_data[cell].values
        if cell not in valid_expression_data.columns:
            print("NA values")
            return -1
        similarity_scores = valid_expression_data.iloc[:, 1:].apply(lambda x: self.pearson_similarity(x.values, input_profile), axis=0)
        # remove the cell score by adding +1 to k and removing the 1.0 at the top
        nearest_neighbors = similarity_scores.sort_values(ascending=False)[:self.k+1].index
        neighbor_scores = drug_data.loc[nearest_neighbors[1:]]
        num_sensitive = neighbor_scores[neighbor_scores == 1].count()
        score = num_sensitive / self.k
        return score
    
    def classifier2(self, drug_idx, cell):
        drug_data = self.drug_response_data.loc[drug_idx]
        valid_idx = drug_data.dropna(axis=0).index
        drug_data = drug_data.dropna(axis=0)
        valid_expression_data = self.expression_data[valid_idx]
        input_profile = self.expression_data[cell].values
        if cell not in valid_expression_data.columns:
            print("NA values")
            return -1
        similarity_scores = valid_expression_data.iloc[:, 1:].apply(lambda x: self.pearson_similarity(x.values, input_profile), axis=0)
        # remove the cell score by adding +1 to k and removing the 1.0 at the top
        nearest_neighbors = similarity_scores.sort_values(ascending=False)[:self.k+1].index
        neighbor_scores = drug_data.loc[nearest_neighbors[1:]]
        score = 0
        for yi in nearest_neighbors[1:]:
            if neighbor_scores[yi] == 1:
                sign = 1
            else:
                sign = -1
            corr = self.pearson_similarity(valid_expression_data[yi].values, input_profile)
            score += sign * corr
        return score
    
    def loocv(self, cell):
        loo = LeaveOneOut()
        scores = []
        for train_idx, test_idx in loo.split(self.drug_response_data):
            drug_idx = test_idx[0]
            score = self.classifier(drug_idx, cell)
            if score == -1:
                continue
            scores.append(score)
        return np.mean(scores)
    
    def roc(self, scores, drug_idx, all_drugs, scores2, scores3, scores4, scores5):
        labels = self.drug_response_data.iloc[drug_idx, 1:].dropna(axis=0).values
        labels = labels.astype(int)
        fpr, tpr, thresholds = roc_curve(labels, list(scores.values()))
        roc_auc = auc(fpr, tpr)
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve for drug 1 (AUC = %0.2f)' % roc_auc)
        plt.title('ROC Curve for Drug %d' % (drug_idx+1))
        # EXTRA CREDIT
        # Compute ROC curve for SVM classifier
        # only do this if we're plotting one drug
        if all_drugs == False:
            valid_idx = self.drug_response_data.iloc[drug_idx, 1:].dropna(axis=0).index
            valid_expression_data = self.expression_data[valid_idx].dropna(axis=0).T
            x = valid_expression_data.values
            clf = svm.SVC(kernel='linear', probability=True)
            clf.fit(x, labels)
            svm_scores = clf.predict_proba(x)[:,1]
            svm_fpr, svm_tpr, _ = roc_curve(labels, svm_scores)
            svm_auc = auc(svm_fpr, svm_tpr)
            plt.plot(svm_fpr, svm_tpr, color='blue', lw=2, linestyle='--', label='SVM classifier (AUC = %0.2f)' % svm_auc)
        # Random
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random classifier (AUC = %0.2f)' % 0.5)
        # plot all drugs
        if all_drugs == True:
            labels2 = self.drug_response_data.iloc[1, 1:].dropna(axis=0).values
            labels2 = labels2.astype(int)
            fpr2, tpr2, thresholds = roc_curve(labels2, list(scores2.values()))
            roc_auc2 = auc(fpr2, tpr2)
            plt.plot(fpr2, tpr2, color='purple', lw=2, label='ROC curve for drug 2 (AUC = %0.2f)' % roc_auc2)
            
            labels3 = self.drug_response_data.iloc[2, 1:].dropna(axis=0).values
            labels3 = labels3.astype(int)
            fpr3, tpr3, thresholds = roc_curve(labels3, list(scores3.values()))
            roc_auc3 = auc(fpr3, tpr3)
            plt.plot(fpr3, tpr3, color='green', lw=2, label='ROC curve for drug 3 (AUC = %0.2f)' % roc_auc3)

            labels4 = self.drug_response_data.iloc[3, 1:].dropna(axis=0).values
            labels4 = labels4.astype(int)
            fpr4, tpr4, thresholds = roc_curve(labels4, list(scores4.values()))
            roc_auc4 = auc(fpr4, tpr4)
            plt.plot(fpr4, tpr4, color='yellow', lw=2, label='ROC curve for drug 4 (AUC = %0.2f)' % roc_auc4)

            labels5 = self.drug_response_data.iloc[4, 1:].dropna(axis=0).values
            labels5 = labels5.astype(int)
            fpr5, tpr5, thresholds = roc_curve(labels5, list(scores5.values()))
            roc_auc5 = auc(fpr5, tpr5)
            plt.plot(fpr5, tpr5, color='pink', lw=2, label='ROC curve for drug 5 (AUC = %0.2f)' % roc_auc5)
            plt.title('ROC Curve for all Drugs')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

def helper(test):
    scores = {}
    scores2 = {}
    scores3 = {}
    scores4 = {}
    scores5 = {}
    for c in test.expression_data.columns[1:]:
        score = test.classifier(0, c) # drug0
        score2 = test.classifier(1, c) # drug1
        score3 = test.classifier(2, c) # drug2
        score4 = test.classifier(3, c) # drug3
        score5 = test.classifier(4, c) # drug4
        if score != -1: # if NaN
            scores[c] = score
        if score2 != -1:
            scores2[c] = score2
        if score3 != -1:
            scores3[c] = score3
        if score4 != -1:
            scores4[c] = score4
        if score5 != -1:
            scores5[c] = score5
    return scores, scores2, scores3, scores4, scores5

def helper2(test):
    scores = {}
    scores2 = {}
    scores3 = {}
    scores4 = {}
    scores5 = {}
    for c in test.expression_data.columns[1:]:
        score = test.classifier2(0, c) # drug0
        score2 = test.classifier2(1, c) # drug1
        score3 = test.classifier2(2, c) # drug2
        score4 = test.classifier2(3, c) # drug3
        score5 = test.classifier2(4, c) # drug4
        if score != -1: # if NaN
            scores[c] = score
        if score2 != -1:
            scores2[c] = score2
        if score3 != -1:
            scores3[c] = score3
        if score4 != -1:
            scores4[c] = score4
        if score5 != -1:
            scores5[c] = score5
    return scores, scores2, scores3, scores4, scores5

# TESTING STARTS HERE
test = kNN(5, "DREAM_data.txt")
drug_idx = 0
cell = "ZR751"

print("________Q1-kNN________")
score = test.classifier(drug_idx, cell)
print(score)

print("________Q2-LOOCV________")
accuracy = test.loocv(cell)
print(accuracy)

print("________Q2-EACH CELL LINE________")
# SCORES FOR Everolimus(mTOR)
# SCORES FOR Disulfiram(ALDH2)
# SCORES FOR Methylglyoxol(Pyruvate)
# SCORES FOR Mebendazole(Tubulin)
# SCORES FOR 4-HC(DNA alkylator)
scores, scores2, scores3, scores4, scores5 = helper(test)
print(scores)
print(scores2)
print(scores3)
print(scores4)
print(scores5)

print("________Q2-ROC________")
test.roc(scores, 0, False, scores2, scores3, scores4, scores5)
test.roc(scores2, 1, False, scores, scores3, scores4, scores5)
test.roc(scores3, 2, False, scores2, scores, scores4, scores5)
test.roc(scores4, 3, False, scores2, scores3, scores, scores5)
test.roc(scores5, 4, False, scores2, scores3, scores4, scores)

print("________Q3-CHANGING K VALUES________")
# with k = 5
test.roc(scores, 0, True, scores2, scores3, scores4, scores5)
# with k = 3
test3 = kNN(3, "DREAM_data.txt")
scores, scores2, scores3, scores4, scores5 = helper(test3)
test3.roc(scores, 0, True, scores2, scores3, scores4, scores5)
# with k = 7
test7 = kNN(7, "DREAM_data.txt")
scores, scores2, scores3, scores4, scores5 = helper(test7)
test7.roc(scores, 0, True, scores2, scores3, scores4, scores5)

print("________Q3-DIFFERENT WEIGHTED SCORE________")
scores, scores2, scores3, scores4, scores5 = helper2(test)
test.roc(scores, 0, False, scores2, scores3, scores4, scores5)
test.roc(scores2, 1, False, scores, scores3, scores4, scores5)
test.roc(scores3, 2, False, scores2, scores, scores4, scores5)
test.roc(scores4, 3, False, scores2, scores3, scores, scores5)
test.roc(scores5, 4, False, scores2, scores3, scores4, scores)
test.roc(scores, 0, True, scores2, scores3, scores4, scores5)

print("________DONE________")
print(emoji.emojize(":partying_face:") + emoji.emojize(":partying_face:"))