from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

class TreeExplainer:
    def __init__(self, clf):
        self.clf = clf
        self.dic = dict() # needed afterwards
        self.n_cat = -1

    # computing banzhaf values
    def pred_tree(self, coalition, row, node=0):
        left_node = self.clf.tree_.children_left[node]
        right_node = self.clf.tree_.children_right[node]
        is_leaf = left_node == right_node

        if is_leaf:
            return self.clf.tree_.value[node].squeeze()

        feature = row.index[self.clf.tree_.feature[node]]
        if feature in coalition:
            if row.loc[feature] <= self.clf.tree_.threshold[node]:
                # go left
                return self.pred_tree(coalition, row, node=left_node)
            # go right
            return self.pred_tree(coalition, row, node=right_node)

        # take weighted average of left and right
        wl = self.clf.tree_.n_node_samples[left_node] / self.clf.tree_.n_node_samples[node]
        wr = self.clf.tree_.n_node_samples[right_node] / self.clf.tree_.n_node_samples[node]
        value = wl * self.pred_tree(coalition, row, node=left_node)
        value += wr * self.pred_tree(coalition, row, node=right_node)
        return value

    def make_value_function(self, row, col):
        def value(c):
            marginal_gain = self.pred_tree(c + [col], row) - self.pred_tree(c, row)
            return marginal_gain
        return value

    @staticmethod
    def make_coalitions(row, col):
        rest = [x for x in row.index if x != col]
        for i in range(len(rest) + 1):
            for x in combinations(rest, i):
                yield list(x)

    def compute_banzhaf(self, row, col, category=None):
        # when using a classifier we compute more than one banzhaf value
        v = self.make_value_function(row, col)
        result = sum([v(coal) / (2 ** (len(row) - 1)) for coal in self.make_coalitions(row, col)])
        if type(result) == np.ndarray and category == None:
            print("Enter a valid category number!")
            return 
        elif type(result) == np.ndarray and 0 <= category and category < len(result):
            return result[category]
        elif type(result) == np.float64:
            return result
            
    
    @property
    def expected_value(self):
        return self.clf.tree_.value[0].squeeze().item()
    
    @property
    def n_categories(self):
        if self.n_cat == -1:
            self.n_cat = self.clf.n_classes_
        return self.n_cat

    def get_category(self, num):
        return self.clf.classes_[num]
    # computing interaction values

    @staticmethod
    def make_coalitions_interactions(row, col1, col2):
        rest = [x for x in row.index if x != col1 and x != col2]
        for i in range(len(rest) + 1):
            for x in combinations(rest, i):
                yield list(x)

    def make_value_function_interactions(self, row, col1, col2):
        def value(c):
            marginal_gain = self.pred_tree(c + [col1, col2], row)  - self.pred_tree(c + [col1], row) - self.pred_tree(c + [col2], row) + self.pred_tree(c, row)
            return marginal_gain
        return value

    def compute_banzhaf_interactions(self, row, col1, col2, category=None):
        v = self.make_value_function_interactions(row, col1, col2)
        result = sum([v(coal) / (2 ** (len(row) - 2)) for coal in self.make_coalitions_interactions(row, col1, col2)])
        if type(result) == np.ndarray and category == None:
            print("Enter a valid category number!")
            return 
        elif type(result) == np.ndarray and 0 <= category and category < len(result):
            return result[category]
        elif type(result) == np.float64:
            return result

    # plotting

    def feature_importance(self, dataframe, row=None, category=None):
        if row != None:
            assert type(row) == int and row <= dataframe.shape[0] and row >= 0, "Please enter a valid row"
            my_banzhaf = [self.compute_banzhaf(dataframe[row:row + 1].T.squeeze(), x, category) for x in dataframe.columns]
            # Create a data frame
            df2 = pd.DataFrame ({
                    'features': list(dataframe.columns),
                    'banzhaf_values': my_banzhaf
            })

            color = []
            for i in list(my_banzhaf):
                if i >= 0:
                    color.append('dodgerblue')
                else:
                    color.append('crimson')
            # Create horizontal bars
            plt.figure(figsize=(15,15))
            plot = plt.barh(y=df2.features, width=df2.banzhaf_values, color=color)
            def autolabel(rects):
                for rect in rects:
                    width = rect.get_width()
                    plt.text(0.5*rect.get_width(), rect.get_y()+0.5*rect.get_height(),
                            f'{width:.2f}',
                            ha='center', va='center', fontsize = 13)
            autolabel(plot)
            plt.xlabel("Banzhaf Value")
            plt.ylabel("Feature")
            if category:
                plt.title(f"Feature importance for the observation : {row}, Category = {category}")
            else:
                plt.title(f"Feature importance for the observation : {row}")
            plt.show()
        else:
            n = len(list(dataframe.columns)) #number of features
            m = dataframe.shape[0] #number of rows
            liste = [[self.compute_banzhaf(dataframe[i:i+1].T.squeeze(), x, category) for x in dataframe.columns] for i in range(dataframe.shape[0]) ]
            for x in list(dataframe.columns):
                self.dic[x] = []
            for i in range(m):
                for j in range(n):
                    self.dic[list(dataframe.columns)[j]].append(liste[i][j])
            # Create a data frame
            my_banzhaf = [sum(list(map(abs,tmp))) / len(tmp) for tmp in self.dic.values() ]
            df2 = pd.DataFrame ({
                    'features': list(dataframe.columns),
                    'banzhaf_values': my_banzhaf,
            })

            color = []
            for i in list(my_banzhaf):
                if i >= 0:
                    color.append('dodgerblue')
                else:
                    color.append('crimson')
            # Create horizontal bars
            plt.figure(figsize=(15,15))
            plot = plt.barh(y=df2.features, width=df2.banzhaf_values, color=color)
            def autolabel(rects):
                for rect in rects:
                    width = rect.get_width()
                    plt.text(0.5*rect.get_width(), rect.get_y()+0.5*rect.get_height(),
                            f'{width:.2f}',
                            ha='center', va='center', fontsize = 13)
            autolabel(plot)
            plt.xlabel("Mean(|Banzhaf Value|)")
            plt.ylabel("Feature")
            if category != None and 0 <= category and category < self.n_categories:
                plt.title(f"Overall Feature Importance, Category = {category}")
            else:
                plt.title("Overall Feature Importance")
            plt.show()
        self.dic = {}
            

    def summary_plot(self, dataframe, category=None):
        # plt summary plot 
        n = len(list(dataframe.columns)) #number of features
        m = dataframe.shape[0] #number of rows
        liste = [[self.compute_banzhaf(dataframe[i:i+1].T.squeeze(), x, category) for x in dataframe.columns] for i in range(dataframe.shape[0]) ]
        for x in list(dataframe.columns):
            self.dic[x] = []
        for i in range(m):
            for j in range(n):
                self.dic[list(dataframe.columns)[j]].append(liste[i][j])
        plasma_reversed = matplotlib.cm.get_cmap('plasma_r')
        plt.figure(figsize=(15,15))
        maxes = []
        mins = []
        for key, data in self.dic.items():
            plt.scatter(data, [key for i in range(m)], c=data, cmap = plasma_reversed)
            maxes.append(max(data))
            mins.append(min(data))
        maX = max(maxes)
        miN = min(mins)
        cbar = plt.colorbar()
        plt.clim(miN, maX) 
        cbar.set_ticks([miN, (miN + maX) / 2, maX])
        cbar.set_ticklabels(['low', 'medium', 'high'])
        if category != None and 0 <= category and category < self.n_categories:
            plt.title(f"Summary Plot, Category = {category}")
        else:
            plt.title("Summary Plot")
        plt.show()
        self.dic = {}
    
    def dependence_plot(self, dataframe, column, interaction = None, category=None):
        assert column in dataframe.columns, 'Please specify a valid column.'
        assert (interaction in dataframe.columns) or (interaction == None), 'Please specify a valid column or set interaction to None.'
        if interaction:
            # color scheme 
            color_intensity = [] # == interaction value
            n = len(list(dataframe.columns)) #number of features
            m = dataframe.shape[0] #number of rows
            for k in range(m):
                interaction_matrix = [[0 for i in range(n)] for j in range(n)]
                for i,col1 in enumerate(list(dataframe.columns)):
                    if col1 == column:
                        a = i
                    for j,col2 in enumerate(list(dataframe.columns)):
                        if col2 == interaction:
                            b = j
                        interaction_matrix[i][j] = self.compute_banzhaf_interactions(dataframe[k:k+1].T.squeeze(), col1, col2, category)
                color_intensity.append(interaction_matrix[a][b])
            plt.figure(figsize = (15, 15))
            plt.scatter(dataframe[column], self.dic[column], s = 10, c = color_intensity, cmap='viridis')
            cbar = plt.colorbar()  
            cbar.ax.set_ylabel(interaction)

        else:
            plt.figure(figsize = (15, 15))
            plt.scatter(dataframe[column], self.dic[column], s = 10)
        plt.xlabel(column)
        plt.ylabel(f"Banzhaf Value for {column}")
        if category != None and 0 <= category and category < self.n_categories:
            plt.title(f"Dependence plot for {column}, interaction = {interaction}, category = {category}")
        else:
            plt.title(f"Dependence plot for {column}, interaction = {interaction}")
        plt.show()

class ForestExplainer:
    """
        We compute the banzhaf values as explained in the report, then we use the past plotting 
        functions with the adequate dictionary of values.
        After implementing it, we realized that our implementation wasn't practical because it takes
        too much time to plot and compute the banzhaf values (we loop through all the tree then take
        the mean)
        We'll try to reimplement it
    """
    pass