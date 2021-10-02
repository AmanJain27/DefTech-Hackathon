from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score


# define a function that returns new updated class attributes

def diffprivacy(epsilons, dp, X_train, y_train, X_test, y_test, y_arr, y, x):
	i = 0
	survival_status_updated = []
	acc_2d = []
	for d in range(2,12,2):
		i += 1
		acc = []
		survival_status_for_each_depth = []
		for e in epsilons:

			decision = dp.DecisionTreeClassifier(max_depth=d, e=e, s=1, min_samples_leaf=5)
			decision.fit(X_train, y_train)
			a_n = decision.score(X_test, y_test)
			survival_for_each_epsilon =  []
			#print(f"survival status for epsilon = {e} and depth = {d}")
			for z in range(len(np.array(y))):
				survival_for_each_epsilon.append(y_arr[np.argmax(decision.tree_.predict(np.array(x, dtype=np.float32))[z])])
			#print(survival_for_each_epsilon)
			survival_status_for_each_depth.append(survival_for_each_epsilon)


			a_t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=1).fit(X_train,y_train).predict(X_test)
			a = decision.score(X_test, y_test)
			acc_sc = accuracy_score(y_test, decision.predict(X_test))
			acc_sc_t = accuracy_score(y_test, a_t)
			acc.append(acc_sc)
		acc_2d.append(acc)
		survival_status_updated.append(survival_status_for_each_depth)
		print(f"loop = {i}")

		#print(d, decision.tree_.weighted_n_node_samples.shape, decision.tree_.n_node_samples[decision.tree_.children_left != -1], decision.tree_.node_count)


		#plt.plot(epsilons, acc, color=color[d])
	print(survival_status_updated)
	return acc_2d, survival_status_updated
	# plt.legend(color_l)
	# plt.show()

# K cross validation for Differential Privacy



# Plot the average accuracies in the graph
def plot_avg(s_l, ep, k,color, color_l, plt):

	for i in range(len(s_l)):
		plt.plot(ep, s_l[i], color=color[(i+1)*2])
		color_l.append((i+1)*2)
	plt.title(f"for K = {k}")
	plt.legend(color_l)
	plt.show()


def cross_val_average_and_plot(hb, epsilons, dp, y, x,y_arr,color, color_l, plt):
	from sklearn.model_selection import KFold
	sk_main = []
	for k in range(5, 11):
		acc = []

		kf = KFold(k, shuffle=True, random_state=1)
		for train, test in kf.split(hb):
			X_train = hb.iloc[train, :-1]
			y_train = hb.iloc[train, -1]
			# print(train ,test)
			X_test = hb.iloc[test, :-1]
			y_test = hb.iloc[test, -1]
			acc.append(diffprivacy(epsilons, dp, X_train, y_train, X_test, y_test, y_arr, y, x)[0])
		acc_re = np.array(acc)
		print(f"acc = {acc}")
		print(acc_re.shape)
		# print(acc_re)
		s_main = []
		#for depth
		for i in range(acc_re.shape[1]):
			# s_i = []
			s_l = [] # for each depth append accuracies for different epsilons
			#for epsilons
			for j in range(acc_re.shape[2]):
				s_j = 0 # initiate s_j to add all the accuracies for same depth and same epsilon with different k values
				#for k
				for m in range(acc_re.shape[0]): #different k values
					s_j += acc[m][i][j] # for eg for k = 0, 1st epsilon, 1st depth then for k = 1 and rest condition same

				s_l.append(s_j / k)

				print(k, end=" ")
			s_main.append(s_l)
			print(s_main)
		sk_main.append(s_main)

		#plot_avg(s_main, epsilons, k,color, color_l, plt)

	print(np.array(sk_main).shape)
	sk_main_np_arr = np.array(sk_main)
	# return the average accuracies for a particular depth and particular epsilon for different folds
	avg_accuraciesForEachDepth_withDiffKFoldValues = []
	for x_d in range(sk_main_np_arr.shape[1]): # for particular depth
		particular_d_acc_list = []
		for y_e in range(sk_main_np_arr.shape[2]): # for particular epsilon
			sum_of_acc_epsilons = 0
			for z_k in range(sk_main_np_arr.shape[0]): # for particular fold value k = z
				sum_of_acc_epsilons += sk_main[z_k][x_d][y_e]

			avg_of_acc_epsilons = sum_of_acc_epsilons / sk_main_np_arr.shape[0]

			particular_d_acc_list.append(avg_of_acc_epsilons)

		#
		avg_accuraciesForEachDepth_withDiffKFoldValues.append(particular_d_acc_list)
		print(np.array(avg_accuraciesForEachDepth_withDiffKFoldValues).shape)


	print(avg_accuraciesForEachDepth_withDiffKFoldValues)
