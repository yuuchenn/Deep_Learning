def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
  # confusion_matrix: 
  #                 predict positive, predict negative
  # actual positive [[    TP,              FN],
  # actual negative  [    FP,              TN]]
  
	confusion_matrix = [[0,0],[0,0]]
	for i in range(len(actual)):
		if actual[i] == 1 and predicted[i] == 1:
			confusion_matrix[0][0]+=1
		elif actual[i] == 1 and predicted[i] == 0:
			confusion_matrix[0][1]+=1
		elif actual[i] == 0 and predicted[i] == 1:
			confusion_matrix[1][0]+=1
		else :
			confusion_matrix[1][1]+=1
	
  # accuracy: (TP+TN)/(TP+FN+FP+TN), the ratio of correct predictions
	accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1])/len(actual)
  
  # precision: TP/(TP+FP), the ratio of correct predictions when predict positive 
	precision = confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])

  # recall: TP/(TP+FN), the ratio of correct predictions in the actual positive
  recall = confusion_matrix[0][0]/sum(confusion_matrix[0])

  # f1: 2/(1/precision+1/recall) = 2 * precision * recall /(precision + recall)
	f1 = 2 * precision * recall /(precision + recall)
  
  # specificity: TN/(FP+TN), the ratio of correct predictions in the actual negative
	specificity = confusion_matrix[1][1]/sum(confusion_matrix[1])

  # negativePredictive: TN/(FN+TN), the ratio of correct predictions when predict negative 
	negativePredictive = confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
