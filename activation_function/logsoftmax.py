# log_softmax :
# logsoftmax = Xi - max(X) - log(sum(e^(Xi - max(X))))
# sum of e^logsoftmax = 1


# better way:
def log_softmax(scores: list) -> np.ndarray:
  scores = scores - np.max(scores)
  return scores - np.log(sum(np.exp(scores)))

# my-v0:
def log_softmax(scores: list) -> np.ndarray:
	scores = np.array(scores)
	max_arr = np.full(len(scores),max(scores))
	max_arr = scores - max_arr
	log_sum = np.log(sum([np.exp(ele) for ele in max_arr]))
	
	return [ele - log_sum for ele in max_arr]
