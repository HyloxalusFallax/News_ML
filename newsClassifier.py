import codecs
import re
import json
import math
import random
from itertools import starmap
from operator import mul

def load_data(data_directory):
	labels = []
	headlines = []
	texts = []
	with codecs.open('news_train.txt',encoding='utf-8') as file:
		for line in file:
			lower_line = line.lower()
			splitted = lower_line.split("	")
			labels.append(splitted[0])
			headlines.append(splitted[1])
			texts.append(splitted[2])
	return labels, headlines, texts

def load_target_data(data_directory):
	labels = []
	headlines = []
	texts = []
	with codecs.open('news_test.txt',encoding='utf-8') as file:
		for line in file:
			lower_line = line.lower()
			splitted = lower_line.split("	")
			headlines.append(splitted[0])
			texts.append(splitted[1])
	return headlines, texts
	
target_headlines, target_texts = load_target_data('./')

labels, headlines, texts = load_data('./')

stopwords = set([u'и', u'в', u'что', u'на', u'с', u'не', u'как', u'об', u'этом', u'по', 
				u'из', u'за', u'для', u'также', u'он', u'она', u'был', u'того', u'о',
				u'к', u'а', u'от', u'его', u'до', u'у', u'при', u'во', u'это', u'со', u'её',
				u'однако', u'их', u'но', u'ее', u'было', u'том', u'так', u'то', u'они'])

list_of_unique_labels = ['style', 'business', 'sport', 'life', 'media', 'culture', 'forces', 'science', 'travel', 'economics']			

ind_labels = []
for label in labels:
	vect = []
	for ul in list_of_unique_labels:
		if (label == ul):
			vect.append(1)
		else:
			vect.append(0)
	ind_labels.append(vect)

cleaned_headlines = []	
	
for headline in headlines:
	cleaned_headline = list(filter(None, re.split("[, \-!?:;\.«»\'\"\—\n]+", headline)));
	cleaned_headline = list(filter(lambda x: not (x in stopwords), cleaned_headline)); 
	cleaned_headlines.append(cleaned_headline)
cleaned_texts = []
for text in texts:
	cleaned_text = list(filter(None, re.split("[, \-!?:;\.«»\'\"\—\n]+", text)));
	cleaned_text = list(filter(lambda x: not (x in stopwords), cleaned_text));	
	cleaned_texts.append(cleaned_text)
dict = {}
for headline in cleaned_headlines:
	for word in headline:
		if word in dict:
			dict[word]=dict[word]+1
		else:
			dict[word] = 1
for text in cleaned_texts:
	for word in text:
		if word in dict:
			dict[word]=dict[word]+1
		else:
			dict[word] = 1
how_many_most_common = 3000
most_common = []
while (len(most_common) < how_many_most_common):
	most_common.append(("", 0))
print(len(dict))
counter = 0
for word in dict:
	if (counter % 1000 == 0):
		print(counter)
	for i in range(0, len(most_common)):
		if dict[word] > most_common[i][1]:
			most_common.insert(i, (word, dict[word]))
			most_common = most_common[:-1]
			break
	counter += 1

bow = []
print(len(cleaned_headlines))
for i in range(0, len(cleaned_headlines)):
	if (i % 1000 == 0):
		print(i)
	vect = []
	for word in most_common:
		vect.append(cleaned_headlines[i].count(word[0]) + cleaned_texts[i].count(word[0]))
	bow.append(vect)
		
with open('data.txt', 'w') as f:
    json.dump(bow, f)

cleaned_target_headlines = []
for headline in target_headlines:
	cleaned_headline = list(filter(None, re.split("[, \-!?:;\.«»\'\"\—\n]+", headline)));
	cleaned_headline = list(filter(lambda x: not (x in stopwords), cleaned_headline)); 
	cleaned_target_headlines.append(cleaned_headline)
cleaned_target_texts = []
for text in target_texts:
	cleaned_text = list(filter(None, re.split("[, \-!?:;\.«»\'\"\—\n]+", text)));
	cleaned_text = list(filter(lambda x: not (x in stopwords), cleaned_text));	
	cleaned_target_texts.append(cleaned_text)

bow = []
print(len(cleaned_target_headlines))
for i in range(0, len(cleaned_target_headlines)):
	if (i % 1000 == 0):
		print(i)
	vect = []
	for word in most_common:
		vect.append(cleaned_target_headlines[i].count(word[0]) + cleaned_target_texts[i].count(word[0]))
	bow.append(vect)

test_data = data[-1000:]
train_data = data[:-1000]

test_labels = ind_labels[-1000:]
train_labels = ind_labels[:-1000]

batch_size = 5
learning_rate = 1e-1
epochs = 3

b = []
for i in range(len(ind_labelss[0])):
	b.append(0)
	
W = []
for i in range(len(ind_data[0])):
	line = []
	for j in range(len(ind_labelss[0])):
		line.append(random.random())
	W.append(line)
	
def vect_mult_dot(a, b):
	c = []
	for i in range(len(b[0])):
		sum = 0
		for j in range(len(a)):
			sum += a[j] * b[j][i]
		c.append(sum)
	return c	
	
def vect_mult(a, b):
	if (len(a) != len(b)):
		print("Wrong dimesionality of vectors!")
		exit(0)
	c = []
	for i in range(len(a)):
		c.append(a[i] * b[i])
	return c

def matrix_subtract(a,b):
	return [[x - y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
	
def matrix_sum(a,b):
	return [[x + y for x, y in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]
	
def vect_sum(a, b):
	c = []
	for i in range(len(a)):
		c.append(a[i] + b[i])
	return c

def vect_sub(a, b):
	c = []
	for i in range(len(a)):
		c.append(a[i] - b[i])
	return c

def mult_matrix_by_scalar(m, sc):
	return [[x * sc for x in row] for row in m]

def softmax(y_linear):
	max_y = max(y_linear)
	exp = [math.exp(x-max_y) for x in y_linear]
	norms = sum(exp)
	return [x / norms for x in exp]
	
def cross_entropy(p, r):
	return -sum(vect_mult(r, list(map(lambda x: math.log(x + 1e-6), p))))

def model(X):
	y_linear = vect_sum(matmul(X, W), b)
	pred = softmax(y_linear)
	return pred

def compute_grad(p, y, X):
	dscores = list(p)
	dscores[argmax(y)] -= 1
	dw = matmul2(X, dscores)
	db = dscores
	return (dw, db)

def optimize(dw, db, lr):
	global W, b
	W = matrix_subtract(W, mult_matrix_by_scalar(dw, lr))
	b = vect_sub(b, [x * lr for x in db])

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]

def matmul(first,second):
	return [sum(starmap(mul, zip(first, col))) for col in zip(*second)]

def matmul2(first,second):
	return [[first_elem * second_elem for second_elem in second] for first_elem in first] 
	
def train_loop(features, labels):
	global learning_rate, last_improvment, best_last_accuracy
	for k in range(epochs):
		print('epoch '+ str(k))
		for i in range(0, len(features), batch_size):
			pred = model(features[i])
			dw, db = compute_grad(pred, labels[i], features[i])
			dw = mult_matrix_by_scalar(W, 0.)
			average_dw = dw
			db = list(map(lambda x: x*0, b))
			average_db = db
			if (len(labels) > i + batch_size):
				finish = i + batch_size
			else:
				finish = len(labels)
			for j in range(i, finish, 1):
				pred = model(features[j])
				dw, db = compute_grad(pred, labels[j], features[j])
				average_dw = matrix_sum(average_dw, dw)
				average_db = vect_sum(average_db, db)
			average_dw = list(map(lambda i: list(map(lambda x: x / batch_size, average_dw[i])), range(len(average_dw))))
			average_db = list(map(lambda x: x / batch_size, average_db))
			optimize(average_dw, average_db, learning_rate)
			if (i % 1000 == 0):
				print('iteration ' + str(i))
				accuracy = 0
				loss = 0
				for j in range(1000):
					ind = random.randint(0, len(labels) - 1)
					pred = model(features[ind])
					loss += cross_entropy(pred, labels[ind])
					if (argmax(pred) == argmax(labels[ind])):
						accuracy += 1
				accuracy /= 1000
				loss /= 1000
				print('accuracy: ' + str(accuracy))
				print('loss: ' + str(loss))
				print()
		learning_rate /= 2
	average_loss = 0
	for i in range(100):
		ind = random.randint(0, len(test_labels) - 1)
		pred = model(test_data[ind])
		average_loss += cross_entropy(pred, test_labels[ind])
	average_loss /= 100
	print('test loss: ' + str(average_loss))
	accuracy = 0
	for i in range(len(test_labels)):
		pred = model(test_data[i])
		if (argmax(pred) == argmax(test_labels[i])):
			accuracy += 1
	accuracy /= 1000
	print('test accuracy: ' + str(accuracy))
	target_pred = []
	for i in range(len(target_data)):
		target_pred.append(argmax(model(target_data[i])))
	with codecs.open('output.txt','w',encoding='utf8') as f:
		for e in target_pred:
			f.write(list_of_unique_labels[e] + '\n')
	
train_loop(train_data, train_labels)