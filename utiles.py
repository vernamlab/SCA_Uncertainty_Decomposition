
import random

import numpy as np

def shuffle_data(profiling_x,label_y):
    l = list(zip(profiling_x,label_y))
    random.shuffle(l)
    shuffled_x,shuffled_y = list(zip(*l))
    shuffled_x = np.array(shuffled_x)
    shuffled_y = np.array(shuffled_y)
    return (shuffled_x, shuffled_y)



def horizontal_standardization(X_profiling, X_attack):
    mn = np.repeat(np.mean(X_profiling, axis=1, keepdims=True), X_profiling.shape[1], axis=1)
    std = np.repeat(np.std(X_profiling, axis=1, keepdims=True), X_profiling.shape[1], axis=1)
    X_profiling_processed = (X_profiling - mn)/std

    mn = np.repeat(np.mean(X_attack, axis=1, keepdims=True), X_attack.shape[1], axis=1)
    std = np.repeat(np.std(X_attack, axis=1, keepdims=True), X_attack.shape[1], axis=1)
    X_attack_processed = (X_attack - mn)/std
    
    return X_profiling_processed, X_attack_processed



def rank(predictions, key, targets, ntraces, interval=10):
    ranktime = np.zeros(int(ntraces/interval))
    pred = np.zeros(256)

    idx = np.random.randint(predictions.shape[0], size=ntraces)
    
    for i, p in enumerate(idx):
        for k in range(predictions.shape[1]):
            pred[k] += predictions[p, targets[p, k]]
            
        if i % interval == 0:
            ranked = np.argsort(pred)[::-1]
            ranktime[int(i/interval)] = list(ranked).index(key)
            
    return ranktime



def modrank(predictions, key, targets, ntraces, interval=1):
    ranktime = np.zeros(int(ntraces/interval))
    pred = np.zeros(256)

    idx = range(predictions.shape[0])
    
    for i , p in enumerate(idx):
        for k in range(predictions.shape[1]):
            pred[k] = predictions[p, targets[p, k]]
            
        if i % interval == 0:
            ranked = np.argsort(pred)[::-1]
            ranktime[int(i/interval)] = list(ranked).index(key)
            
    return ranktime