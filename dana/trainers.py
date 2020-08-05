import time
import numpy as np
import tensorflow as tf

class ModelTrainer():
    def __init__(self, model, loss_fn, optimizer, data_class_weights,
                 num_epochs=512, batch_size=128, patience=64,
                 W_combinations=None, H_combinations=None, n_batch_per_train_setp=1,
                 W_combinations_validation=None,
                 H_combinations_validation=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_class_weights = data_class_weights
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.W_combinations = W_combinations
        self.H_combinations = H_combinations        
        self.n_batch_per_train_setp = n_batch_per_train_setp
        self.W_combinations_validation = W_combinations_validation
        self.H_combinations_validation = H_combinations_validation        

    
    def evaluate_model(self, X_val, Y_val, W_comb=None, H_comb=None):
        if not W_comb:
            W_comb = X_val.shape[1]
        if not H_comb:
            H_comb = list(range(X_val.shape[2]))
        val_accuracy = tf.keras.metrics.Accuracy()
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.batch(self.batch_size)        
        for (X, Y) in val_dataset:
            X = X.numpy()                                    
            X = X[:,:,H_comb,:]                     
            X = tf.image.resize(X, (W_comb, len(H_comb)))
            logits = self.model(X, training=False)
            prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
            val_accuracy(prediction, Y)        
        return val_accuracy.result()
    
    def standard_training(self, train_dataset, epoch, n_iterations):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()           
        for i in range(n_iterations):                
            rnd_order_H = np.random.permutation(len(self.H_combinations))
            rnd_order_W = np.random.permutation(len(self.W_combinations))
            with tf.GradientTape() as tape:
                X, Y = next(train_dataset)
                X = X.numpy()                        
                sample_weight = [self.data_class_weights[y] for y in Y.numpy()]
                rnd_H = self.H_combinations[rnd_order_H[0]]
                X = X[:,:,rnd_H,:] 
                rnd_W = self.W_combinations[rnd_order_W[0]]
                X = tf.image.resize(X, (rnd_W, len(rnd_H)))                        
                logits =  self.model(X)               
                loss_value = self.loss_fn(Y, logits, sample_weight)
            gradients = tape.gradient(loss_value, self.model.trainable_weights)
            gradients = [g*(1.-epoch/self.num_epochs) for g in gradients]                                                       
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))                        
            epoch_loss_avg.update_state(loss_value)
        iter_time = time.time() - start_time
        return epoch_loss_avg, iter_time
    
    def weight_avg_training(self, train_dataset, epoch, n_iterations):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()           
        for i in range(n_iterations):                
            rnd_order_H = np.random.permutation(len(self.H_combinations))
            rnd_order_W = np.random.permutation(len(self.W_combinations))
            n_samples = 0.
            old_weights = self.model.get_weights()
            accu_weights = None 
            for j in range(self.n_batch_per_train_setp):   
                with tf.GradientTape() as tape:                               
                    try:
                        X, Y = next(train_dataset)
                    except:
                        break  
                    X = X.numpy()                        
                    sample_weight = [self.data_class_weights[y] for y in Y.numpy()]
                    rnd_H = self.H_combinations[rnd_order_H[j%len(rnd_order_H)]]
                    X = X[:,:,rnd_H,:] 
                    rnd_W = self.W_combinations[rnd_order_W[j%len(rnd_order_W)]]
                    X = tf.image.resize(X, (rnd_W, len(rnd_H)))                
                    logits =  self.model(X)               
                    loss_value = self.loss_fn(Y, logits, sample_weight)                                                       
                gradients = tape.gradient(loss_value, self.model.trainable_weights) 
                gradients = [g*(1.-epoch/self.num_epochs) for g in gradients]                                                       
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))                
                if accu_weights:            
                    accu_weights = [a + b for a, b in zip(accu_weights, self.model.get_weights())]
                else:            
                    accu_weights = self.model.get_weights()
                self.model.set_weights(old_weights)
                epoch_loss_avg.update_state(loss_value)
                n_samples = n_samples + 1.
            accu_weights = [w*(1./n_samples) for w in accu_weights]
            self.model.set_weights(accu_weights)            
        iter_time = time.time() - start_time        
        return epoch_loss_avg, iter_time


    def reptile_training(self, train_dataset, epoch, n_iterations):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()           
        for i in range(n_iterations):                
            rnd_order_H = np.random.permutation(len(self.H_combinations))
            rnd_order_W = np.random.permutation(len(self.W_combinations))
            n_samples = 0.
            old_weights = self.model.get_weights()
            new_weights = None 
            for j in range(self.n_batch_per_train_setp):   
                with tf.GradientTape() as tape:
                    try:
                        X, Y = next(train_dataset)
                    except:
                        break                               
                    X = X.numpy()                        
                    sample_weight = [self.data_class_weights[y] for y in Y.numpy()]
                    rnd_H = self.H_combinations[rnd_order_H[j%len(rnd_order_H)]]
                    X = X[:,:,rnd_H,:] 
                    rnd_W = self.W_combinations[rnd_order_W[j%len(rnd_order_W)]]
                    X = tf.image.resize(X, (rnd_W, len(rnd_H)))                
                    logits =  self.model(X)               
                    loss_value = self.loss_fn(Y, logits, sample_weight)                                 
                gradients = tape.gradient(loss_value, self.model.trainable_weights)                                                        
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))                
                if new_weights:            
                    new_weights = [a + b for a, b in zip(new_weights, self.model.get_weights())]
                else:            
                    new_weights = self.model.get_weights()
                self.model.set_weights(old_weights)
                epoch_loss_avg.update_state(loss_value)
                n_samples = n_samples + 1.
            reptile_grads = [((new_weights[i]/float(n_samples))-old_weights[i])*((1.-epoch/self.num_epochs)*.1) for i in range(len(old_weights))]
            self.optimizer.apply_gradients(zip(reptile_grads, self.model.trainable_weights))            
        iter_time = time.time() - start_time        
        return epoch_loss_avg, iter_time

    def dat_training(self, train_dataset, epoch, n_iterations):
        start_time = time.time()
        epoch_loss_avg = tf.keras.metrics.Mean()           
        for i in range(n_iterations):    
            rnd_order_H = np.random.permutation(len(self.H_combinations))
            rnd_order_W = np.random.permutation(len(self.W_combinations))
            n_samples = 0.
            with tf.GradientTape() as tape:
                accum_loss = tf.Variable(0.)
                for j in range(self.n_batch_per_train_setp):
                    try:
                        X, Y = next(train_dataset)
                    except:
                        break
                    X = X.numpy()
                    sample_weight = [self.data_class_weights[y] for y in Y.numpy()]
                    rnd_H = self.H_combinations[rnd_order_H[j%len(rnd_order_H)]]
                    X = X[:,:,rnd_H,:] 
                    rnd_W = self.W_combinations[rnd_order_W[j%len(rnd_order_W)]]
                    X = tf.image.resize(X, (rnd_W, len(rnd_H)))    
                    logits =  self.model(X)               
                    accum_loss = accum_loss + self.loss_fn(Y, logits, sample_weight)
                    n_samples = n_samples + 1.
            gradients = tape.gradient(accum_loss, self.model.trainable_weights)
            gradients = [g*(1./n_samples)*(1.-epoch/self.num_epochs) for g in gradients]
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
            epoch_loss_avg.update_state(accum_loss*(1./n_samples))
        iter_time = time.time() - start_time
        return epoch_loss_avg, iter_time
    

    def train_model(self, X_train, Y_train, X_val=None, Y_val=None, method=None, save_dir=None, verbose=10):
        train_loss_results = []                    
        if not self.W_combinations:
            self.W_combinations = [list(range(X_train.shape[1]))]
        if not self.H_combinations:
            self.H_combinations = [list(range(X_train.shape[2]))]
        training_time_log = []
        best_accuracy = 0.
        n_no_improve_on_val = 0
        best_accuracy_record = np.array(0)
        for epoch in range(self.num_epochs):                               
            ######## Dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
            train_dataset = iter(train_dataset.shuffle(len(X_train)).batch(self.batch_size))
            ######## Training ########
            if method == "standard":
                n_iterations_per_epoch = len(X_train)//(self.batch_size)
                epoch_loss_avg, iter_time = self.standard_training(train_dataset, epoch, n_iterations_per_epoch)
            elif method == "weight_avg":
                n_iterations_per_epoch = len(X_train)//(self.batch_size*self.n_batch_per_train_setp)+1                
                epoch_loss_avg, iter_time = self.weight_avg_training(train_dataset, epoch, n_iterations_per_epoch)
            elif method == "reptile":
                n_iterations_per_epoch = len(X_train)//(self.batch_size*self.n_batch_per_train_setp)+1
                epoch_loss_avg, iter_time = self.reptile_training(train_dataset, epoch, n_iterations_per_epoch)
            elif method == "dat":
                n_iterations_per_epoch = (len(X_train)//(self.batch_size*self.n_batch_per_train_setp))+1
                epoch_loss_avg, iter_time = self.dat_training(train_dataset, epoch, n_iterations_per_epoch)
            elif method == "dat2":
                n_iterations_per_epoch = (len(X_train)//(self.batch_size*self.n_batch_per_train_setp))+1
                epoch_loss_avg, iter_time = self.dat_training_2(train_dataset, epoch, n_iterations_per_epoch)
            else:
                raise Exception("Method <{}> is not defiend!".format(method))  
            training_time_log.append(iter_time)
            train_loss_results.append(epoch_loss_avg.result())
            #if epoch % verbose == 0:                
            #    print("Epoch {:03d}, Loss: {:.5f},Iteration Time:{:.4f}s".format(
            #                                   epoch,epoch_loss_avg.result(),avg_training_time/(epoch+1)), end = " | ") 
            ######## Validation ########
            if X_val is not None:
                accuracy_record = np.zeros((len(self.W_combinations_validation),len(self.H_combinations_validation)))
                for w in range(len(self.W_combinations_validation)):
                    for h in range(len(self.H_combinations_validation)):                    
                        accuracy_record[w,h] = self.evaluate_model(X_val, Y_val,
                                                            self.W_combinations_validation[w],
                                                            self.H_combinations_validation[h])
                avg_val_acc = np.mean(accuracy_record)
                if epoch % verbose == 0:
                    print("Epoch {:03d}, Best Acc: {:.3f}, Max Acc: {:.3f}".format(epoch, best_accuracy, best_accuracy_record.max()))

                ####### Saving the Best Model ########
                if avg_val_acc > best_accuracy:            
                    best_accuracy = avg_val_acc
                    best_accuracy_record = accuracy_record
                    # print("*** Epoch {:03d}, Best Model is Saved on {:.3%},Iteration Time:{:.4f}s, \n {}".format(epoch, best_accuracy,
                    #                                                                                 avg_training_time/(epoch+1),
                    #                                                                                 best_accuracy_record))
                    n_no_improve_on_val = 0
                    if save_dir:
                        self.model.save_weights(save_dir)
                else:
                    n_no_improve_on_val += 1                
                if n_no_improve_on_val > self.patience:
                    break                            
        avg_training_time = np.mean(training_time_log)
        std_training_time = np.std(training_time_log)
        print("Terminated on epoch {:03d} . Average Training Time {:.3f}s\n".format(epoch,avg_training_time))
        if X_val is not None:
            print("Best Accuraccy: {:.3f} \n {})".format(best_accuracy,best_accuracy_record))
        return self.model, best_accuracy_record, [avg_training_time, std_training_time]