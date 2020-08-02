'''
Created on 05 lug 2020

@author: Silvio Fabi
'''
import time, json
import main

def runModel(model, queue, target_names):
    QUEUE=queue
    # continually pool for new  text to classify
    while True:
        # attempt to grab a json of documents from the database
        #queue = db.lrange(MLP_QUEUE, 0, BATCH_SIZE - 1)
        q = main.db.lpop(QUEUE)
        if q is not None:
            #for q in queue:
            print('q value:',q)
            #deserialize the object for get the data
            q=json.loads(q.decode("utf-8"))
            #print(type(qq))
            print("-------------------------------")
            documents=q['documents']
            codice=q['codice']
            # set document ID (docIDs)
            docIDs=q['id']
            i=0
            output=[]
            y_pred=model.predict(documents)
            for x_t in y_pred:
                #prediction = model_svm.predict(x_t)
                predicted_label=target_names[x_t]
                print('ID Document: {} -- Predicted label: {}'.format(codice[i], predicted_label))
                r = {"did": codice[i], "testo": documents[i], "classe": str(predicted_label)}
                output.append(r)
                #print('Identificativo documento: {} -- Actual label: {} -- Predicted label: {}'.format(code[i], labels[i], target_names[predicted_label]))
                i += 1
            #r = {"did": cod, "classe": labels}        
            #print(output)
            print(type(output))
            print(json.dumps(output, indent=4))
            # store the output predictions in the database, using
            # the docIDs as the key so we can fetch the results
            main.db.set(docIDs, json.dumps(output))
            # sleep for a small amount
            time.sleep(main.SERVER_SLEEP)
            # remove the docIDs from our queue
            #db.ltrim(MLP_QUEUE, len(docIDs), -1)
            #db.lpop(docIDs)
            main.db.delete(docIDs)
