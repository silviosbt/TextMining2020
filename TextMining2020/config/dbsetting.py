'''
Created on 07 lug 2020

@author: Utente
'''
import redis

REDISDB='127.0.0.1'

def setDB():
    db = redis.StrictRedis(host=REDISDB, port=6379, db=0)
    db.flushall()
    return db