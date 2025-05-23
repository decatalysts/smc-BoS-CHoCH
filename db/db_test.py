from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['smc']

collection = db['bos_downtrend']
results = collection.find({
                'product': 'CY',
                'break_datetime': {'$gte': '2025-05-20 10:00:00'}
            })
print(list(results))

client.close()