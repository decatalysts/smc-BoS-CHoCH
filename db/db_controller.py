from pymongo import MongoClient

class DBController:
    def __init__(self, url, collection):
        print(url)
        self.client = MongoClient(url)
        self.db = self.client[collection]
        print('mongodb connected')

    def insert(self, target, data):
        if target == 'bos_uptrend':
            collection = self.db['bos_uptrend']
        elif target == 'bos_downtrend':
            collection = self.db['bos_downtrend']
        elif target == 'choch_uptrend':
            collection = self.db['choch_uptrend']
        elif target == 'choch_downtrend':
            collection = self.db['choch_downtrend']
        else:
            return None

        return collection.insert_one(data)

    def find_by_condition(self, target, condition):
        if target == 'bos_uptrend':
            collection = self.db['bos_uptrend']
        elif target == 'bos_downtrend':
            collection = self.db['bos_downtrend']
        elif target == 'choch_uptrend':
            collection = self.db['choch_uptrend']
        elif target == 'choch_downtrend':
            collection = self.db['choch_downtrend']
        else:
            return None

        results = collection.find(condition)
        results = list(results)

        if results:
            return results
        else:
            return None

    def close(self):
        print('mongodb closed')
        self.client.close()