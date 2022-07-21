import unittest
from flaskr.training import LSTMPrediction
from flaskr.finance import CompanyStock

class TrainingTest(unittest.TestCase):
    test_company = 'AAPL'
    test_type = 'Close'
    company = CompanyStock(test_company)
    training = LSTMPrediction(company.get_item(test_type))

    def test_something(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
