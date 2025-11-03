import unittest
import json
from app import app

class FlaskTest(unittest.TestCase):

    # Check if home route works
    def test_home(self):
        tester = app.test_client(self)
        response = tester.get('/')
        self.assertEqual(response.status_code, 200)

    # Check prediction route
    def test_predict(self):
        tester = app.test_client(self)
        sample_input = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = tester.post('/predict',
                               data=json.dumps(sample_input),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.get_json())

if __name__ == '__main__':
    unittest.main()
