import unittest
import pandas as pd
from main import getGainRatios

class TestGainRatios(unittest.TestCase):
    def test_calculate_gain_ratios(self):

        expected_gain_ratios = {
            'Outlook': 	0.156,  
            'Temperature': 0.019,  
            'Humidity': 0.152,  
            'Wind': 0.049  
        }
        
        data = pd.DataFrame({
            'Outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
            'Temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
            'Humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
            'Wind': [False, True, False, False, False, True, True, False, False, False, True, True, False, True],
            'Play': ['NO', 'NO', 'YES', 'YES', 'YES', 'NO', 'YES', 'NO', 'YES', 'YES', 'YES', 'YES', 'YES', 'NO']
        })

        data['Outlook'] = pd.factorize(data['Outlook'])[0]  
        data['Temperature'] = pd.factorize(data['Temperature'])[0]
        data['Humidity'] = pd.factorize(data['Humidity'])[0]
        data['Play'] = pd.factorize(data['Play'])[0]

        gainRatios = getGainRatios(data, 'Play')

        for atrb in expected_gain_ratios:
            #self.assertIsInstance(gr, float)
            self.assertAlmostEqual(gainRatios[atrb], expected_gain_ratios[atrb], places=2)

if __name__ == '__main__':
    unittest.main()