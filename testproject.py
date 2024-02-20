from experiment1 import *
from experiment2 import *
import ManualStrategy as ml

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "dpatel426"  # Change this to your user ID

if __name__ == "__main__":
    np.random.seed(903465461)
    manual = ml.ManualStrategy()
    manual.runman()
    runexp1()
    runexp2()