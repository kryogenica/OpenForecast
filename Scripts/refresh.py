from random import randint
import time
import os
def refresher(seconds):
   while True:
      mainDir = os.path.dirname(__file__)
      filePath = os.path.join(mainDir, 'dummy.py')
      with open(filePath, 'w') as f:
         f.write(f"Content = '{randint(0, 1000)}'")
      time.sleep(seconds)
while(1):
   refresher(10) # 20s for example