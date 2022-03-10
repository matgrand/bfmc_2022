    
from turtle import delay
import serial, string
from time import sleep

if __name__ == '__main__':   
    output = " "
    ser = serial.Serial('/dev/ttyACM0', 256000, 8, 'N', 1, timeout=1)
    while True:
        print("----")
        while output != "":
            output = ser.readline()
            print(output)
            sleep(0.1)
        output = " "