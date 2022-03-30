#!/usr/bin/python3

from numpy import angle
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep

factory = PiGPIOFactory()
servo = Servo(26,pin_factory=factory)
val = -1

def deg2pwm(f_angle):
	f_angle -= -3.0 #offset
	pwm_raspi = 0.002273 * f_angle# AFTER CALIBRATION WITH CIRCLE PATH
	return pwm_raspi

if __name__ == '__main__':
	angle_ref = 0.0
	angle_increment = 1.0

	# value_ref = 0.0
	# value_increment = 0.01

	try:
		while True:
			val = deg2pwm(angle_ref)
			if angle_ref == 32:
				print(f'big val is {val}')
			# print(f'val is {val}')
			# print(f'angle ref is {angle_ref}')
			# print('-----')
			servo.value = val
			sleep(0.05)
			# value_ref += value_increment

			angle_ref += angle_increment
			if angle_ref > 32.0:
				angle_increment = -1.0
			if angle_ref < -32.0:
				angle_increment = 1.0 
			
			
			# value_ref += value_increment
			# if value_ref > 0.5:
			# 	value_increment = -0.01
			# if value_ref < -0.5:
			# 	value_increment = 0.01
			
	except KeyboardInterrupt:
		print("Program stopped")


      