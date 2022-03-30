#!/usr/bin/python3
from time import sleep
from automobile_data import Automobile_Data     # I/O car manager



if __name__ == '__main__':

    # init the car flow of data
    car = Automobile_Data(trig_control=True, trig_cam=True, trig_gps=False, trig_bno=True, trig_enc=True, trig_sonar=True)

    try:
        car.drive_speed(0)
        car.drive_angle(0)
        sleep(1)

    except rospy.ROSInterruptException:
        print('inside interrupt exeption')
        pass
      