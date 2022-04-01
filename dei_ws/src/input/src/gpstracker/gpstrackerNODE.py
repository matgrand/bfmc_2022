#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

from multiprocessing import Pipe
from locsys import LocalisationSystem

import time

import rospy

from utils.msg import localisation

class gpstrackerNODE():
    
    def __init__(self):
        beacon = 12345
        id = 4
        serverpublickey = 'publickey_server_test.pem'
        
        self.gpsStR, gpsStS = Pipe(duplex = False)
        self.LocalisationSystem = LocalisationSystem(id, beacon, serverpublickey, gpsStS)
        
        rospy.init_node('gpstrackerNODE', anonymous=False)
        
        # BNO publisher object
        self.GPS_publisher = rospy.Publisher("/automobile/localisation", localisation, queue_size=1)
        
        
    #================================ RUN ========================================
    def run(self):
        rospy.loginfo("starting gpstrackerNODE")
        self.LocalisationSystem.start()
        time.sleep(5)
        self._getting()
    
    #================================ GETTING ========================================
    def _getting(self):
        while not rospy.is_shutdown():
            try:
                coora = self.gpsStR.recv()

                loc=localisation()
                
                loc.timestamp = coora['timestamp']
                loc.posA = coora['coor'][0].real
                loc.posB = coora['coor'][0].imag
                loc.rotA = coora['coor'][1].real
                loc.rotB = coora['coor'][1].imag
                    
                self.GPS_publisher.publish(loc)
            except KeyboardInterrupt:
                break

        self.LocalisationSystem.stop()
        self.LocalisationSystem.join()
            

if __name__ == '__main__':
    gptrkNODE = gpstrackerNODE()

    gptrkNODE.run()
